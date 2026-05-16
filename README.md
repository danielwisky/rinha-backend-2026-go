# rinha-backend-2026-go

Solução para a **Rinha de Backend 2026** — serviço de detecção de fraude em tempo real usando k-NN com índice **IVF (inverted-file) quantizado em int8**, *mmap* em processo. Foco em throughput máximo dentro de 1 vCPU / 350 MB RAM.

## O que faz?

Recebe eventos de transação (valor, parcelamento, dados do cliente, estabelecimento e terminal) e retorna um `fraud_score` com a decisão de aprovação ou recusa. A decisão é baseada nos 5 vizinhos mais próximos no espaço vetorial de transações históricas rotuladas.

## Arquitetura

```
                        ┌──► api1:8080 ──┐
Client ──► nginx:9999 ──┤                │   (índice IVF mmap'd
                        │                │    em cada processo;
                        └──► api2:8080 ──┘    páginas compartilhadas
                                              pelo page cache do kernel)
```

Não há serviço de store separado — o índice é carregado in-process via `mmap` direto do arquivo `ivf.bin` embutido na imagem. Como `api1` e `api2` rodam a partir da mesma imagem (mesmo inode), o Linux compartilha as páginas físicas do arquivo entre os dois containers.

| Componente | Papel | Limite de recursos |
|---|---|---|
| **nginx** | Load balancer round-robin entre api1 e api2, keep-alive 256 | 0.05 CPU / 10 MB |
| **api1 / api2** | Vetorização + busca k-NN no IVF mmap | 0.475 CPU / 170 MB cada |
| **Total** |  | 1.00 CPU / 350 MB |

> Memória: as 170 MB por api são virtuais. O índice (~43 MB em disco, int8) entra na RSS de cada processo via mmap mas as **páginas físicas são compartilhadas** entre api1 e api2.

### API (`cmd/api`)

- HTTP server [fasthttp](https://github.com/valyala/fasthttp), JSON com [sonic](https://github.com/bytedance/sonic)
- Converte o payload de entrada em um vetor de 14 dimensões (`internal/vectorize`)
- Roteia o vetor para um bucket do IVF e faz busca exata dentro do bucket (`internal/ivf`)
- Pool de `domain.Request` + tabela pré-computada das 12 respostas possíveis (`approved × k=0..5`) no hot path

### Índice IVF (`internal/ivf`)

- **64 buckets**, particionados por 6 features binárias (last_tx ausente, online, card_present, unknown_merchant, PM, weekend) — escolhidas para espalhar o dataset de ~3 M vetores e manter o query no mesmo bucket dos vizinhos relevantes
- **Quantização int8** (escala 127): cada vetor ocupa 14 bytes vs 56 em float32
- Busca exata dentro do bucket: L2² desenrolado em Dim=14 — o compilador vetoriza para SSE/AVX no amd64
- Top-K=5 via insertion-sort sobre array fixo

### Build do índice

O `Dockerfile.api` faz o build do índice **dentro da própria imagem**, em estágio separado, para evitar re-rodar o build (~10 s) a cada edição em `cmd/api`:

1. Compila `cmd/buildivf` em um estágio isolado
2. Roda `buildivf -refs references.json.gz -out ivf.bin` em outro estágio (cacheado pelo hash do binário + dados de referência)
3. Copia `ivf.bin` para a imagem final

### Vetor de features (14 dimensões)

| # | Feature |
|---|---|
| 0 | Valor da transação (normalizado) |
| 1 | Número de parcelas (normalizado) |
| 2 | Ratio valor / média do cliente |
| 3 | Hora do dia (0–23 → 0–1) |
| 4 | Dia da semana (seg=0…dom=6 → 0–1) |
| 5 | Minutos desde última transação (-1 se ausente) |
| 6 | Km desde última transação (-1 se ausente) |
| 7 | Km de casa |
| 8 | Qtd. de transações nas últimas 24h |
| 9 | Terminal online (0/1) |
| 10 | Cartão presente (0/1) |
| 11 | Estabelecimento desconhecido (0/1) |
| 12 | Risco do MCC |
| 13 | Valor médio do estabelecimento (normalizado) |

### Lógica de score

O IVF retorna os rótulos dos 5 vizinhos mais próximos (`0`=legítimo, `1`=fraude).

```
fraud_score = fraudCount / 5
approved    = fraud_score < 0.6   (≤ 2 vizinhos fraudulentos)
```

## Endpoints

### `POST /fraud-score`

```json
// Request
{
  "id": "uuid",
  "transaction": { "amount": 150.00, "installments": 1, "requested_at": "2026-05-14T10:00:00Z" },
  "customer":    { "avg_amount": 120.0, "tx_count_24h": 3, "known_merchants": ["m1", "m2"] },
  "merchant":    { "id": "m3", "mcc": "5411", "avg_amount": 200.0 },
  "terminal":    { "is_online": true, "card_present": false, "km_from_home": 2.5 },
  "last_transaction": { "timestamp": "2026-05-14T09:00:00Z", "km_from_current": 1.2 }
}

// Response
{ "approved": true, "fraud_score": 0.2 }
```

### `GET /ready`

Retorna `200 OK` quando o índice IVF foi mapeado e está pronto, `503` caso contrário.

## Como rodar

### Pré-requisitos

- Docker e Docker Compose
- Arquivo `resources/references.json.gz` (usado no build da imagem)

### Subir a stack completa

```bash
docker compose up --build
```

A API ficará disponível em `http://localhost:9999`.

### Desenvolvimento local (sem Docker)

O projeto é **pure Go** (CGO_ENABLED=0), então não é preciso C++/g++:

```bash
# 1) Gere o índice IVF a partir das referências
go run ./cmd/buildivf -refs ./resources/references.json.gz -out ./resources/ivf.bin

# 2) Suba a API
RESOURCES_PATH=./resources IVF_PATH=./resources/ivf.bin go run ./cmd/api
```

### Testes

```bash
go test ./...
```

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `IVF_PATH` | `/app/resources/ivf.bin` | Arquivo do índice IVF |
| `RESOURCES_PATH` | `/app/resources` | Diretório com `normalization.json` e `mcc_risk.json` |
| `LISTEN_ADDR` | `:8080` | Endereço de escuta da API |
| `GOMAXPROCS` | `1` | Limita o runtime Go ao orçamento de CPU |
| `GOGC` | `200` | Reduz frequência do GC (índice é mmap, alocação no hot path é mínima) |

## Stack

- **Go 1.24** (pure Go, sem CGO)
- [fasthttp](https://github.com/valyala/fasthttp) — HTTP server de alta performance
- [sonic](https://github.com/bytedance/sonic) — JSON marshal/unmarshal otimizado
- IVF + L2² em int8 — busca k-NN exata dentro do bucket, implementação própria em `internal/ivf`
- **Nginx 1.25** — load balancer
- **Docker Compose** — orquestração local

## Tradeoffs

| Dá up em | Para ganhar |
|---|---|
| Recall máximo (k-NN global) | Latência: exato-dentro-do-bucket evita travessia de grafo / CGO |
| Flexibilidade de modelo | Cache-friendliness: um bucket cabe em L2/L3, busca é memory-bandwidth bound |
| Atualização online do índice | Memória: índice é imutável, mmap compartilhado entre os 2 processos |
| Generalidade (HNSW seria mais robusto) | Simplicidade: pure Go, ~300 linhas, sem dependências nativas |
