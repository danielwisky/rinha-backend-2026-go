# rinha-backend-2026-go

Solução para a **Rinha de Backend 2026** — detecção de fraude em tempo real via k-NN com índice **IVF (inverted-file) k-means + multi-probe, quantizado em int8**, *mmap* em processo. Foco em throughput máximo dentro de 1 vCPU / 350 MB RAM.

## Score

Testado localmente com o k6 oficial (ramp 1→900 RPS em 120 s, 54 100 requisições):

| Métrica | Valor |
|---|---|
| `p99` | ~2,4 ms |
| `p99_score` | ~2600 |
| `detection_score` | ~1600 |
| **`final_score`** | **~4200** |
| `failure_rate` | 0,23% |

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
| **nginx** | Load balancer round-robin entre api1 e api2, keep-alive 256 | 0,20 CPU / 10 MB |
| **api1 / api2** | Vetorização + busca k-NN no IVF mmap | 0,40 CPU / 170 MB cada |
| **Total** |  | 1,00 CPU / 350 MB |

> Memória: as 170 MB por api são virtuais. O índice (~50 MB em disco, int8) entra na RSS de cada processo via mmap mas as **páginas físicas são compartilhadas** entre api1 e api2 (mesmo inode na imagem) e ficam **`mlock`'d** para que o kernel não evicte sob pressão.

### API (`cmd/api`)

- HTTP server [fasthttp](https://github.com/valyala/fasthttp) com buffers tunados (Concurrency=4096, ReadBuf=1024)
- JSON com [sonic.ConfigFastest](https://github.com/bytedance/sonic) — pula validação UTF-8 e ordenação de chaves
- Converte o payload em vetor de 14 dimensões (`internal/vectorize`)
- Roteia o vetor para os top-`NProbe` clusters do IVF e faz busca exata dentro deles (`internal/ivf`)
- Pool de `domain.Request` (com `KnownMerchants` pré-alocado em cap=8) + tabela pré-computada das 12 respostas possíveis (`approved × k=0..5`) no hot path
- **Warmup**: antes de marcar `/ready=200`, executa ~1 000 buscas sintéticas para faultar páginas mmap e aquecer L1/L2 + sonic JIT

### Índice IVF (`internal/ivf`)

- **2048 clusters k-means** (mini-batch, Sculley 2010 — 300 iters × 20 000 batch)
- **Multi-probe**: a query é roteada para os `NProbe=4` clusters mais próximos do centroide, e o top-K=5 final é tirado da união
- **Quantização int8** (escala 127): cada vetor ocupa 16 bytes (14 dims + 2 padding) vs 56 em float32
- **Distância**: L2² em assembly SSE2 (`internal/ivf/dist_amd64.s`) — ~6 instruções por vetor, sem branch
- Top-K=5 via insertion-sort sobre array fixo
- `mlock`ado em RAM após `mmap` (com fallback warning-only se `RLIMIT_MEMLOCK` não permitir)

Trade-off de busca: `2048` distâncias contra centroides + `4 × ~1500` distâncias contra vetores de cluster = **~8 000 distâncias por query** (vs centenas de milhares de uma varredura linear).

### Build do índice

O `Dockerfile.api` faz o build do índice **dentro da própria imagem**, em estágio separado, para evitar re-rodar o build (~2 min) a cada edição em `cmd/api`:

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
| 12 | Risco do MCC (lookup denso `[10000]float32`) |
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

Retorna `200 OK` somente após o warmup ter completado (índice mmap'd + ~1 000 buscas sintéticas executadas). Retorna `503` enquanto o warmup ainda está em andamento.

## Como rodar

### Pré-requisitos

- Docker e Docker Compose
- Arquivo `resources/references.json.gz` (usado no build da imagem)

### Subir a stack completa

```bash
docker compose up --build
```

A API ficará disponível em `http://localhost:9999`.

### Rodar o teste oficial (k6)

```bash
# repo de testes em paralelo a este
cd ../rinha-de-backend-2026
./run.sh
```

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

Inclui um *sanity test* (`TestSearchMatchesBruteForce`) que constrói um índice pequeno e mede recall@5 contra brute-force int8 — protege contra regressões na busca multi-probe.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `IVF_PATH` | `/app/resources/ivf.bin` | Arquivo do índice IVF |
| `RESOURCES_PATH` | `/app/resources` | Diretório com `normalization.json` e `mcc_risk.json` |
| `LISTEN_ADDR` | `:8080` | Endereço de escuta da API |
| `GOMAXPROCS` | `1` | Limita o runtime Go ao orçamento de CPU |
| `GOGC` | `200` | Reduz frequência do GC (alocação no hot path é mínima) |
| `GOMEMLIMIT` | `150MiB` | Soft heap limit (pause antes do hard limit do container) |
| `PPROF` | (off) | Se `=1`, abre `/debug/pprof/...` em `:6060` |

## Stack

- **Go 1.24** (pure Go, sem CGO)
- [fasthttp](https://github.com/valyala/fasthttp) — HTTP server de alta performance
- [sonic.ConfigFastest](https://github.com/bytedance/sonic) — JSON ultra-rápido
- IVF k-means + multi-probe + SSE2 — implementação própria em `internal/ivf` (~700 linhas Go)
- **Nginx 1.25** — load balancer
- **Docker Compose** — orquestração local

## Histórico de otimizações

| Versão | Score | p99 | Mudança |
|---|---|---|---|
| v1 (binary-bucket IVF, 256 buckets) | 1839 | 661 ms | baseline pré-otimização |
| v2 base (k-means IVF 2048 + multi-probe NProbe=4) | 2917 | 48 ms | troca do particionamento + warmup + mlock + sonic fastest + CPU rebalance (nginx 0,10) |
| v3 (CPU sweet spot) | **~4200** | **~2,4 ms** | nginx 0,20 / api 0,40 — single nginx worker estava saturando a 0,10 sob burst |

## Tradeoffs

| Dá up em | Para ganhar |
|---|---|
| Recall máximo (k-NN exato global) | Latência: multi-probe sobre k-means com quantização int8, sem travessia de grafo nem CGO |
| Flexibilidade de modelo | Cache-friendliness: centroides + cluster top-4 cabem em L1/L2, busca é memory-bandwidth bound |
| Atualização online do índice | Memória: índice é imutável, mmap compartilhado entre os 2 processos, `mlock`ado |
| Generalidade (HNSW seria mais robusto) | Simplicidade: pure Go, sem dependências nativas |
