# rinha-backend-2026-go

Solução para a **Rinha de Backend 2026** — serviço de detecção de fraude em tempo real usando busca por vizinhos mais próximos (k-NN) via HNSW, com foco em throughput máximo dentro de recursos extremamente limitados (1 vCPU, 350 MB RAM).

## O que faz?

Recebe eventos de transação (valor, parcelamento, dados do cliente, estabelecimento e terminal) e retorna um `fraud_score` com a decisão de aprovação ou recusa. A decisão é baseada nos 5 vizinhos mais próximos no espaço vetorial de transações históricas rotuladas.

## Arquitetura

```
                        ┌──► api1:8080 ──┐
Client ──► nginx:9999 ──┤                ├──► store:9990 (HNSW)
                        └──► api2:8080 ──┘
```

| Componente | Papel | Limite de recursos |
|---|---|---|
| **nginx** | Load balancer round-robin entre api1 e api2 | 0.05 CPU / 12 MB |
| **api1 / api2** | Vetorização + proxy de busca | 0.20 CPU / 20 MB cada |
| **store** | Índice HNSW em memória, busca k-NN | 0.55 CPU / 298 MB |
| **Total** | | 1.00 CPU / 350 MB |

### API (`cmd/api`)

- HTTP server com [fasthttp](https://github.com/valyala/fasthttp) e JSON parsing com [sonic](https://github.com/bytedance/sonic)
- Converte o payload de entrada em um vetor de 14 dimensões (`internal/vectorize`)
- Encaminha o vetor ao **store** e traduz os rótulos retornados em `fraud_score`

### Store (`cmd/store`)

- Carrega ~3 M vetores de referência de um arquivo `.json.gz` em background
- Indexa via [hnswlib](https://github.com/nmslib/hnswlib) com binding CGO (`internal/hnsw`)
- Expõe dois endpoints HTTP internos: `GET /ready` e `POST /search`

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

O store retorna os rótulos dos 5 vizinhos mais próximos (`0`=legítimo, `1`=fraude).

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

Retorna `200 OK` quando o índice HNSW terminou de carregar, `503` enquanto ainda está indexando.

## Como rodar

### Pré-requisitos

- Docker e Docker Compose
- Arquivo `resources/references.json.gz` com os vetores de referência

### Subir a stack completa

```bash
docker compose up --build
```

A API ficará disponível em `http://localhost:9999`.

### Desenvolvimento local (sem Docker)

O **store** requer CGO e g++ instalado (para compilar o hnswlib):

```bash
# Store (requer g++)
CGO_ENABLED=1 go build -o store ./cmd/store
REFS_PATH=./resources/references.json.gz EF_BUILD=20 EF_SEARCH=20 ./store

# API (em outro terminal)
CGO_ENABLED=0 go build -o api ./cmd/api
STORE_URL=http://localhost:9990 RESOURCES_PATH=./resources ./api
```

O `docker-compose.override.yml` já usa `EF_BUILD=20 EF_SEARCH=20` para indexação mais rápida em desenvolvimento.

### Testes

```bash
go test ./...
```

## Variáveis de ambiente

| Variável | Serviço | Padrão | Descrição |
|---|---|---|---|
| `STORE_URL` | api | `http://store:9990` | URL do serviço store |
| `RESOURCES_PATH` | api | `/app/resources` | Diretório com `normalization.json` e `mcc_risk.json` |
| `API_PORT` | api | `8080` | Porta de escuta da API |
| `REFS_PATH` | store | `/app/resources/references.json.gz` | Arquivo de vetores de referência |
| `STORE_PORT` | store | `9990` | Porta de escuta do store |
| `EF_BUILD` | store | `200` | Parâmetro efConstruction do HNSW (acurácia de build) |
| `EF_SEARCH` | store | `200` | Parâmetro ef do HNSW (acurácia de busca) |

## Stack

- **Go 1.24**
- [fasthttp](https://github.com/valyala/fasthttp) — HTTP server de alta performance
- [sonic](https://github.com/bytedance/sonic) — JSON marshal/unmarshal otimizado
- [hnswlib](https://github.com/nmslib/hnswlib) — índice HNSW via CGO (C++17)
- **Nginx 1.25** — load balancer
- **Docker Compose** — orquestração local

## Tradeoffs

| Dá up em | Para ganhar |
|---|---|
| Flexibilidade de modelo | Latência mínima (k-NN puro, sem modelo ML pesado) |
| Atualizações online do índice | Memória eficiente (índice imutável após load) |
| Simplicidade de deploy (CGO) | Velocidade de busca (hnswlib nativo C++) |
| Precisão máxima (ef alto) | Throughput (ef=200 é o balanço competição/recall) |
