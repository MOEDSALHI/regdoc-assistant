# RegDoc Assistant

[![CI](https://github.com/<user>/regdoc-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/regdoc-assistant/actions/workflows/ci.yml)

> RAG (Retrieval-Augmented Generation) pour documents réglementaires français (RGPD, CNIL, ANSSI), avec évaluation chiffrée de 4 stratégies de retrieval et observabilité Prometheus.

**Stack EU-compliant** — Mistral AI · pgvector · FastAPI · Python 3.12 · Docker · Prometheus

---

## Le problème

Les LLM hallucinent sur les questions juridiques. Sur un domaine régulé (banque, santé, données personnelles), une réponse fausse engage la responsabilité. RegDoc Assistant montre comment ancrer un LLM dans des sources officielles, mesurer objectivement la qualité du système, choisir la stratégie de retrieval qui correspond au corpus, et instrumenter le tout pour la production.

## Ce qui est mesuré, pas juste implémenté

| Stratégie de retrieval     | Faithfulness | Answer Rel. | Context Recall | Context Precision |
|----------------------------|:------------:|:-----------:|:--------------:|:-----------------:|
| **naive** (cosine)         | **0.988**    | 0.935       | 1.000          | **1.000**         |
| hybrid (BM25 + RRF)        | 0.967        | 0.927       | 1.000          | 0.875             |
| reranked (cross-encoder)   | 0.982        | 0.931       | 1.000          | 0.833             |
| HyDE                       | 0.974        | **0.938**   | 1.000          | 1.000             |

> 12 questions ground-truth sur les Articles 5, 17, 32, 35, 37, 83 du RGPD. Juge : `mistral-small-latest`. Analyse complète : [`notes/ragas_results.md`](notes/ragas_results.md).

**Lecture des résultats :** sur ce corpus (petit, propre), le retrieval naïf gagne. Les techniques avancées dégradent la precision en ajoutant du bruit. La sophistication paie sur des corpus larges et bruités — pas par défaut.

## Architecture

```
┌────────────┐    ┌───────────────────────┐    ┌──────────────┐
│  Question  │──▶│  FastAPI /ask         │──▶│  Mistral     │
└────────────┘    │  retrieval_mode       │    │  (génération)│
                  │  ┌──────────────────┐ │    └──────────────┘
                  │  │ naive            │ │            ▲
                  │  │ hybrid (BM25+RRF │ │            │ contexte
                  │  │   + vector)      │ │    ┌──────────────┐
                  │  │ reranked         │ │──▶│  pgvector    │
                  │  │ HyDE             │ │    │  PostgreSQL  │
                  │  └──────────────────┘ │    └──────────────┘
                  └───────────────────────┘
                              │
                              ▼
                  ┌──────────────────┐
                  │  /metrics        │  Prometheus exposition
                  │  custom RAG +    │  (queries, latencies,
                  │  HTTP automatic  │   tokens, errors)
                  └──────────────────┘
```

Détails techniques :
- **Chunking par article** (préserve la structure réglementaire) + recursive chunking fallback
- **Embeddings Mistral** (1024 dim) → pgvector avec index HNSW
- **Hybrid search** : BM25 in-memory + cosinus, fusion via Reciprocal Rank Fusion (k=60)
- **Reranking** : cross-encoder multilingue `mmarco-mMiniLMv2`, two-stage retrieval (top-10 → top-3)
- **HyDE** : génération d'un document hypothétique avant embedding (Gao et al., 2022)
- **Prompt injection defense** : détection directe + sandwich prompting
- **Observabilité Prometheus** : métriques custom (queries, retrieval, LLM tokens, errors) + HTTP auto-instrumentation

## Démarrage rapide

Prérequis : Docker, et optionnellement Python 3.12 + [`uv`](https://github.com/astral-sh/uv) pour le développement local.

### Voie A — Docker Compose (recommandé)

```bash
git clone https://github.com/<user>/regdoc-assistant.git
cd regdoc-assistant
cp .env.example .env
# Éditer .env et ajouter ta clé MISTRAL_API_KEY

docker compose up -d
# → http://localhost:8000/docs
# → http://localhost:8000/metrics
```

Toute la stack (API + pgvector) démarre orchestrée avec healthchecks. Image API : 1.94 GB (multi-stage, CPU-only torch).

### Voie B — Développement local

```bash
uv sync
cp .env.example .env

# Démarrer juste pgvector via compose
docker compose up -d pgvector

# Lancer l'API en local avec reload
uv run uvicorn main:app --reload
```

### Exemple d'appel

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quand une AIPD est-elle obligatoire ?",
    "retrieval_mode": "naive",
    "top_k": 3
  }'
```

### Observabilité

```bash
curl -s http://localhost:8000/metrics | grep "^rag_"
# rag_queries_total{mode="naive",status="success"}     1247
# rag_retrieval_duration_seconds_sum{mode="naive"}     34.2
# rag_llm_tokens_total{type="prompt",model="..."}      612000
# rag_injection_blocked_total                          3
# ...
```

## Lancer l'évaluation

```bash
# Évaluer une configuration sur les 12 questions ground-truth
uv run python tests/evaluation/evaluate.py naive
uv run python tests/evaluation/evaluate.py hybrid
uv run python tests/evaluation/evaluate.py reranked
uv run python tests/evaluation/evaluate.py hyde
```

Chaque run produit `notes/ragas_results_<mode>.json`. Le rapport comparatif est dans `notes/ragas_results.md`.

## Lancer les tests

```bash
uv run pytest                  # 94 tests, ~30s
uv run pytest -v --co          # liste détaillée
uv run ruff check src/ tests/  # lint
uv run ruff format --check src/ tests/ main.py  # check formatting
```

## Stack technique

| Composant       | Choix                              | Raison                                |
|-----------------|------------------------------------|---------------------------------------|
| LLM             | Mistral AI (`mistral-small-latest`)| RGPD-compliant, EU-hosted             |
| Embeddings      | `mistral-embed` (1024 dim)         | Cohérence stack                       |
| Vector store    | pgvector / PostgreSQL 16           | SQL standard, pas de vendor lock      |
| API             | FastAPI + Pydantic v2              | Async natif, typage strict            |
| DB driver       | asyncpg + SQLAlchemy 2.x async     | Pool de connexions, transactions      |
| Cross-encoder   | sentence-transformers + torch CPU  | Modèle multilingue local              |
| Évaluation      | Custom LLM-as-judge avec Mistral   | Évite les conflits ragas+instructor   |
| Observabilité   | prometheus-client + FastAPI instr. | Standard industrie, /metrics endpoint |
| Tooling         | uv, ruff, pytest-asyncio           | Stack Python 2025                     |
| Infra           | Docker multi-stage + compose       | Image 1.94 GB, CPU-only torch         |
| CI/CD           | GitHub Actions                     | Lint + tests + Docker build           |

## Décisions architecturales notables

- **Pas de LangChain** dans le pipeline principal. Chaque étape (chunking, retrieval, prompting, parsing) est écrite à la main pour comprendre le coût de chaque sophistication et garder le contrôle des prompts.
- **RAGAS réimplémenté** avec Mistral comme juge. La librairie `ragas` a des conflits de dépendances irréconciliables avec `mistralai` v2. Réimplémenter 4 métriques montre que je comprends ce que RAGAS fait — pas juste comment l'installer.
- **Torch CPU-only** forcé via `[tool.uv.sources]` + index PyTorch dédié. Économie de 4.4 GB sur l'image Docker (6.29 → 1.94 GB).
- **Cross-encoder baked-in** dans l'image au build time → pas de download au démarrage, pas de dépendance réseau en runtime (critique en environnement régulé).
- **Idempotency d'ingestion** via filename — recharger un document n'écrit pas de doublons.
- **Sandwich prompting** + détection d'injection directe + compteur Prometheus dédié pour les tentatives bloquées.
- **Observabilité séparée** : `Instrumentator().instrument(app)` pour la collecte (middleware), route manuelle `/metrics` pour l'exposition. Sépare clairement les deux préoccupations, simplifie l'auth ultérieure.

## Structure du projet

```
src/
├── api/
│   ├── routes/        # endpoints FastAPI (ask, ingest, chat, health, metrics)
│   └── schemas/       # Pydantic schemas (AskRequest, IngestRequest, etc.)
├── rag/               # ingestion, retrieval, hybrid_search, reranker,
│                      # query_expansion (HyDE), parent_child
├── embeddings/        # chunking + appels mistral-embed
├── prompts/           # system prompts + parsing JSON structuré
├── security/          # prompt injection defense
├── services/          # client Mistral (tenacity retries), token counter
├── observability/     # Prometheus custom RAG metrics
├── db/                # asyncpg pool, schema SQL, repository
└── config.py          # pydantic-settings (env vars)

tests/
├── evaluation/        # dataset ground-truth + 4 métriques + script eval
└── test_*.py          # pytest, 94 tests

.github/workflows/
└── ci.yml             # GitHub Actions: lint + tests + docker build
```

## Roadmap

- ✅ **Bloc 1** : prompt engineering, fondations LLM
- ✅ **Bloc 2** : RAG de base (ingestion, retrieval cosinus, citations)
- ✅ **Bloc 3** : RAG avancé (hybrid, reranking, HyDE, parent-child)
- ✅ **Bloc 4** : évaluation RAGAS-equivalent multi-configuration
- ✅ **Bloc 5** : Docker multi-stage, observabilité Prometheus, CI/CD GitHub Actions
