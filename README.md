# CLAUDE.md — Context persistant pour Claude Code

## Ce projet
Pipeline RAG complet (FastAPI + pgvector + Mistral).
Projet portfolio GitHub pour entretiens développeur IA.
Développeur : Senior Python, Banque de France.

## Stack — respecter strictement
- LLM : Mistral AI (mistral-small-latest par défaut)
- Embeddings : mistral-embed (dimension 1024)
- Vector DB : pgvector sur PostgreSQL 16 (localhost:5432/ragdb)
- Backend : FastAPI + SQLAlchemy 2.x + asyncpg
- Gestion paquets : uv (jamais pip)
- Python 3.12, type hints partout, Pydantic v2

## Structure du projet
src/
├── api/          → endpoints FastAPI
├── rag/          → pipeline (ingestion, retrieval, generation)
├── embeddings/   → chunking + appels mistral-embed
└── db/           → modèles SQLAlchemy + requêtes pgvector
notebooks/        → exploration et tests de concepts
tests/            → pytest async

## Conventions
- Async partout (httpx, asyncpg, FastAPI async)
- loguru pour les logs (pas print)
- tenacity pour les retries sur les appels API
- Variables d'env via python-dotenv (.env)
- Jamais de clé API dans le code
- Logger tous les appels LLM : modèle, tokens, latence