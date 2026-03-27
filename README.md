# AI Knowledge Graph Builder for Enterprise Intelligence

## Overview

This project is an AI-powered platform that builds dynamic knowledge graphs from enterprise data sources such as documents, databases, and structured datasets. It combines entity extraction, graph databases, and Retrieval-Augmented Generation (RAG) to enable intelligent search and insight generation.

The system allows users to explore relationships between entities and make data-driven decisions through a unified interface.

---

## Project Statement

The goal of this project is to develop a system that automatically extracts entities and relationships from enterprise data and organizes them into a knowledge graph. By integrating semantic search and large language models, the platform enables users to uncover hidden patterns and generate meaningful insights.

---

## Key Outcomes

* Automated extraction of entities and relationships from data
* Dynamic knowledge graph construction
* Hybrid semantic and keyword search
* Context-aware answer generation using LLMs
* Interactive dashboard for exploration and analysis

---

## Architecture

```id="4wq9mj"
User Query
   ↓
Streamlit UI
   ↓
Hybrid Search (FAISS + BM25)
   ↓
Relevant Documents
   ↓
LLM (Groq / Llama)
   ↓
Generated Answer

Parallel:
results.json → Entities & Relationships → Graph Visualization
```

---

## Modules Implemented

### Data Ingestion & Processing Layer

* Handles structured data input
* Cleans and transforms data into documents

### Entity & Relationship Extraction Engine

* Uses LLM for Named Entity Recognition and relation extraction
* Produces structured triples

### Graph Construction & Storage Hub

* Stores knowledge graph in Neo4j
* Supports querying and updates

### RAG + Semantic Search Layer

* Embeddings using SentenceTransformers
* FAISS for vector search
* BM25 for keyword search
* Hybrid retrieval for improved accuracy

### Interactive Dashboard

* Streamlit-based UI
* Enables query input, result viewing, and graph exploration

---

## Tech Stack

* Python
* Streamlit
* Pandas
* SentenceTransformers
* FAISS
* BM25
* Neo4j
* Groq API (Llama models)
* NetworkX, Matplotlib

---

## Milestones

* Milestone 1: Data ingestion and schema design
* Milestone 2: Entity extraction and graph construction
* Milestone 3: Semantic search and RAG implementation
* Milestone 4: Dashboard 

---

## Author

Developed as part of an internship project.
