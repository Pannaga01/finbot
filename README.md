# 💼 FinBot – Advanced RAG System with RBAC, Guardrails & Evaluation

## 📌 Overview

✅ This repository has the FinBot GitHub App installed on it.

✅ The project uses LangGraph with a multi-agent architecture.

**FinBot** is an enterprise-grade Retrieval-Augmented Generation (RAG) system built for **FinSolve Technologies**, a B2B fintech company.

The system is designed to:
- Enable employees to query internal documents using natural language
- Enforce **Role-Based Access Control (RBAC)** at the retrieval level
- Prevent unauthorized access, prompt injection, and hallucinated outputs
- Provide **accurate, cited, and grounded responses**
- Be evaluated rigorously using **RAGAS metrics**

---

## 🎯 Objective

The goal of this project is to build a **production-ready AI assistant** that:

- Retrieves information only from **authorized document collections**
- Uses **structured document parsing (Docling)** for better retrieval quality
- Routes queries intelligently using **semantic routing**
- Applies **input/output guardrails** for safety and reliability
- Measures performance using **RAGAS evaluation metrics**

---

## 🏗️ System Architecture

### 🔄 High-Level Flow
User Query
↓
Input Guardrails (Injection / Off-topic / PII)
↓
Semantic Router (Intent Classification)
↓
RBAC Enforcement (Role + Route Intersection)
↓
Vector Retrieval (Qdrant with Metadata Filters)
↓
LLM (Groq – LLaMA 3.1)
↓
Output Guardrails (Grounding + Citation Check)
↓
Final Answer + Citations


---

## ⚙️ Tech Stack

### 🧠 Backend
- **Python**
- **LangChain**
- **Groq (LLaMA 3.1 8B Instant)** – LLM
- **SentenceTransformers** – Embeddings
- **Qdrant** – Vector Database
- **Docling** – Structured document parsing & hierarchical chunking
- **Semantic Router** – Query intent classification
- **LangSmith** – Tracing & observability

---

### 🗄️ Data Processing
- Hierarchical chunking using **Docling**
- Metadata-rich chunks with:
  - `collection`
  - `access_roles`
  - `section_title`
  - `page_number`
  - `chunk_type`
  - `parent_chunk_id`

---

### 🛡️ Guardrails

#### Input Guardrails
- Prompt Injection Detection
- Off-topic Query Detection
- PII Detection
- Rate Limiting

#### Output Guardrails
- Grounding Check (prevents hallucinated numbers)
- Cross-role leakage prevention
- Citation enforcement

---

### 🌐 Frontend
- **Next.js**
- Role-based login system
- Chat interface with:
  - Answer + citations
  - User role display
  - Selected semantic route
  - Guardrail warnings

---

## 🔐 Role-Based Access Control (RBAC)

RBAC is enforced **at the vector database level** using metadata filtering in **Qdrant**.

| Role        | Access |
|------------|--------|
| employee   | General |
| finance    | Finance + General |
| engineering| Engineering + General |
| marketing  | Marketing + General |
| c_level    | All collections |

➡️ Unauthorized data is **never retrieved**, even with adversarial prompts.

---

## 🧭 Query Routing

Semantic routing classifies queries into:

- `finance_route`
- `engineering_route`
- `marketing_route`
- `hr_general_route`
- `cross_department_route`

The selected route is then **validated against user role** before retrieval.

---

## 📊 Evaluation (RAGAS)

The system is evaluated using **RAGAS** with a custom dataset of 40+ QA pairs.

### 📈 Key Metrics

| Metric              | Score |
|-------------------|------|
| Answer Relevancy  | **0.93** |
| Context Recall    | **1.00** |


---

## 🚀 Key Features

- ✅ Secure retrieval with RBAC at source level  
- ✅ Structured document understanding via hierarchical chunking  
- ✅ Intelligent query routing  
- ✅ Strong guardrails against misuse  
- ✅ High evaluation scores with measurable performance  
- ✅ Scalable and modular architecture  


## ⚡ Setup Instructions

```bash
# Clone repo
git clone <your-repo-link>

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn main:app --reload

# Run frontend
cd frontend
npm install
npm run dev