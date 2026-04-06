# 🔋 Energy Policy Intelligence — RAG-Powered State Energy Plan Analyzer

> Ask policy questions. Get cited answers. Instantly.

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you query U.S. state energy plans in plain English — powered by LLaMA 3.3, Pinecone vector search, and a LangChain LCEL pipeline. Built to turn dense government PDFs into actionable intelligence.

---

## 🚀 What It Does

Most state energy policy documents are buried in PDFs nobody reads. This tool ingests those plans, indexes them semantically, and gives you a conversational interface to extract specific targets, strategies, and policy data — with **source citations on every answer**.

- 💬 **Chat interface** to query state energy plans in natural language
- 📄 **Source-cited responses** — every answer traces back to the original document
- ⚡ **Sub-second retrieval** via Pinecone vector store (k=5 semantic chunks)
- 🧠 **LLaMA 3.3 70B** (via Groq) for analyst-grade response quality
- 🔁 Designed for **50 states + D.C.** with a repeatable ingestion pipeline

> **Current Status:** Prototype — New York indexed. Additional states in progress.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | LLaMA 3.3-70B via Groq |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | Pinecone |
| RAG Framework | LangChain (LCEL) |

---

## 🏗️ Architecture
User Query
↓
HuggingFace Embeddings → Pinecone Vector Store (semantic retrieval)
↓
LangChain LCEL Chain → Groq LLaMA 3.3 70B
↓
Cited Answer + Source Documents

text

---

## 📋 Data Scope

- **Coverage:** 50 U.S. States + D.C. (territories optional)
- **Recency filter:** Plans published or updated within the last 24 months only
- **Document types:** State energy strategies, clean energy roadmaps, IRP-equivalent summaries, governor/energy office plans
- **Key data extracted:** Resource mix goals, emissions targets, transmission strategy, storage targets, firm capacity (nuclear, gas, geothermal), load themes (data centers, EVs, industrial)

---

## 💡 Who Benefits From This

This isn't just a demo — tools like this solve a real bottleneck in energy and policy work:

- **Energy policy analysts** — cross-state comparisons in seconds, not days
- **Investment & market research teams** — rapidly identify which states have favorable clean energy mandates or transmission buildout plans
- **Regulatory consultants** — surface specific statutory authority, timelines, and funding mechanisms without manual doc review
- **Think tanks & nonprofits** — monitor policy momentum across jurisdictions at scale
- **Developers & IPPs** — quickly assess state-level clean energy targets, storage mandates, or permitting posture before committing resources

---

## ⚙️ Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/srikarananand/state-energy-rag.git
cd state-energy-rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add API keys**

Create `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "your-pinecone-key"
GROQ_API_KEY = "your-groq-key"
```

**4. Run**
```bash
streamlit run app.py
```

---

## 📦 Requirements
streamlit
langchain / langchain-community
langchain-huggingface
langchain-groq
langchain-pinecone
pinecone
sentence-transformers

text

---

## 🗺️ Roadmap

- [x] NY state plan indexed and queryable
- [ ] Ingest all 50 states + D.C.
- [ ] Add metadata filters (state, year, plan type)
- [ ] Cross-state comparison mode
- [ ] Quarterly refresh pipeline

---

## 👤 Author

**Srikar Anand** — [GitHub](https://github.com/srikarananand)

*Built to make energy policy research faster, smarter, and actually usable.*
