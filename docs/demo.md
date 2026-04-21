# Demo Walkthrough

This guide shows how to demo the AI Project Intelligence System effectively — useful for interviews or presentations.

## Setup (2 minutes)

1. Get free Groq key at [console.groq.com](https://console.groq.com)
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Demo Script

### Step 1 — Upload the sample file
Use `data/uploads/sample_project_log.md` included in the repo.
This simulates a real weekly project log with progress, risks, and action items.

### Step 2 — Query Tab
Try these questions to show RAG in action:
- *"What are the main blockers?"*
- *"What was the model accuracy this week?"*
- *"What are the next steps for the team?"*
- *"Who is on leave and when do they return?"*

Each answer cites the source chunk — this is what separates RAG from a plain LLM.

### Step 3 — NLP Insights Tab
Point out:
- Sentiment score and breakdown (% positive/negative/neutral sentences)
- Top keywords by TF-IDF weight
- Key noun phrases extracted by POS tagging
- Readability grade
- Extractive summary — generated with zero LLM calls

### Step 4 — Report Tab
Fill in:
- Project Name: `AI Model Development`
- Author: your name
- Click **Generate Report**
- Show the Markdown output and download the PDF

## Interview Talking Points

**"Why RAG instead of just prompting the LLM?"**
> RAG grounds the model's output in actual document content, preventing hallucination. The LLM can only answer using what we retrieved — we can verify every claim against the source.

**"Why FAISS over a vector database like Pinecone?"**
> For a local demo and portfolio project, FAISS is zero-cost and runs in-memory. In production I'd migrate to Pinecone or Weaviate for persistence and scale.

**"Why sentence-transformers and not OpenAI embeddings?"**
> It runs fully local — no API cost, no latency, and no data leaves the machine. `all-MiniLM-L6-v2` gives strong semantic similarity at 384 dimensions.

**"How does the NLP analyzer work without an LLM?"**
> VADER for sentiment is a lexicon-based model — fast and interpretable. Keywords use a TF-IDF approximation with NLTK lemmatization. This shows I understand classical NLP, not just LLM wrappers.
