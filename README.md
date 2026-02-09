# 🚀 VisionRAG

**VisionRAG** is an image-visible **Multimodal Retrieval-Augmented Generation (RAG)** system that can retrieve and display **text, tables, and images** from documents using vision-aware semantic search.

Unlike traditional RAG systems that only work with text, VisionRAG understands **what an image represents**, retrieves it when relevant, and displays the actual image alongside the generated answer.

---

## ✨ Key Features

* 🔍 **Multimodal Retrieval** – Search across text, tables, and images
* 🖼️ **Image-Visible RAG** – Retrieved images are displayed, not just described
* 🧠 **Vision-Aware Embeddings** – Images are indexed using semantic summaries
* 📄 **PDF Understanding** – Supports complex PDFs with tables and figures
* ⚙️ **Production-Style Architecture** – Multi-vector retrieval + docstore

---

## 🧠 How VisionRAG Works

1. **Document Ingestion**

   * PDFs are partitioned into text, tables, and images using *Unstructured*

2. **Image Understanding**

   * Each image is summarized using a Vision LLM
   * The summary is embedded for semantic search

3. **Storage Strategy**

   * **Vector Store** → Text summaries (searchable)
   * **Docstore** → Original content (text, tables, image base64)

4. **Query Flow**

   * User query → semantic search over summaries
   * Matching documents retrieved
   * Images are rendered and answers are generated using LLM

---

## 🏗️ Architecture Overview

```

PDF Documents
     ↓
Partition (Text | Table | Image)
     ↓
Image → Vision Summary → Embedding
Text  → Text Summary  → Embedding
     ↓
Vector Store (Search)
     ↓
Docstore (Original Content)
     ↓
LLM + UI (Text + Image Response)
```

---

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **Unstructured** (PDF parsing)
* **OpenAI GPT-4.1-nano** (Text & Vision)
* **OpenAI Embeddings**
* **ChromaDB** (Vector Store)
* **MultiVectorRetriever**

---

## 📦 Installation

```bash
pip install langchain unstructured chromadb openai
```

Make sure you have your OpenAI API key set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## ▶️ Usage Example

```python
question = "Explain the multi-head attention diagram"
answer, context = query_rag(question)

# Display retrieved images
show_images(context["images"])

print(answer)
```

---

## 🎯 Example Queries

* "Explain the attention mechanism shown in the diagram"
* "What does the transformer architecture image describe?"
* "Summarize the table comparing attention heads"

---

## 💡 Why VisionRAG?

Traditional RAG systems:

* ❌ Cannot retrieve images
* ❌ Lose visual context

VisionRAG:

* ✅ Retrieves image semantics
* ✅ Displays the actual image
* ✅ Enables real multimodal reasoning

---

## 🧪 Use Cases

* Research paper analysis
* Technical documentation QA
* Educational content understanding
* Multimodal knowledge assistants

---

## 📌 Future Improvements

* Streamlit / Web UI
* Citation and page-number tracking
* Reranking for higher accuracy
* Support for audio & video

---

## 👨‍💻 Author

**Ahmed2797**
AI & ML Enthusiast | Multimodal Systems Learner

---

## ⭐ Acknowledgements

* LangChain
* OpenAI
* Unstructured

---

If you found this project useful, feel free to ⭐ the repository!
