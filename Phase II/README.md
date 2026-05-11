# AI/ML Engineering Tasks (Phase II)

## 4️⃣ Task 4: Context-Aware Chatbot Using LangChain or RAG

### Objective:
Build a Retrieval-Augmented Generation (RAG) chatbot that grounds responses in a custom vectorized knowledge base.

### Architecture:

* UI/Deployment: Streamlit (with Custom HTML/CSS Injection).
* Orchestration: LangChain.
* Storage: FAISS (In-Memory Vector Database).
* LLM Engine: OpenAI API.

### Execution:

* Chunked and vectorized documents using RecursiveCharacterTextSplitter and OpenAI Embeddings.
* Injected relevant document chunks into the LLM context using FAISS similarity search.
* Built a custom HTML/CSS chat interface using Streamlit's unsafe_allow_html.
* Decoupled conversational memory from Streamlit’s rerun loop using st.session_state.

### Key Insights:

* Streamlit reruns scripts on every interaction; vaulting chat history inside session state prevents data loss.
* Separating data ingestion from the frontend UI prevents redundant API calls and lowers compute costs.

### Tech Stack:
* Python
* LangChain
* Streamlit
* FAISS
* OpenAI API

## 5️⃣ Task 5: Auto Tagging Support Tickets Using LLM
