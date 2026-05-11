# AI/ML Engineering Tasks (Phase II)

## 4️⃣ Task 4: Context-Aware Chatbot Using LangChain or RAG

### Objective:
Engineer a Retrieval-Augmented Generation (RAG) conversational agent capable of grounding its responses in a custom vectorized knowledge base while maintaining multi-turn context and a custom UI.

### Architecture:
* UI/Deployment: Streamlit (with Custom HTML/CSS Injection).
* Orchestration: LangChain.
* Storage: FAISS (In-Memory Vector Database).
* LLM Engine: OpenAI API.

### Execution:

* Implemented dynamic document parsing (PDF/Text) and chunking via RecursiveCharacterTextSplitter to optimize context windows.
* Engineered a semantic search mechanism using OpenAI Embeddings to inject the most relevant document chunks into the LLM context.
* Bypassed native Streamlit chat components, utilizing unsafe_allow_html to inject a highly customized HTML/CSS chat interface.
* Designed a robust memory engine utilizing st.session_state to decouple conversational memory from Streamlit’s execution loop, preventing state-loss.
