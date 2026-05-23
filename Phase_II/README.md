# 2️⃣ Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset.

## Architecture
* Preprocessing Engine: Scikit-learn Pipeline integrating StandardScaler (numerical) and OneHotEncoder (categorical).
* Modeling Core: Interchangeable Logistic Regression and Random Forest classifiers.
* Optimization: GridSearchCV for automated hyperparameter tuning across multiple algorithms.

## Execution
* Implemented robust data preprocessing steps encapsulated entirely within a Pipeline to prevent look-ahead bias.
* Trained and evaluated Logistic Regression and Random Forest models simultaneously.
* Optimized model parameters using cross-validation via GridSearchCV to maximize predictive accuracy.
* Exported the complete, production-ready pipeline and preprocessing steps as a serialized artifact using joblib.

## Key Insights
1. Encapsulating scaling and encoding within the Scikit-learn Pipeline is mandatory to prevent data leakage during cross-validation.
2. Serializing the entire pipeline (not just the model) ensures raw production data is preprocessed exactly like the training data.

## Tech Stack
Python | Scikit-learn | Pandas | NumPy | Joblib


# 4️⃣ Task 4: Context-Aware Chatbot Using LangChain or RAG

## Objective
Build a RAG-powered AI assistant strictly scoped to answering domain-specific queries about AI, ML, NLP, and Neural Networks.

## Architecture
* Frontend: Custom CSS-themed Streamlit UI (Dark mode).
* Orchestration: LangChain for prompt routing and state management.
* RAG Core: Local FAISS vector database with OpenAI Embeddings and GPT-4o-Mini.

## Execution
* Engineered a responsive, Gemini-styled interface with secure markdown parsing to prevent HTML injection.
* Integrated top-k FAISS document retrieval to ground responses and enforce strict conversational guardrails.
* Resolved dynamic pathing (os.path) and Linux dependency conflicts for a successful Streamlit Community Cloud launch.

## Key Insights
1. Hardcoding file paths breaks cloud environments; dynamic absolute pathing is mandatory for production.
2. Combining strict system prompts with vector retrieval effectively forces the LLM to refuse out-of-scope questions.

## Tech Stack
Python | Streamlit | LangChain | FAISS | OpenAI API (GPT-4o-Mini)

# 5️⃣ Task 5: Auto Tagging Support Tickets Using LLM
