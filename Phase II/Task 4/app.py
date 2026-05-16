import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from ui import apply_styles, render_sidebar, render_header, render_welcome, render_messages

load_dotenv()

st.set_page_config(
    page_title="Bee-MO AI Assistant",
    page_icon="🤖",
    layout="wide"
)

apply_styles()
selected_model = render_sidebar()

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs):
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "Wikipedia")
        if src not in sources:
            sources.append(src)
    return sources

if "messages" not in st.session_state:
    st.session_state.messages = []

render_header(selected_model)

if not st.session_state.messages:
    render_welcome()

render_messages()

user_input = st.chat_input("Ask about AI, ML, NLP, or Neural Networks...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    retriever = load_retriever()
    llm = ChatOpenAI(model_name=selected_model, temperature=0, streaming=True)

    history_text = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    docs = retriever.invoke(user_input)
    context = format_docs(docs)
    sources = get_sources(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Bee-MO, a helpful assistant. Only answer questions related to Greetings,
        Artificial Intelligence, Machine Learning, Natural Language Processing, and Neural Networks (Everything around this domain).
        If the question is outside these topics, respond with:
        'Sorry, I can only answer questions about AI, ML, NLP, and Neural Networks.' Do not make up answers if the context doesn't contain the information.

        Context:
        {context}

        Chat History:
        {history}"""),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    with st.chat_message("assistant", avatar="🤖"):
        full_answer = st.write_stream(chain.stream({
            "context": context,
            "history": history_text,
            "input": user_input
        }))

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources
    })

    st.rerun()