import wikipedia
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

topics = ["Artificial Intelligence", "Machine Learning", "Neural Networks"]

documents = []
for topic in topics:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        documents.append(Document(page_content=page.content, metadata={"source": topic}))
        print(f"Loaded: {topic}")
    except Exception as e:
        print(f"Failed: {topic} — {e}")

print(f"Total documents loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")

print("Done. Vectorstore saved.")