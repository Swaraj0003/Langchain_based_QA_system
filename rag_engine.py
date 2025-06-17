from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Setup Hugging Face API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_token_here"

def build_rag_chain(file_path="/workspaces/Langchain_based_QA_system/data/global_warming.txt"):
    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return rag_chain
