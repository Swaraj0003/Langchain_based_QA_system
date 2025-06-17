import streamlit as st
from rag_engine import build_rag_chain

st.set_page_config(page_title="RAG with LangChain + FAISS", layout="centered")

st.title("🧠 RAG System with LangChain + FAISS")
st.markdown("Ask a question based on the uploaded knowledge base (`data.txt`).")

# Build chain once
rag_chain = build_rag_chain()

query = st.text_input("❓ Your question:")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        result = rag_chain({"query": query})
        st.subheader("📌 Answer")
        st.write(result["result"])

        st.subheader("📂 Source Snippets")
        for doc in result["source_documents"]:
            st.markdown(f"**From**: `{doc.metadata.get('source', 'N/A')}`\n\n> {doc.page_content[:300]}...")
