import streamlit as st
from rag_pipeline import generate_answer

st.set_page_config(page_title="Swiggy Annual Report RAG", layout="wide")

st.title("📄 Swiggy Annual Report Q&A")
st.write("Ask questions based on the uploaded Annual Report PDF.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        answer, docs = generate_answer(query)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)