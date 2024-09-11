import streamlit as st
from pdf import load_pdf_and_create_index,res


st.title("PDF Question Answering App")
pdf_file = st.file_uploader("Upload a PDF file")
if st.button('submit'):
    if pdf_file is not None:
        load_pdf_and_create_index(pdf_file)
        st.success("PDF loaded!")

question = st.text_input("Ask a question about the PDF:")

if question:
        st.spinner("loading your answer...")
        answer= res(question)
        st.write("Answer:", answer)