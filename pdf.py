from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
load_dotenv()

def load_pdf_and_create_index(pdf_file):

    file_name = f"uploaded_pdf_{pdf_file.name}"
    with open(file_name, "wb") as f:
        f.write(pdf_file.getbuffer())

    loader = PyPDFLoader(
            file_path=file_name,
          
        )

    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(document)


    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

    vector_db= FAISS.from_documents(texts, embeddings)
    save = vector_db.save_local("Faiss_data")
    db = FAISS.load_local('faiss_data',embeddings,allow_dangerous_deserialization=True)
    return db

def res(query):

        llm = ChatGroq() 
        embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
        vectorstore = FAISS.load_local("faiss_data", embeddings,allow_dangerous_deserialization=True)

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain )

        result = retrival_chain.invoke(input={"input": query })

        final = result['answer']

        return final
        