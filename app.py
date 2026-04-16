import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# --- THE FIX: Now importing from langchain_classic ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. UI & APP SETUP ---
st.set_page_config(page_title="Local AI Document Analyzer", page_icon="📄")
st.title("📄 Local AI Document Analyzer")
st.markdown("Secure, offline Retrieval-Augmented Generation (RAG) system using **Llama 3**.")

# --- 2. INITIALIZE LLM & EMBEDDINGS ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3")

# --- 3. PROMPT TEMPLATE ---
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- 4. FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    with st.spinner("Processing document and building local Vector DB..."):
        try:
            # A. Load the PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            # B. Split the text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # C. Create Vector Database
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            # D. Create RAG Chain
            retriever = vectorstore.as_retriever()
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            st.success("Document processed successfully! Database is highly available.")

            # --- 5. CHAT INTERFACE ---
            user_query = st.text_input("Ask a question about your document:")
            
            if user_query:
                with st.spinner("Analyzing and Generating Answer..."):
                    response = rag_chain.invoke({"input": user_query})
                    st.write("### AI Response:")
                    st.info(response["answer"])

        except Exception as e:
            st.error(f"An error occurred while processing the document: {e}")
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
else:
    st.info("Please upload a PDF file to initialize the localized software stack.")