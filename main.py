import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "❌ GOOGLE_API_KEY not found. "
    )
    st.stop()
else:
    st.success("API key loaded successfully!")

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

faiss_index_dir = "faiss_index_gemini"
main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=500,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
)

if process_url_clicked:
    urls = [u for u in urls if u.strip()]
    if not urls:
        st.warning("Please enter at least one valid URL to process.")
        st.stop()

    with st.spinner("Processing URLs... This might take a moment."):
        try:
            main_placeholder.text("Data Loading… Started ✅")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            if not data:
                st.error("No data could be loaded from the provided URLs. Check URLs for accessibility or try different ones.")
                st.stop()

            main_placeholder.text("Text Splitter… Started ✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1_000,
                chunk_overlap=200,
            )
            docs = text_splitter.split_documents(data)
            if not docs:
                st.error("No text chunks could be generated. Check splitter configuration or source content.")
                st.stop()

            main_placeholder.text("Embedding Vector… Building ✅")
            vectorstore = FAISS.from_documents(docs, embeddings)
            time.sleep(1)

            vectorstore.save_local(faiss_index_dir)
            main_placeholder.success("Processing complete! You can now ask questions about the articles.")

        except Exception as e:
            main_placeholder.error(f"An error occurred during URL processing: {e}")
            st.exception(e)
            st.stop()

query = main_placeholder.text_input("Question:")
if query:
    if not os.path.exists(faiss_index_dir) or not os.listdir(faiss_index_dir):
        st.error(f"Vector store not found in '{faiss_index_dir}'. Please click **Process URLs** first to build it.")
        st.stop()

    with st.spinner("Getting your answer... This may take a moment."):
        try:
            vectorstore = FAISS.load_local(
                faiss_index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )

            prompt_template = ChatPromptTemplate.from_template("""
            You are an AI assistant for a news research tool.
            Answer the question based only on the following context, providing sources if available.
            If the answer is not in the context, clearly state that you don't have enough information.

            Context:
            {context}

            Question: {input}
            """)

            document_chain = create_stuff_documents_chain(llm, prompt_template)

            retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)

            response = retrieval_chain.invoke({"input": query})

            st.header("Answer")
            answer = response.get("answer", "").strip()
            if answer:
                st.write(answer)
            else:
                st.info("The model could not generate a relevant answer from the provided sources.")

            st.subheader("Sources")
            sources_found = False
            if "context" in response and isinstance(response["context"], list):
                unique_sources = set()
                for doc in response["context"]:
                    if isinstance(doc, Document) and hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        unique_sources.add(doc.metadata['source'])

                if unique_sources:
                    sources_found = True
                    for src in sorted(list(unique_sources)):
                        st.write(f"- {src}")

            if not sources_found:
                st.info("No specific sources were found in the retrieved context for this answer.")

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
            st.exception(e)
            st.stop()