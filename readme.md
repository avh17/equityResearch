# RockyBot: News Research Tool 📈

RockyBot is an AI-powered news research tool that allows you to extract information and get answers from multiple news articles using Google's Gemini models. Simply provide URLs to news articles, and RockyBot will process them to create a searchable knowledge base, then answer your questions based on the content of those articles.

## Features

* **URL-based Data Ingestion:** Easily load content from multiple news article URLs.
* **Intelligent Text Processing:** Utilizes LangChain's `RecursiveCharacterTextSplitter` to break down articles into manageable chunks.
* **Vector Embeddings:** Transforms text chunks into numerical representations using Google's `embedding-001` model for efficient similarity search.
* **FAISS Vector Store:** Stores and retrieves relevant document chunks quickly using Facebook AI Similarity Search (FAISS).
* **Gemini-powered Q&A:** Leverages Google's `gemini-1.5-flash` model for answering questions based on the retrieved article content.
* **Source Attribution:** Provides the URLs of the articles used to generate the answers, enhancing trustworthiness.
* **Interactive Streamlit UI:** A user-friendly web interface for seamless interaction.

