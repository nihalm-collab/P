import os
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# --- Ortam Değişkenleri ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Streamlit Arayüzü ---
st.title("📚 Kitapyurdu RAG Chatbot (LangChain + Gemini)")
st.markdown("Kullanıcı yorumlarına dayalı kitap asistanı 💬")

# --- Veri Setini Yükle ---
@st.cache_data(show_spinner=False)
def load_data():
    dataset = load_dataset("alibayram/kitapyurdu_yorumlar", split="train", token=HF_TOKEN)
    return dataset["review"]

data = load_data()

# --- Metinleri Böl ---
@st.cache_data(show_spinner=False)
def split_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents(texts)
    return docs

docs = split_texts(data[:5000])  # hızlı başlatmak için ilk 5000 yorumu alıyoruz

# --- Embedding Modeli ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- ChromaDB Oluştur ---
PERSIST_DIR = "chroma_db"
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Gemini Modelini Tanımla ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

# --- LangChain RAG Pipeline ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # bağlamı doğrudan birleştir
    retriever=retriever,
    return_source_documents=True
)

# --- Kullanıcı Sorgusu ---
user_query = st.text_input("Bir kitap hakkında ne öğrenmek istiyorsun?")
if st.button("Sor"):
    if user_query:
        with st.spinner("Yanıt hazırlanıyor..."):
            response = qa_chain(user_query)
            st.write("💬 **Asistan:**", response["result"])

            with st.expander("📚 Kaynak Yorumlar"):
                for doc in response["source_documents"]:
                    st.markdown(f"- {doc.page_content}")
    else:
        st.warning("Lütfen bir soru gir.")

