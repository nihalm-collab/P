import os
from dotenv import load_dotenv
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import streamlit as st

# --- Ortam değişkenlerini yükle ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)

# --- Başlık ---
st.title("📚 Kitapyurdu Yorum Asistanı Chatbot")

# --- Veri setini yükle ---
@st.cache_data(show_spinner=False)
def load_data():
    dataset = load_dataset("alibayram/kitapyurdu_yorumlar", split="train", token=HF_TOKEN)
    return dataset

data = load_data()

# --- Embedding Modeli ---
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# --- ChromaDB kur ---
PERSIST_DIR = "chroma_db"
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))
collection = client.get_or_create_collection("kitapyurdu_yorumlar")

# --- Veriyi vektörleştir ve ekle ---
if len(collection.get()['ids']) == 0:
    texts = data["review"]
    embeddings = embedder.encode(texts)
    ids = [str(i) for i in range(len(texts))]
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    st.info("Veriler ChromaDB'ye eklendi ✅")

# --- Sorgu işlemi ---
def search_similar_reviews(query):
    query_embedding = embedder.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    return results["documents"][0]

# --- Kullanıcı Girişi ---
user_query = st.text_input("Bir kitapla ilgili sorunu veya isteğini yaz:")

if st.button("Sor"):
    if user_query:
        similar_reviews = search_similar_reviews(user_query)
        context = "\n".join(similar_reviews)

        prompt = f"""
        Aşağıda bazı kullanıcı yorumları var:
        {context}

        Yukarıdaki yorumlara dayanarak, şu soruya doğal bir şekilde yanıt ver:
        Soru: {user_query}
        """

        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        st.write("💬 **Asistan:**", response.text)
    else:
        st.warning("Lütfen bir soru yaz.")
