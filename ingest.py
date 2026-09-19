"""Run: python ingest.py"""
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(
    model_name=os.getenv("EMBED_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
)

docs = SimpleDirectoryReader("./docs").load_data()

vector_store = LanceDBVectorStore(
    uri=os.getenv("LANCEDB_URI", "./lancedb"),
    table_name=os.getenv("LANCEDB_TABLE", "edtech_kb"),
    mode="overwrite",
    query_type="hybrid",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
index.storage_context.persist("./storage")
print(f"Ingested {len(docs)} docs into LanceDB.")