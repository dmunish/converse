import os
import re

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(
    model_name=os.getenv("EMBED_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
)

_DATE_RE = re.compile(
    r"(?:Last reviewed|Last updated|Effective(?: date)?|Updated|Reviewed)"
    r"\s*:?\s*([A-Z][a-z]+ \d{4})"
)


def _attach_date(node) -> None:
    m = _DATE_RE.search(node.text or "")
    if m:
        node.metadata["last_reviewed"] = m.group(1)


docs = SimpleDirectoryReader("./docs").load_data()

pipeline = IngestionPipeline(transformations=[MarkdownNodeParser()])
nodes = pipeline.run(documents=docs)

for n in nodes:
    _attach_date(n)

vector_store = LanceDBVectorStore(
    uri=os.getenv("LANCEDB_URI", "./lancedb"),
    table_name=os.getenv("LANCEDB_TABLE", "edtech_kb"),
    mode="overwrite",
    query_type="hybrid",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
index.storage_context.persist("./storage")
print(f"Ingested {len(docs)} docs / {len(nodes)} nodes into LanceDB.")