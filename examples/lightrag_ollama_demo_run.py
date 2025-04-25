import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
import asyncio
import nest_asyncio
import fitz

nest_asyncio.apply()
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "/Users/subhailamathy/Documents/MS/Sp25/CS532/Project/LightRAG/working_dir"
file_path = "/Users/subhailamathy/Documents/MS/Sp25/CS532/Project/LightRAG/inputs/papers.txt"

ngrok_url = "http://localhost:11434"

# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "light_rag"
os.environ["NEO4J_DATABASE"] = "neo4j"  # Explicitly use the default database
# os.environ["NEO4J_apoc_export_file_enabled"] = "true"
# os.environ["NEO4J_apoc_import_file_enabled"] = "true"
# os.environ["NEO4J_apoc_import_file_use__neo4j__config"] = "true"


# milvus
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "Milvus"
os.environ["MILVUS_DB_NAME"] = "default"


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="mistral-nemo",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": ngrok_url,
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts=texts, embed_model="nomic-embed-text", host=ngrok_url
            ),
        ),
        #kv_storage="MongoKVStorage",
        # graph_storage="Neo4JStorage",
        vector_storage="MilvusVectorDBStorage",
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Insert example text
    with open(file_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # pdf_text = extract_text_from_pdf("/Users/subhailamathy/Documents/MS/Sp25/CS532/Project/LightRAG/inputs/2502.15365v1.pdf")
    # rag.insert(pdf_text)
    # prompt = "Extract key themes and insights from research papers in my knowledge base."
    prompt = "Summarize top themes in the paper."

    # prompt = "Summarize recent research studies on features shaping perceived consciousness in LLM-based AI. Extract key themes and insights from research papers in my knowledge base."

    # # Test different query modes
    print("\nNaive Search:")
    print(
        rag.query(
            prompt, param=QueryParam(mode="naive")
        )
    )

    print("\nLocal Search:")
    print(
        rag.query(
            prompt, param=QueryParam(mode="local")
        )
    )

    print("\nGlobal Search:")
    print(
        rag.query(
            prompt, param=QueryParam(mode="global")
        )
    )

    print("\nHybrid Search:")
    print(
        rag.query(
            prompt, param=QueryParam(mode="hybrid")
        )
    )


if __name__ == "__main__":
    main()
