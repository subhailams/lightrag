import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


WORKING_DIR = "/Users/subhailamathy/Documents/MS/Sp25/CS532/Project/LightRAG/working_dir"
file_path = "/Users/subhailamathy/Documents/MS/Sp25/CS532/Project/LightRAG/inputs/papers.txt"

ngrok_url = "http://localhost:11434"
OPEN_AI_KEY = ""

# redis
os.environ["REDIS_URI"] = "redis://localhost:6379"
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



async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=OPEN_AI_KEY
        # base_url="",
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=512,
    func=lambda texts: ollama_embed(
        texts, embed_model="nomic-embed-text", host=ngrok_url
    ),
)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_max_token_size=32768,
        embedding_func=embedding_func,
        chunk_token_size=512,
        chunk_overlap_token_size=256,
        # kv_storage="RedisKVStorage",
        # graph_storage="Neo4JStorage",
        # vector_storage="MilvusVectorDBStorage",
        # doc_status_storage="RedisKVStorage",
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Insert example text
    with open(file_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())


    prompt = "Extract key themes and insights from research papers in my knowledge base."

    # prompt = "Summarize top themes in the paper."

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
