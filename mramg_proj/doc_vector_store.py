from typing import Any, Dict, List

from mramg_proj.bge_m3_embedder import BgeM3Embedder
from mramg_proj.doc_vector_store_config import DocVectorStoreConfig
from mramg_proj.document_processor import DocumentProcessor
from mramg_proj.milvus_manager import MilvusManager


class DocVectorStore:
    """
    Doc 벡터 저장소 클래스
    Args:
        config: DocVectorStoreConfig
    """

    def __init__(self, config: DocVectorStoreConfig):
        self.embedder = BgeM3Embedder()
        self.processor = DocumentProcessor(config.chunk_size, config.chunk_overlap)
        self.db = MilvusManager(
            host=config.milvus_host,
            port=config.milvus_port,
            collection_name=config.collection_name,
            embedding_dim=config.embedding_dim,
            text_max_len=config.text_max_len,
            image_ids_max_len=config.image_ids_max_len,
        )

    def insert_documents(self, file_path: str) -> List[int]:
        docs = self.processor.process_jsonl(file_path)
        embeddings = self.embedder.embed([d["text"] for d in docs])
        return self.db.insert(docs, embeddings)

    def create_index(self) -> None:
        self.db.create_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed([query])[0]
        return self.db.search(query_vec, top_k=top_k)
