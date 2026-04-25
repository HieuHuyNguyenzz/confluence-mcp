"""Milvus Vector Database client for Confluence knowledge."""

import logging
from typing import Any, List, Dict
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

log = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self, url: str, collection_name: str, token: str = None):
        self.url = url
        self.token = token
        self.collection_name = collection_name
        self._dim = None
        self._connect()

    def _connect(self):
        if self.token:
            connections.connect("default", uri=self.url, token=self.token)
            log.info(f"Connected to Milvus at {self.url} (with token)")
        else:
            connections.connect("default", uri=self.url)
            log.info(f"Connected to Milvus at {self.url}")

    def set_dimension(self, dim: int):
        self._dim = dim

    def ensure_collection(self):
        if not self._dim:
            raise ValueError("Dimension not set. Run embedding test first.")
        
        if utility.has_collection(self.collection_name):
            log.info(f"Dropping existing collection '{self.collection_name}'...")
            utility.drop_collection(self.collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="group", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="column_type", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="aggregation", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="source_title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="global_context", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
        ]
        schema = CollectionSchema(fields, description="Confluence Knowledge Base")
        collection = Collection(name=self.collection_name, schema=schema)
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        )
        log.info(f"Created collection '{self.collection_name}' with dim={self._dim}")

    def upload_batch(self, batch: List[Dict[str, Any]]):
        collection = Collection(self.collection_name)
        collection.load()
        
        data = [
            {
                "id": c["id"],
                "vector": c["embedding"],
                "content": c["content"][:65534],
                "group": c["group"],
                "domain": c["domain"],
                "column_type": c["column_type"],
                "aggregation": c["aggregation"],
                "keywords": ", ".join(c["keywords"]) if isinstance(c["keywords"], list) else c["keywords"],
                "source_file": c["source_file"],
                "source_title": c["source_title"],
                "heading": c["heading"],
                "global_context": c.get("global_context", ""),
                "parent_id": c.get("parent_id", ""),
                "chunk_type": c.get("chunk_type", "child"),
            }
            for c in batch
        ]
        collection.insert(data=data)
        collection.flush()

    def search_and_get_parents(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top-k child chunks and return their parent context."""
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["parent_id", "content", "chunk_type"]
        )
        
        if not results or not results[0]:
            return []
            
        parent_ids = set()
        for hit in results[0]:
            p_id = hit.entity.get("parent_id")
            if p_id:
                parent_ids.add(p_id)
            else:
                parent_ids.add(hit.id)
        
        if not parent_ids:
            return []
            
        filter_expr = f"id in {tuple(parent_ids)}" if len(parent_ids) > 1 else f"id == '{list(parent_ids)[0]}'"
        
        parents = collection.query(
            expr=filter_expr,
            output_fields=["content", "source_title"]
        )
        return parents
