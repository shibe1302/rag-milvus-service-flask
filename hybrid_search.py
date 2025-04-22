import requests
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

class HybridSearch:
    def __init__(self, embedding_service_url="http://localhost:5000/embed", milvus_uri="http://localhost:19530", collection_name="hybrid_search_collection"):
        """
        Initialize the HybridSearch class.
        
        Args:
            embedding_service_url (str): URL of the embedding service
            milvus_uri (str): URI for Milvus connection
            collection_name (str): Name of the collection to search in
        """
        self.embedding_service_url = embedding_service_url
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.ranker = RRFRanker(100)
    
    def emb_text(self, text):
        """Create embedding from text using the embedding service."""
        try:
            response = requests.post(
                self.embedding_service_url,
                json={"text": text},
                timeout=10  # Set a reasonable timeout
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()["embedding"]
        except requests.RequestException as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def search(self, query, limit=2, dense_nprobe=10):
        # Dense (vector) search request
        search_param_1 = {
            "data": [self.emb_text(query)],
            "anns_field": "dense",
            "param": {
                "metric_type": "IP",
                "params": {"nprobe": dense_nprobe}
            },
            "limit": limit
        }
        request_1 = AnnSearchRequest(**search_param_1)
        
        # Sparse (keyword) search request
        search_param_2 = {
            "data": [query],
            "anns_field": "sparse",
            "param": {
                "metric_type": "BM25",
            },
            "limit": limit
        }
        request_2 = AnnSearchRequest(**search_param_2)
        
        reqs = [request_1, request_2]
        
        # Execute hybrid search
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=self.ranker,
            limit=limit,
            output_fields=["text"]
        )
        
        return results

# For direct execution
if __name__ == "__main__":
    searcher = HybridSearch()
    query = input("Ban hay hoi di: ")
    results = searcher.search(query)
    print(results)