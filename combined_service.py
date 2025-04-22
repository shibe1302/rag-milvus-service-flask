# combined_service.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from ollama import chat
import time

app = Flask(__name__)

# Global model and client initializations that happen only once
print("Loading models and initializing connections...")
start_time = time.time()

# Load SBERT model once
sbert_model = SentenceTransformer.load("./models/vietnamese-sbert")
print(f"SBERT model loaded in {time.time() - start_time:.2f} seconds")

# Initialize Milvus client once
milvus_client = MilvusClient(uri="http://localhost:19530")
ranker = RRFRanker(100)
collection_name = "hybrid_search_collection"

# System prompt for LLM
system_prompt = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
please reply in VIETNAMESE.
"""

@app.route('/embed', methods=['POST'])
def embed():
    """Endpoint to create embeddings from text"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Create embeddings
    try:
        embedding = sbert_model.encode(text).tolist()
        return jsonify({"embedding": embedding})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def hybrid_search():
    """Endpoint to perform hybrid search"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    query = request.json.get('query')
    limit = request.json.get('limit', 2)
    dense_nprobe = request.json.get('dense_nprobe', 10)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Create embedding for the query
        query_embedding = sbert_model.encode(query).tolist()
        
        # Dense (vector) search request
        search_param_1 = {
            "data": [query_embedding],
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
        results = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=["text"]
        )
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/answer', methods=['POST'])
def get_answer():
    """Endpoint to get an answer using the entire pipeline"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # 1. Get context using hybrid search
        query_embedding = sbert_model.encode(query).tolist()
        
        # Dense (vector) search request
        search_param_1 = {
            "data": [query_embedding],
            "anns_field": "dense",
            "param": {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            },
            "limit": 2
        }
        request_1 = AnnSearchRequest(**search_param_1)
        
        # Sparse (keyword) search request
        search_param_2 = {
            "data": [query],
            "anns_field": "sparse",
            "param": {
                "metric_type": "BM25",
            },
            "limit": 2
        }
        request_2 = AnnSearchRequest(**search_param_2)
        
        reqs = [request_1, request_2]
        
        # Execute hybrid search
        results = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=2,
            output_fields=["text"]
        )
        
        # Extract text from results
        retrieved_lines = [data_row["entity"]["text"] for data_row in results[0]]
        context = ''.join([text + '\n' for text in retrieved_lines])
        
        # 2. Format the user prompt
        user_prompt = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{query}
</question>
DO NOT CONTAIN TAG TOKEN LIKE <context>,<question>.. IN RESPONSE(YOUR ANSWER) !
"""
        
        # 3. Generate response using ollama
        response = chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        return jsonify({
            "query": query,
            "context": context,
            "answer": response["message"]["content"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Service is ready to use!")
    # Run the service on port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)