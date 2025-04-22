from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import time

app = Flask(__name__)

# Load model once when the service starts
print("Loading model...")
start_time = time.time()
model = SentenceTransformer("./models/vietnamese-sbert")
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

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
        embedding = model.encode(text).tolist()
        return jsonify({"embedding": embedding})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the service on port 5000
    app.run(host='0.0.0.0', port=5000)