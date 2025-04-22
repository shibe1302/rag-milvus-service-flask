# client_example.py
import requests
import json

def get_embedding(text, service_url="http://localhost:5000/embed"):
    """Get embedding for text"""
    response = requests.post(
        service_url,
        json={"text": text}
    )
    return response.json()

def perform_search(query, service_url="http://localhost:5000/search"):
    """Perform hybrid search"""
    response = requests.post(
        service_url,
        json={"query": query}
    )
    return response.json()

def get_answer(query, service_url="http://localhost:5000/answer"):
    """Get answer to a question"""
    response = requests.post(
        service_url,
        json={"query": query}
    )
    return response.json()

if __name__ == "__main__":
    # Example queries
    query = "Ban Lãnh đạo Khoa hiện tại là những ai (trưởng khoa, phó khoa) ?"
    
    # Get just the embedding
    # embedding_result = get_embedding(query)
    # print("Embedding (first 5 values):", embedding_result["embedding"][:5])
    # print()
    
    # Get search results
    # search_result = perform_search(query)
    # print("Search results:")
    # print(json.dumps(search_result, indent=2, ensure_ascii=False))
    # print()
    
    # Get full answer
    answer_result = get_answer(query)
    print("Complete answer:")
    print(f"Query: {answer_result['query']}")
    print(f"Answer: {answer_result['answer']}")