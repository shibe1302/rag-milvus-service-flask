# search_app.py
from hybrid_search import HybridSearch
from ollama import chat
from ollama import ChatResponse

class SearchApp:
    def __init__(self):
        self.hybrid_search = HybridSearch()
        self.system_prompt = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
please reply in VIETNAMESE.
"""

    def get_context(self, query):
        """Retrieve relevant information using hybrid search."""
        res = self.hybrid_search.search(query=query)
        retrieved_lines = [data_row["entity"]["text"] for data_row in res[0]]
        return ''.join([text + '\n' for text in retrieved_lines])

    def format_prompt(self, query, context):
        """Format the prompt with context and question."""
        return f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{query}
</question>
DO NOT CONTAIN TAG TOKEN LIKE <context>,<question>.. IN RESPONSE(YOUR ANSWER) !
"""

    def get_answer(self, query):
        """Main function to get answer from query."""
        # Get context using hybrid search
        context = self.get_context(query)
        
        # Format the user prompt
        user_prompt = self.format_prompt(query, context)
        
        # Generate response using ollama
        response: ChatResponse = chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        return response["message"]["content"]


# Example usage when run directly
if __name__ == "__main__":
    search_app = SearchApp()
    question = "mục tiêu và tầm nhìn của khoa ?"
    answer = search_app.get_answer(question)
    print(answer)