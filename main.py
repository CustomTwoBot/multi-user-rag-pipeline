from fastapi import FastAPI
from pydantic import BaseModel # Pydantic validates incoming request data
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (OllamaEmbeddingFunction)

app = FastAPI() # Create the FastAPI application

client = chromadb.PersistentClient(path="./chroma_db") # Connects to the same ChromaDB collection as in base.py

ef = OllamaEmbeddingFunction(model_name = "nomic-embed-text", url = "http://localhost:11434") # Create an embedding function using the Ollama model
collection = client.get_or_create_collection(name="personal_profile", embedding_function=ef) # Get or create a ChromaDB collection with the specified embedding function

# Define the expected shape of incoming data for the POST endpoint
class DocumentSubmission(BaseModel):
    user_name: str # Who this profile belongs to
    content: str # The profile text to store

@app.post("/documents") # POST endpoint - accepts data in the request body
def add_document(submission: DocumentSubmission):
    # Split the submitted profile into chunks by paragraph
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    # Store each chunk in ChromaDB with the user's name attached as metadata
    collection.add(
        ids = [f"{submission.user_name}--chunk{i}" for i in range(len(chunks))],
        documents = chunks,
        metadatas = [{"source": "profile", "user_name": submission.user_name, "chunk_index": i}
                     for i in range(len(chunks))] # user_name metadata lets us filter by user
    )

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks)
    }

@app.get("/ask") # Define a GET endpoint at /ask
def ask(question: str, user: str = None): # FastAPI automatically reads "question" from the URL query string, user is optional (None means search all profiles)

    # Build the query parameters
    query_params = {
        "query_texts": [question],
        "n_results": 2 #  Return the 2 most relevant chunks
    }

    # If a user name was provided, only search that user's profile chunks
    if user:
        query_params["where"] = {"user_name": user} # ChromaDB metadata filter

    # Step 1: RETRIEVE - search ChromaDB for the most relevant chunks
    results = collection.query(**query_params) # unpacks the dictionary as keyword augments
    context = "\n\n".join(results["documents"][0])
    # Step 2: AUGMENT - build a prompt that includes the retrieved context
    augmented_prompt = f"""Use the following context to answer the question. 
    If the context doesn't contain relevant information, say so.

    Context: 
    
    {context}

    Question: {question}"""

        # Step 3: GENERATE - send the augmented prompt to the local LLM
    response = ollama.chat(
            model = "qwen2.5:0.5b", 
            messages = [{"role": "user", "content": augmented_prompt}]
    )
    
    return {
            "question": question,
            "answer": response["message"]["content"],
            "context_used": results["documents"][0],
            "filtered_by_user": user, # Shows the user was filtered (or None for all users)
    }
    