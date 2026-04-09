import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (OllamaEmbeddingFunction)

# Load the profile document
with open('profile.txt', "r") as f:
    text = f.read()

# Split the text into chunks by parapgraph = each blank line becomes a split point
# strip() removes extra whitespace and the if check skips empty chunks
chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]

print(f"Loaded {len(chunks)} chunks from profile.txt")

#Initialize ChromaDB - PersistentClient saves data to disk so it survives restarts
client = chromadb.PersistentClient(path="./chroma_db")

# Connect to Ollama's embedding model to convert text into vectors
ef = OllamaEmbeddingFunction(model_name = "nomic-embed-text", url = "http://localhost:11434")

# Create (or reuse) a collection - like a table in a database
collection = client.get_or_create_collection(name="personal_profile", embedding_function=ef)

# Add the chunks to the collection with unique IDs - ChromaDB automatically generates embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],  # Unique IDs for each chunk
    documents=chunks,  # The actual text content
    metadatas=[{"source": "profile.txt", "chunk_index": i} for i in range(len(chunks))]  # Optional metadata for each chunk
)

print(f"Added {len(chunks)} chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")