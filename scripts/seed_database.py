import os
import pymongo
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---

# Load environment variables from a .env file
load_dotenv()

# Get credentials from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Generative AI SDK
genai.configure(api_key=GEMINI_API_KEY)

# --- Knowledge Base Content ---

# This is our sample knowledge base. In a real application,
# this would come from a file, a database, or an external API.
KNOWLEDGE_BASE = [
    "Redis is an in-memory data store known for its speed and is often used for caching, session management, and real-time analytics.",
    "MongoDB Atlas is a multi-cloud database service that provides a flexible document model and native support for vector search, making it ideal for AI applications.",
    "Vercel is a cloud platform for frontend developers, providing the frameworks, workflows, and infrastructure to build and deploy web applications at the edge.",
    "GroqCloud offers ultra-low-latency inference for large language models, making it a powerful choice for real-time AI applications.",
    "A RAG (Retrieval-Augmented Generation) pipeline combines a retrieval system (like vector search) with a generative model to produce context-aware responses.",
    "The combination of Vercel, MongoDB Atlas, Redis, and a fast AI model like Groq or Gemini creates a powerful, scalable, and high-performance modern AI stack.",
]

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for a given text using the Gemini model."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: '{text[:30]}...'")
        print(f"Error: {e}")
        return None

def main():
    """Connects to MongoDB, generates embeddings, and seeds the database."""
    print("Starting database seeding process...")

    # --- 1. Connect to MongoDB ---
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        db = client.get_database("ai_stack_db")
        collection = db.get_collection("knowledge_base")
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    # --- 2. Clear existing data ---
    print(f"Clearing existing documents from the '{collection.name}' collection...")
    collection.delete_many({})

    # --- 3. Generate embeddings and insert data ---
    print("Generating embeddings and inserting data into the database...")
    documents_to_insert = []
    for text in KNOWLEDGE_BASE:
        embedding = get_embedding(text)
        if embedding:
            documents_to_insert.append({
                "text": text,
                "embedding": embedding,
            })

    if documents_to_insert:
        collection.insert_many(documents_to_insert)
        print(f"Successfully inserted {len(documents_to_insert)} documents into the database.")
    else:
        print("No documents were inserted. Please check for embedding generation errors.")

    print("\nDatabase seeding process complete.")
    client.close()

if __name__ == "__main__":
    main()