import os
import psycopg2
import google.generativeai as genai
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

# --- Configuration ---

# Load environment variables from a .env file
load_dotenv()

# Get credentials from environment variables
POSTGRES_URL = os.getenv("POSTGRES_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Generative AI SDK
genai.configure(api_key=GEMINI_API_KEY)

# --- Knowledge Base Content ---
KNOWLEDGE_BASE = [
    "Supabase is an open-source Firebase alternative that provides a suite of tools for building applications, including a Postgres database, authentication, and storage.",
    "pgvector is a PostgreSQL extension that enables storing and querying vector embeddings, making it a powerful tool for AI applications like similarity search.",
    "Vercel is a cloud platform for frontend developers, providing the frameworks, workflows, and infrastructure to build and deploy web applications at the edge.",
    "A RAG (Retrieval-Augmented Generation) pipeline combines a retrieval system (like vector search in Postgres) with a generative model to produce context-aware responses.",
    "Using Supabase with Vercel allows for a tightly integrated, serverless-first architecture where the database connection is managed automatically via environment variables.",
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
    """Connects to Supabase, sets up the database, and inserts data."""
    print("Starting Supabase database seeding process...")

    conn = None
    try:
        # --- 1. Connect to Supabase Postgres ---
        conn = psycopg2.connect(POSTGRES_URL)
        cur = conn.cursor()
        print("Successfully connected to Supabase.")

        # --- 2. Set up the database schema ---
        print("Setting up the database schema...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn) # Register the vector type with psycopg2

        # The vector dimension is 768 for the 'embedding-001' model
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id serial PRIMARY KEY,
                content TEXT,
                embedding vector(768)
            );
        """)

        # --- 3. Clear existing data ---
        print("Clearing existing documents from the 'knowledge' table...")
        cur.execute("TRUNCATE TABLE knowledge;")

        # --- 4. Generate embeddings and insert data ---
        print("Generating embeddings and inserting data...")
        for text in KNOWLEDGE_BASE:
            embedding = get_embedding(text)
            if embedding:
                cur.execute(
                    "INSERT INTO knowledge (content, embedding) VALUES (%s, %s)",
                    (text, embedding)
                )

        conn.commit()
        print(f"Successfully inserted {len(KNOWLEDGE_BASE)} documents into the database.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print("\nDatabase seeding process complete.")

if __name__ == "__main__":
    main()