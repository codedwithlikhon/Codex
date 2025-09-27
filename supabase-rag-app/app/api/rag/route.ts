import { StreamingTextResponse, streamText } from 'ai';
import { groq } from '@ai-sdk/groq';
import postgres from 'postgres';
import { z } from 'zod';
import { NextRequest } from 'next/server';

// --- Environment Variable Validation ---
const POSTGRES_URL = process.env.POSTGRES_URL;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!POSTGRES_URL || !GROQ_API_KEY || !GEMINI_API_KEY) {
  throw new Error('Missing required environment variables.');
}

// --- Client Initialization ---
const sql = postgres(POSTGRES_URL, {
  // pgvector requires vector type to be registered
  types: {
    vector: {
      to: 2951, // OID for vector type
      from: [2951],
      serialize: (value: number[]) => `[${value.join(',')}]`,
      parse: (value: string) => value.substring(1, value.length - 1).split(',').map(Number),
    },
  },
});

const groqModel = groq(GROQ_API_KEY);

// --- Request Body Schema ---
const requestSchema = z.object({
  prompt: z.string(),
});

// --- Helper Function: Generate Embedding ---
async function getEmbedding(text: string): Promise<number[]> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=${GEMINI_API_KEY}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'models/embedding-001',
      content: { parts: [{ text }] },
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to generate embedding: ${await response.text()}`);
  }
  const data = await response.json();
  return data.embedding.values;
}

// --- Main API Route Handler ---
export async function POST(req: NextRequest) {
  try {
    // 1. Validate Request Body
    const body = await req.json();
    const parsedRequest = requestSchema.safeParse(body);
    if (!parsedRequest.success) {
      return new Response(JSON.stringify(parsedRequest.error), { status: 400 });
    }
    const { prompt } = parsedRequest.data;

    // 2. Generate Embedding for the Prompt
    const queryEmbedding = await getEmbedding(prompt);

    // 3. Perform Vector Similarity Search in Supabase
    // The <=> operator calculates the cosine distance (1 - cosine similarity)
    const documents = await sql`
      SELECT content, 1 - (embedding <=> ${sql.vector(queryEmbedding)}) as similarity
      FROM knowledge
      ORDER BY similarity DESC
      LIMIT 3;
    `;

    const context = documents.map(doc => doc.content).join('\n---\n');

    // 4. Construct Augmented Prompt
    const augmentedPrompt = `
      You are a helpful AI assistant. Answer the user's question based on the following context.
      If the context does not provide the answer, say so.

      Context:
      ${context}

      Question:
      ${prompt}
    `;

    // 5. Stream Response from Groq
    const result = await streamText({
      model: groqModel('llama3-8b-8192'),
      prompt: augmentedPrompt,
    });

    return new StreamingTextResponse(result.textStream);

  } catch (error) {
    console.error('Supabase RAG API Error:', error);
    return new Response('An internal server error occurred.', { status: 500 });
  }
}