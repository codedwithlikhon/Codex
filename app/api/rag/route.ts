import { StreamingTextResponse, streamText } from 'ai';
import { groq } from '@ai-sdk/groq';
import { MongoClient } from 'mongodb';
import Redis from 'ioredis';
import { z } from 'zod';
import { NextRequest } from 'next/server';

// --- Environment Variable Validation ---
const MONGODB_URI = process.env.MONGODB_URI;
const REDIS_URI = process.env.REDIS_URI;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!MONGODB_URI || !REDIS_URI || !GROQ_API_KEY || !GEMINI_API_KEY) {
  throw new Error('Missing required environment variables.');
}

// --- Client Initialization ---
// Initialize clients outside the request handler to be reused
const mongoClient = new MongoClient(MONGODB_URI);
const redisClient = new Redis(REDIS_URI);
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

    // 2. Check Redis Cache
    const cacheKey = `rag:${prompt.replace(/\s/g, '_')}`;
    const cachedResponse = await redisClient.get(cacheKey);
    if (cachedResponse) {
      return new Response(cachedResponse, { status: 200, headers: { 'X-Cache-Hit': 'true' } });
    }

    // 3. Generate Embedding for the Prompt
    const queryEmbedding = await getEmbedding(prompt);

    // 4. Perform Vector Search in MongoDB Atlas
    await mongoClient.connect();
    const db = mongoClient.db('ai_stack_db');
    const collection = db.collection('knowledge_base');

    const documents = await collection.aggregate([
      {
        $vectorSearch: {
          index: 'default',
          path: 'embedding',
          queryVector: queryEmbedding,
          numCandidates: 10,
          limit: 3,
        },
      },
      {
        $project: {
          _id: 0,
          text: 1,
          score: { $meta: 'vectorSearchScore' },
        },
      },
    ]).toArray();

    const context = documents.map(doc => doc.text).join('\n---\n');

    // 5. Construct Augmented Prompt
    const augmentedPrompt = `
      You are a helpful AI assistant. Answer the user's question based on the following context.
      If the context does not provide the answer, say so.

      Context:
      ${context}

      Question:
      ${prompt}
    `;

    // 6. Stream Response from Groq and Cache in Redis
    const result = await streamText({
      model: groqModel('llama3-8b-8192'),
      prompt: augmentedPrompt,
    });

    // Fork the stream to pipe it to the client and to the cache
    const [streamToClient, streamToCache] = result.textStream.tee();

    // Asynchronously cache the full response
    (async () => {
      let fullResponse = '';
      for await (const chunk of streamToCache) {
        fullResponse += chunk;
      }
      // Cache for 1 hour
      await redisClient.set(cacheKey, fullResponse, 'EX', 3600);
    })();

    return new StreamingTextResponse(streamToClient);

  } catch (error) {
    console.error('RAG API Error:', error);
    return new Response('An internal server error occurred.', { status: 500 });
  } finally {
    // Ensure the MongoDB client is closed after the request
    if (mongoClient) {
      await mongoClient.close();
    }
  }
}