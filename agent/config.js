import "dotenv/config";

function getNumber(name, fallback) {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }

  const value = Number(raw);
  if (Number.isNaN(value)) {
    throw new Error(`Environment variable ${name} must be a valid number.`);
  }

  return value;
}

export function getSettings() {
  const geminiApiKey = process.env.GEMINI_API_KEY?.trim();
  if (!geminiApiKey) {
    throw new Error("Missing GEMINI_API_KEY. Add it to your environment or .env file.");
  }

  const chunkSize = getNumber("CHUNK_SIZE", 1200);
  const chunkOverlap = getNumber("CHUNK_OVERLAP", 200);

  if (chunkOverlap >= chunkSize) {
    throw new Error("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.");
  }

  return {
    geminiApiKey,
    geminiModel: process.env.GEMINI_MODEL?.trim() || "gemini-2.5-flash",
    embeddingModel: process.env.GEMINI_EMBEDDING_MODEL?.trim() || "gemini-embedding-001",
    collectionPrefix: process.env.COLLECTION_PREFIX?.trim() || "pdf",
    chunkSize,
    chunkOverlap,
    topK: getNumber("TOP_K", 4),
    summaryGroupSize: getNumber("SUMMARY_GROUP_SIZE", 8),
    embeddingDimensions: getNumber("EMBEDDING_DIMENSIONS", 3072),
    localVectorStoreDir: process.env.LOCAL_VECTOR_STORE_DIR?.trim() || "storage/local-vectors"
  };
}
