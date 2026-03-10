import fs from "node:fs/promises";
import path from "node:path";

function cosineSimilarity(left, right) {
  let dotProduct = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dotProduct += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }

  if (!leftNorm || !rightNorm) {
    return 0;
  }

  return dotProduct / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

export class LocalVectorStore {
  constructor(baseDirectory) {
    this.baseDirectory = path.resolve(baseDirectory);
  }

  async hasCollection(name) {
    try {
      await fs.access(this.getCollectionPath(name));
      return true;
    } catch {
      return false;
    }
  }

  async count(name) {
    const collection = await this.getCollection(name);
    return collection.items.length;
  }

  async deleteCollection(name) {
    try {
      await fs.rm(this.getCollectionPath(name), { force: true });
    } catch {
      return;
    }
  }

  async saveCollection(name, items) {
    await this.ensureBaseDirectory();
    const payload = {
      name,
      savedAt: new Date().toISOString(),
      items
    };
    await fs.writeFile(this.getCollectionPath(name), JSON.stringify(payload, null, 2), "utf8");
  }

  async getCollection(name) {
    const raw = await fs.readFile(this.getCollectionPath(name), "utf8");
    const payload = JSON.parse(raw);
    return {
      name: payload.name,
      items: Array.isArray(payload.items) ? payload.items : []
    };
  }

  async getDocuments(name) {
    const collection = await this.getCollection(name);
    return collection.items
      .slice()
      .sort((left, right) => (left.metadata?.chunkIndex || 0) - (right.metadata?.chunkIndex || 0));
  }

  async query(name, queryEmbedding, topK) {
    const collection = await this.getCollection(name);
    return collection.items
      .map((item) => {
        const similarity = cosineSimilarity(queryEmbedding, item.embedding || []);
        return {
          id: item.id,
          text: item.document,
          metadata: item.metadata || {},
          similarity,
          distance: 1 - similarity
        };
      })
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, topK);
  }

  getCollectionPath(name) {
    return path.join(this.baseDirectory, `${name}.json`);
  }

  async ensureBaseDirectory() {
    await fs.mkdir(this.baseDirectory, { recursive: true });
  }
}
