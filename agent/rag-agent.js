import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import { GoogleGenAI } from "@google/genai";
import pdf from "pdf-parse";

import { getSettings } from "./config.js";
import { LocalVectorStore } from "./local-vector-store.js";
import {
  buildQaPrompt,
  buildSummaryPrompt,
  buildSummaryReducePrompt,
  QA_SYSTEM_PROMPT,
  SUMMARY_REDUCE_SYSTEM_PROMPT,
  SUMMARY_SYSTEM_PROMPT
} from "./prompts.js";

function normalizeWhitespace(text) {
  return text.replace(/\r/g, "").replace(/\t/g, " ").replace(/\u0000/g, "").trim();
}

function splitTextIntoChunks(text, chunkSize, chunkOverlap) {
  const normalized = normalizeWhitespace(text);
  if (!normalized) {
    return [];
  }

  const chunks = [];
  let start = 0;

  while (start < normalized.length) {
    let end = Math.min(start + chunkSize, normalized.length);
    if (end < normalized.length) {
      const lastBoundary = Math.max(
        normalized.lastIndexOf("\n\n", end),
        normalized.lastIndexOf(". ", end),
        normalized.lastIndexOf(" ", end)
      );
      if (lastBoundary > start + Math.floor(chunkSize * 0.6)) {
        end = lastBoundary + 1;
      }
    }

    const chunk = normalized.slice(start, end).trim();
    if (chunk) {
      chunks.push(chunk);
    }

    if (end >= normalized.length) {
      break;
    }

    start = Math.max(end - chunkOverlap, start + 1);
  }

  return chunks;
}

function groupItems(items, groupSize) {
  const groups = [];
  for (let index = 0; index < items.length; index += groupSize) {
    groups.push(items.slice(index, index + groupSize));
  }
  return groups;
}

function getResponseText(response) {
  return response?.text?.trim() || "";
}

export class PdfRagAgent {
  constructor(settings = getSettings()) {
    this.settings = settings;
    this.genAI = new GoogleGenAI({ apiKey: settings.geminiApiKey });
    this.localVectorStore = new LocalVectorStore(settings.localVectorStoreDir);
  }

  async extractDocument(pdfPath) {
    const resolvedPath = path.resolve(pdfPath);
    const buffer = await fs.readFile(resolvedPath);
    const parsed = await pdf(buffer);
    const text = normalizeWhitespace(parsed.text || "");

    if (!text) {
      throw new Error("No text could be extracted from the PDF.");
    }

    return {
      pdfPath: resolvedPath,
      text,
      pageCount: parsed.numpages || 0,
      info: parsed.info || {}
    };
  }

  async indexPdf(pdfPath, { rebuild = false } = {}) {
    const document = await this.extractDocument(pdfPath);
    const collectionName = await this.buildCollectionName(document.pdfPath);
    const vectorStoreBackend = "local";

    if (rebuild) {
      await this.deleteCollectionIfExists(collectionName);
    }

    const existing = await this.tryGetCollection(collectionName, vectorStoreBackend);
    if (existing) {
      return {
        pdfPath: document.pdfPath,
        collectionName,
        pageCount: document.pageCount,
        chunkCount: await this.getCollectionCount(collectionName, vectorStoreBackend),
        reusedExisting: true,
        vectorStoreBackend
      };
    }

    const chunks = splitTextIntoChunks(document.text, this.settings.chunkSize, this.settings.chunkOverlap);
    if (!chunks.length) {
      throw new Error("The PDF was loaded, but no chunks were created from it.");
    }

    const embeddings = await this.embedTexts(chunks, "RETRIEVAL_DOCUMENT");

    const ids = chunks.map((_, index) => `${collectionName}-chunk-${index + 1}`);
    const metadatas = chunks.map((chunk, index) => ({
      chunkIndex: index,
      chunkId: index + 1,
      pdfPath: document.pdfPath,
      length: chunk.length
    }));

    await this.saveCollection(collectionName, vectorStoreBackend, {
      ids,
      documents: chunks,
      embeddings,
      metadatas
    });

    return {
      pdfPath: document.pdfPath,
      collectionName,
      pageCount: document.pageCount,
      chunkCount: chunks.length,
      reusedExisting: false,
      vectorStoreBackend
    };
  }

  async summarizePdf(pdfPath, { rebuild = false } = {}) {
    const indexResult = await this.indexPdf(pdfPath, { rebuild });
    const chunks = await this.getStoredChunks(indexResult.collectionName, indexResult.vectorStoreBackend);

    const grouped = groupItems(chunks, this.settings.summaryGroupSize);
    const partials = [];

    for (const group of grouped) {
      const context = this.formatContext(group);
      const response = await this.generateText({
        contents: buildSummaryPrompt(context),
        systemInstruction: SUMMARY_SYSTEM_PROMPT,
        temperature: 0.2
      });
      partials.push(getResponseText(response));
    }

    const finalResponse = await this.generateText({
      contents: buildSummaryReducePrompt(partials.join("\n\n")),
      systemInstruction: SUMMARY_REDUCE_SYSTEM_PROMPT,
      temperature: 0.2
    });

    return {
      collectionName: indexResult.collectionName,
      chunkCount: chunks.length,
      summary: getResponseText(finalResponse),
      vectorStoreBackend: indexResult.vectorStoreBackend
    };
  }

  async answerQuestion(pdfPath, question, { rebuild = false } = {}) {
    const indexResult = await this.indexPdf(pdfPath, { rebuild });
    const [queryEmbedding] = await this.embedTexts([question], "RETRIEVAL_QUERY");
    const chunks = await this.queryStoredChunks(
      indexResult.collectionName,
      indexResult.vectorStoreBackend,
      queryEmbedding
    );

    const response = await this.generateText({
      contents: buildQaPrompt(question, this.formatContext(chunks)),
      systemInstruction: QA_SYSTEM_PROMPT,
      temperature: 0.2
    });

    return {
      collectionName: indexResult.collectionName,
      answer: getResponseText(response),
      sourceChunks: chunks,
      vectorStoreBackend: indexResult.vectorStoreBackend
    };
  }

  async embedTexts(texts, taskType) {
    const response = await this.genAI.models.embedContent({
      model: this.settings.embeddingModel,
      contents: texts,
      config: {
        taskType,
        outputDimensionality: this.settings.embeddingDimensions
      }
    });

    return (response.embeddings || []).map((item) => item.values);
  }

  async generateText({ contents, systemInstruction, temperature = 0.2 }) {
    return this.genAI.models.generateContent({
      model: this.settings.geminiModel,
      contents,
      config: {
        systemInstruction,
        temperature
      }
    });
  }

  formatContext(chunks) {
    return chunks
      .map((chunk, index) => {
        const chunkId = chunk.metadata?.chunkId || index + 1;
        return `[Chunk ${chunkId}]\n${chunk.text}`;
      })
      .join("\n\n");
  }

  async buildCollectionName(pdfPath) {
    const stats = await fs.stat(pdfPath);
    const fingerprint = crypto
      .createHash("sha256")
      .update(`${pdfPath}:${stats.size}:${stats.mtimeMs}`)
      .digest("hex")
      .slice(0, 16);

    return `${this.settings.collectionPrefix}_${fingerprint}`;
  }

  async tryGetCollection(name) {
    return this.tryGetCollection(name, "local");
  }

  async tryGetCollection(name, backend) {
    return (await this.localVectorStore.hasCollection(name)) ? { name } : null;
  }

  async getCollectionCount(name, backend) {
    return this.localVectorStore.count(name);
  }

  async saveCollection(name, backend, payload) {
    const items = payload.ids.map((id, index) => ({
      id,
      document: payload.documents[index],
      embedding: payload.embeddings[index],
      metadata: payload.metadatas[index] || {}
    }));
    await this.localVectorStore.saveCollection(name, items);
  }

  async getStoredChunks(name, backend) {
    const items = await this.localVectorStore.getDocuments(name);
    return items.map((item) => ({
      text: item.document,
      metadata: item.metadata || {}
    }));
  }

  async queryStoredChunks(name, backend, queryEmbedding) {
    return this.localVectorStore.query(name, queryEmbedding, this.settings.topK);
  }


  async deleteCollectionIfExists(name) {
    await this.localVectorStore.deleteCollection(name);
  }
}
