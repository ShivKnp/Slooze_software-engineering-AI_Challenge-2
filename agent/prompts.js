export const SUMMARY_SYSTEM_PROMPT = `You are an expert research analyst.
Produce a concise but complete summary using only the provided document context.
Highlight the document's objective, methodology, major findings, and conclusion.
If the context is incomplete, say so instead of inventing details.`;

export const SUMMARY_REDUCE_SYSTEM_PROMPT = `You are combining partial summaries of one PDF document.
Create one final summary with sections for Objective, Methodology, Findings, and Conclusion.
Only include claims supported by the summaries you were given.`;

export const QA_SYSTEM_PROMPT = `You are a grounded question-answering assistant.
Answer strictly from the retrieved PDF context.
If the answer is not supported by the context, clearly say that the document context does not provide enough evidence.
Keep the answer direct and cite chunk ids when relevant.`;

export function buildSummaryPrompt(context) {
  return `Summarize this portion of the PDF.\n\nContext:\n${context}\n\nReturn a short structured summary.`;
}

export function buildSummaryReducePrompt(partialSummaries) {
  return `Combine these partial summaries into one document summary.\n\nPartial summaries:\n${partialSummaries}\n\nReturn a grounded final summary with the headings Objective, Methodology, Findings, and Conclusion.`;
}

export function buildQaPrompt(question, context) {
  return `Question: ${question}\n\nRetrieved context:\n${context}\n\nAnswer the question using only the retrieved context.`;
}
