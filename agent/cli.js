import { Command } from "commander";

import { PdfRagAgent } from "./rag-agent.js";

function normalizeCliArgs(argv) {
  const normalized = [...argv];
  const command = normalized[2];
  if (!["index", "summarize", "ask"].includes(command)) {
    return normalized;
  }

  const hasPdfFlag = normalized.includes("--pdf");
  const hasQuestionFlag = normalized.includes("--question");
  const positionalArgs = normalized.slice(3).filter((value) => !value.startsWith("--"));

  if (!hasPdfFlag && (command === "index" || command === "summarize") && positionalArgs.length >= 1) {
    const pdfPath = positionalArgs[0];
    const pdfIndex = normalized.indexOf(pdfPath, 3);
    if (pdfIndex !== -1) {
      normalized.splice(pdfIndex, 1, "--pdf", pdfPath);
    }
  }

  if (command === "ask") {
    if (!hasPdfFlag && positionalArgs.length >= 1) {
      const pdfPath = positionalArgs[0];
      const pdfIndex = normalized.indexOf(pdfPath, 3);
      if (pdfIndex !== -1) {
        normalized.splice(pdfIndex, 1, "--pdf", pdfPath);
      }
    }

    const refreshedPositionals = normalized.slice(3).filter((value) => !value.startsWith("--"));
    if (!hasQuestionFlag && refreshedPositionals.length >= 2) {
      const question = refreshedPositionals.slice(1).join(" ");
      const firstQuestionPartIndex = normalized.indexOf(refreshedPositionals[1], 3);
      if (firstQuestionPartIndex !== -1) {
        normalized.splice(firstQuestionPartIndex, refreshedPositionals.length - 1, "--question", question);
      }
    }
  }

  return normalized;
}

function registerSharedOptions(command) {
  return command
    .requiredOption("--pdf <path>", "Path to the PDF file")
    .option("--rebuild", "Force re-indexing for the PDF", false);
}

function formatSnippet(text, maxLength = 220) {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length <= maxLength ? normalized : `${normalized.slice(0, maxLength)}...`;
}

async function main() {
  const argv = normalizeCliArgs(process.argv);
  const program = new Command();

  program
    .name("pdf-rag-agent")
    .description("PDF summarization and question answering agent powered by Gemini and a local vector store.")
    .showHelpAfterError()
    .addHelpText(
      "after",
      `

Examples:
  node agent/cli.js summarize --pdf "C:\\path\\to\\document.pdf"
  node agent/cli.js ask --pdf "C:\\path\\to\\document.pdf" --question "What methodology was used?"

  npm run index -- --pdf "C:\\path\\to\\document.pdf"
  npm run summarize -- --pdf "C:\\path\\to\\document.pdf"
  npm run ask -- --pdf "C:\\path\\to\\document.pdf" --question "What methodology was used?"

  npm start -- summarize --pdf "C:\\path\\to\\document.pdf"
  npm start -- ask --pdf "C:\\path\\to\\document.pdf" --question "Summarize the document"
`
    );

  if (process.argv.length <= 2) {
    program.help();
  }

  registerSharedOptions(
    program
      .command("index")
      .description("Extract, chunk, embed, and store a PDF in the local vector store.")
      .action(async (options) => {
        const agent = new PdfRagAgent();
        const result = await agent.indexPdf(options.pdf, { rebuild: options.rebuild });
        if (result.reusedExisting) {
          console.log(`Indexed collection reused: ${result.collectionName}`);
          console.log(`Stored chunks: ${result.chunkCount}`);
        } else {
          console.log(`Indexed PDF: ${result.pdfPath}`);
          console.log(`Pages extracted: ${result.pageCount}`);
          console.log(`Chunks stored: ${result.chunkCount}`);
        }
        console.log(`Collection: ${result.collectionName}`);
        console.log(`Vector store: ${result.vectorStoreBackend}`);
      })
  );

  registerSharedOptions(
    program
      .command("summarize")
      .description("Generate a grounded summary of the PDF.")
      .action(async (options) => {
        const agent = new PdfRagAgent();
        const result = await agent.summarizePdf(options.pdf, { rebuild: options.rebuild });
        console.log("Summary:");
        console.log(result.summary);
        console.log(`\nCollection: ${result.collectionName}`);
        console.log(`Chunks used: ${result.chunkCount}`);
        console.log(`Vector store: ${result.vectorStoreBackend}`);
      })
  );

  registerSharedOptions(
    program
      .command("ask")
      .description("Ask a grounded question about the PDF.")
      .requiredOption("--question <text>", "Question to ask about the PDF")
      .action(async (options) => {
        const agent = new PdfRagAgent();
        const result = await agent.answerQuestion(options.pdf, options.question, { rebuild: options.rebuild });
        console.log("Answer:");
        console.log(result.answer);
        console.log("\nRetrieved context:");
        for (const chunk of result.sourceChunks) {
          const chunkId = chunk.metadata?.chunkId || "?";
          const distance = typeof chunk.distance === "number" ? ` | distance ${chunk.distance.toFixed(4)}` : "";
          console.log(`- Chunk ${chunkId}${distance}: ${formatSnippet(chunk.text)}`);
        }
        console.log(`\nCollection: ${result.collectionName}`);
        console.log(`Vector store: ${result.vectorStoreBackend}`);
      })
  );

  try {
    await program.parseAsync(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : JSON.stringify(error, null, 2);
    console.error(message);
    process.exitCode = 1;
  }
}

await main();
