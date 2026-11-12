"""
docu_ask.py - Ask questions about your PDF and Word documents.

How it works (in 4 simple steps):
    1. LOAD   - Read the text out of your file(s). The program looks at each
                file's extension (.pdf, .docx) and picks the right reader.
    2. SPLIT  - Cut the text into small overlapping "chunks".
    3. STORE  - Turn each chunk into numbers ("embeddings") with Google Gemini
                and save them in a Chroma vector database.
    4. ASK    - An AI agent (Gemini + LangChain) searches the database for the
                chunks that match your question and writes an answer from them.

Usage:
    python docu_ask.py --file  path/to/document.pdf
    python docu_ask.py --file  path/to/report.docx
    python docu_ask.py --folder path/to/folder_with_documents
"""

import argparse
import os
import sys
from pathlib import Path

from docx import Document as WordDocument  # renamed so it doesn't clash with LangChain's Document
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Settings - change these if you like
# ---------------------------------------------------------------------------
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHROMA_FOLDER = "chroma_db"   # where the database is saved on disk
CHUNK_SIZE = 1000             # characters per chunk
CHUNK_OVERLAP = 200           # characters shared between neighbouring chunks
TOP_K = 4                     # how many chunks to fetch for each search


# ---------------------------------------------------------------------------
# Step 1a: One reader ("loader") per file format
#
# Every loader takes a file path and returns a list of LangChain Documents.
# A Document = some text + "metadata" (extra info we want to remember, like
# which file and which page/section the text came from).
# ---------------------------------------------------------------------------
def load_pdf(path):
    """PDF: one Document per page."""
    documents = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": path.name, "location": f"page {page_number}"},
            )
        )
    return documents


def load_docx(path):
    """Word: Word files have no fixed pages, so we make one Document per heading section."""
    word_file = WordDocument(path)
    documents = []

    def save_section(location, lines):
        text = "\n".join(lines).strip()
        if text:  # skip empty sections
            documents.append(
                Document(page_content=text, metadata={"source": path.name, "location": location})
            )

    # Walk through the paragraphs. Every heading starts a new section.
    current_location = "beginning of document"
    current_lines = []
    for paragraph in word_file.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading"):
            save_section(current_location, current_lines)   # finish the previous section
            current_location = f"section: {text}"
            current_lines = []
        current_lines.append(text)
    save_section(current_location, current_lines)            # finish the last section

    # Tables are stored separately from paragraphs in Word, so read them too.
    table_lines = []
    for table in word_file.tables:
        for row in table.rows:
            table_lines.append(" | ".join(cell.text.strip() for cell in row.cells))
        table_lines.append("")  # blank line between tables
    save_section("tables", table_lines)

    return documents


# ---------------------------------------------------------------------------
# Step 1b: The "router" - which loader handles which file extension.
# To support a new format, write a load_xxx function and add one line here.
# ---------------------------------------------------------------------------
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
}


# ---------------------------------------------------------------------------
# Step 0: Read the command line (--file or --folder)
# ---------------------------------------------------------------------------
def read_command_line():
    parser = argparse.ArgumentParser(
        description="Ask questions about PDF and Word (.docx) documents using Gemini, LangChain and Chroma."
    )
    # The user must give exactly ONE of --file or --folder.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF or Word (.docx) file")
    group.add_argument("--folder", help="Path to a folder that contains PDF and/or Word files")
    return parser.parse_args()


def find_files(args):
    """Return a list of supported file paths based on --file or --folder."""
    supported = ", ".join(LOADERS)  # e.g. ".pdf, .docx"

    if args.file:
        path = Path(args.file)
        if not path.is_file():
            sys.exit(f"Error: file not found: {path}")
        extension = path.suffix.lower()
        if extension == ".doc":
            sys.exit("Error: old Word files (.doc) are not supported. Open it in Word and save it as .docx.")
        if extension not in LOADERS:
            sys.exit(f"Error: unsupported file type '{extension}'. Supported types: {supported}")
        return [path]

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Error: folder not found: {folder}")

    # rglob("*") looks inside sub-folders too.
    files = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in LOADERS
        and not p.name.startswith("~$")  # skip temporary "lock" files Word creates
    )
    if not files:
        sys.exit(f"Error: no supported files ({supported}) found in {folder}")
    return files


# ---------------------------------------------------------------------------
# Step 1: LOAD the files (each one goes to the right loader)
# ---------------------------------------------------------------------------
def load_documents(files):
    all_documents = []
    for path in files:
        extension = path.suffix.lower()
        loader = LOADERS[extension]  # pick the right reader for this file type
        print(f"  Reading {path.name} ({extension}) ...")
        try:
            all_documents.extend(loader(path))
        except Exception as error:
            print(f"  Skipping {path.name}: {error}")

    if not all_documents:
        sys.exit("Error: could not read any text from the file(s).")
    return all_documents


# ---------------------------------------------------------------------------
# Step 2: SPLIT the documents into chunks
# ---------------------------------------------------------------------------
def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    # Drop empty chunks (e.g. from scanned PDF pages with no text layer).
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        sys.exit("Error: the file(s) contain no readable text (PDFs may be scanned images).")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: STORE the chunks in Chroma
# ---------------------------------------------------------------------------
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_FOLDER,
    )
    # Start fresh each run, so old documents don't mix with the new ones.
    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    return vector_store


# ---------------------------------------------------------------------------
# Step 4: Create the AI agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's documents.

Rules:
- ALWAYS use the search_documents tool to look up information before answering.
- Answer ONLY from the text the tool returns. Do not make things up.
- If the documents do not contain the answer, say "I couldn't find that in the documents."
- At the end of your answer, list the sources you used as (file name, page or section).
"""


def create_document_agent(vector_store):
    # A "tool" is a normal Python function the agent is allowed to call.
    # The docstring tells the AI what the tool does, so keep it clear.
    @tool
    def search_documents(query: str) -> str:
        """Search the uploaded documents and return the most relevant passages."""
        results = vector_store.similarity_search(query, k=TOP_K)
        if not results:
            return "No matching passages found."

        passages = []
        for doc in results:
            source = doc.metadata["source"]
            location = doc.metadata["location"]
            passages.append(f"[Source: {source}, {location}]\n{doc.page_content}")
        return "\n\n---\n\n".join(passages)

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    return create_agent(model=model, tools=[search_documents], system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# The question-and-answer loop
# ---------------------------------------------------------------------------
def chat_loop(agent):
    print("\nReady! Ask a question about your documents.")
    print("Type 'exit' or 'quit' to stop.\n")

    history = []  # keeps the conversation so follow-up questions work
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": question})
        try:
            result = agent.invoke({"messages": history})
        except Exception as error:
            print(f"\nSomething went wrong: {error}\n")
            history.pop()  # forget the question that failed
            continue

        history = result["messages"]   # the full conversation, including the reply
        answer = history[-1].text      # the agent's final answer
        print(f"\nAgent: {answer}\n")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    args = read_command_line()

    load_dotenv()  # reads GOOGLE_API_KEY from the .env file
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("Error: GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")

    files = find_files(args)

    print(f"Step 1/3: Loading {len(files)} file(s)")
    documents = load_documents(files)

    print(f"Step 2/3: Splitting {len(documents)} page(s)/section(s) into chunks")
    chunks = split_into_chunks(documents)

    print(f"Step 3/3: Saving {len(chunks)} chunk(s) to the Chroma database (this can take a moment)")
    vector_store = build_vector_store(chunks)

    agent = create_document_agent(vector_store)
    chat_loop(agent)


if __name__ == "__main__":
    main()
