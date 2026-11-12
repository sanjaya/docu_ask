# docu_ask

Ask questions about your PDF and Word (.docx) files from the terminal. Uses **LangChain**, **Google Gemini** and **Chroma DB**.

## How it works

```
PDF / Word ──► 1. Load text ──► 2. Split into chunks ──► 3. Embed with Gemini & store in Chroma
                                                                      │
You ask a question ──► 4. Agent searches Chroma ──► Gemini writes the answer (with sources)
```

## Setup (one time)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the libraries
pip install -r requirements.txt

# 3. Add your Google API key (free at https://aistudio.google.com/apikey)
cp .env.example .env
# then open .env and paste your key
```

## Usage

```bash
# One PDF or Word file
python docu_ask.py --file reports/annual_report.pdf
python docu_ask.py --file reports/handbook.docx

# Every PDF and Word file in a folder (sub-folders included)
python docu_ask.py --folder reports/
```

Example session:

```
Step 1/3: Loading 1 file(s)
  Reading policy.pdf (.pdf) ...
Step 2/3: Splitting 3 page(s)/section(s) into chunks
Step 3/3: Saving 7 chunk(s) to the Chroma database (this can take a moment)

Ready! Ask a question about your documents.
You: How many days do I have to return an item?
Agent: You can return items within 30 days. (policy.pdf, page 1)
You: exit
```

## Good to know

- Each file's type is detected from its extension and sent to the matching reader (`LOADERS` in `docu_ask.py`). PDFs are cited by page; Word files are cited by heading section, since Word has no fixed pages.
- Old Word files (`.doc`) aren't supported: open them in Word and save as `.docx`.
- The database is saved in the `chroma_db/` folder and is rebuilt every time you run the program.
- Scanned PDFs (images of text) have no text to read, so they need OCR first.
- To change chunk size, number of results, or models, edit the **Settings** section at the top of `docu_ask.py`, or set `GEMINI_CHAT_MODEL` / `GEMINI_EMBEDDING_MODEL` in `.env`.
