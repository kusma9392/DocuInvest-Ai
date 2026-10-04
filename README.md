# Intelligent Document Investigator (ALG-AI-02)

Upload PDFs, images or text files and ask questions in natural language. The app answers from your documents, shows the source file and page, detects conflicting information across documents, and reports low confidence when the answer is not found.

## How it works
Upload -> Text extraction (PyMuPDF, OCR for scans) -> Passage index (TF-IDF) -> Retrieval -> Answer + Sources + Conflict check + Confidence

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Key decisions
- TF-IDF retrieval: fast, lightweight, no API key needed.
- Rule-based conflict detection: compares the same field (amount, due date) across different files.
- Missing-term check: if a key word from the question is not in the best passage, confidence drops to LOW instead of guessing.

## Testing
Tested with two invoices with different amounts and due dates (conflict detected) and a question not covered in the documents (low confidence).

## Known limitations
No LLM-written answers, keyword-based retrieval, English only, simple "Field: value" conflict detection.

## Future improvements
Embeddings + vector DB, LLM answer generation, table extraction, multilingual support.

## AI disclosure
AI assistants (Claude) helped write and debug the code. No external AI API is used at runtime. Libraries: Streamlit, PyMuPDF, Tesseract, scikit-learn.