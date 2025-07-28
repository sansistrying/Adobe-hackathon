# PDF Heading Extractor

## 🧠 Overview

This tool extracts document structure (title and hierarchical headings like H1, H2, H3) from PDF files using a rule-based visual and textual analysis of font size, position, and formatting metadata. It outputs the extracted structure in JSON format.

## ⚙️ Approach

- **Text Block Extraction**: Uses `PyMuPDF` to extract spans, fonts, sizes, and bounding boxes per line.
- **Heuristic Heading Scoring**: Calculates a weighted score based on features like boldness, caps, center alignment, keywords (e.g., "Chapter", "References"), etc.
- **Title Detection**: Top-scoring heading near the top of the document.
- **Outline Structuring**: Assigns H1–H4 based on relative font size rankings.

## 📚 Libraries Used

- `PyMuPDF` (fitz) – PDF text and layout extraction
- `numpy` – For averaging and size analysis
- `re`, `json`, `os` – Standard Python modules

## 🐳 Docker Instructions

### Build Docker Image

```bash
docker build --platform linux/amd64 -t pdfheadingextractor:latest .
