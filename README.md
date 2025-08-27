# Documentation

## Research

### Main concept

1. The system is reading in various files like PDFs, DOCXs, Images or TXTs. The input file type is detected and read accordingly to it.
2. OCR / Text extraction
    - PDFs -> `pdfplumber` if text is selectable, otherwise `pdf2image + pytesseract`
    - DOCXs -> `python-docx`
    - Images -> `pytesseract`
3. Embeddings
    - Text embeddings: `Sentence Transformer`
    - Image embeddings: `CLIP`
4. Vector index
    - One for the texts (text embeddings) and one for the images (image embeddings)
5. Metadata storage

#### OCR

OCR stands for Optical Character Recognition.

It’s a technology that enables computers to recognize and extract text from images or scanned documents. In simple terms, OCR converts printed or handwritten text on paper (or in an image/PDF) into machine-readable text that you can edit, search, or process.

##### Steps

1. Image preprocessing - Cleaning up the scanned image (removing noise, etc.)
2. Text detection
3. Character recognition
4. Post-processing - Correcting errors

#### Embeddings

Reporesentation of a complex data (like text or images) as a vector of numbers

- Each vector might have 128, 512 or 768+ dimensions
- The number capture semantic meaning, not just raw data

##### Importance

1. Semantic search
    - Two different words can be close to each other in embedding space
2. Cross-modal search
3. Efficient retrieval
    - Once all documents/images are converted into embeddings, we use a vector database (FAISS) to find the nearest neighbors

### Problem

General embeddings can match common semantic relations, but may miss special relations.

In progres...