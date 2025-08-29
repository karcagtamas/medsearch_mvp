# app/build_index.py
"""
Build FAISS indices for medical records:
- Extracts text (OCR for images/PDFs)
- Creates embeddings (text + image)
- Saves FAISS indices + metadata
"""

import os, argparse, json
from pathlib import Path
import pdfplumber, pytesseract
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
import faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForTokenClassification, pipeline

def extract_pairs(text: str):
    # model_name = "d4data/biomedical-ner-all"
    # model_name = "samrawal/bert-base-uncased_clinical-ner"
    # model_name = "dmis-lab/biobert-base-cased-v1.1"
    # model_name = "lcampillos/roberta-es-clinical-trials-ner"
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    results = ner_pipeline(text)
    pairs = []
    current_name, current_value = None, None

    print(results)

    for ent in results:
        label = ent["entity_group"]
        word = ent["word"]

        if label in ["Diagnostic_procedure"]:
            current_name = word
        elif label in ["Lab_value"]:
            current_value = word
            if current_name:
                pairs.append({
                    "name": current_name,
                    "value": f"{current_value} {word}"
                })
                current_name, current_value = None, None

    return pairs


def extract_text(file_path: Path) -> str:
    # Handle PDF files
    if file_path.suffix.lower() == ".pdf":
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img)
        return text

    # Handle DOCX files
    elif file_path.suffix.lower() == ".docx":
        doc = Document(file_path.__str__())
        return "\n".join([p.text for p in doc.paragraphs])

    # Handle images
    elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        return pytesseract.image_to_string(Image.open(file_path))

    # Handle texts
    elif file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    return ""


def build_index(input_dir, output_dir, text_model_name, image_model_name, device="cpu"):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    text_model = SentenceTransformer(text_model_name, device=device)
    clip_model = CLIPModel.from_pretrained(image_model_name).to(device)
    clip_proc = CLIPProcessor.from_pretrained(image_model_name)

    text_embs, text_meta = [], []
    image_embs, image_meta = [], []

    for f in input_dir.glob("**/*"):
        if not f.is_file(): continue
        entry = {"file_path": str(f), "has_text": False, "has_image": False}

        text = extract_text(f)

        if text.strip():
            emb = text_model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            text_embs.append(emb[0])
            entry["has_text"] = True
            entry["text_preview"] = text[:300]
            entry["text_index_id"] = len(text_embs) - 1
            pairs = extract_pairs(text)
            print(pairs)

        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img = Image.open(f).convert("RGB")
            inputs = clip_proc(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(): out = clip_model.get_image_features(**inputs)
            emb = out.cpu().numpy().astype("float32")
            image_embs.append(emb[0]);
            entry["has_image"] = True;
            entry["image_index_id"] = len(image_embs) - 1

        if entry["has_text"] or entry["has_image"]:
            text_meta.append(entry)

    if text_embs:
        arr = np.vstack(text_embs)
        idx = faiss.IndexFlatL2(arr.shape[1])
        idx.add(arr)
        faiss.write_index(idx, str(output_dir / "faiss_text.index"))
    if image_embs:
        arr = np.vstack(image_embs)
        idx = faiss.IndexFlatL2(arr.shape[1])
        idx.add(arr)
        faiss.write_index(idx, str(output_dir / "faiss_image.index"))

    with open(output_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in text_meta: f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--image_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    build_index(args.input_dir, args.output_dir, args.text_model, args.image_model, args.device)
