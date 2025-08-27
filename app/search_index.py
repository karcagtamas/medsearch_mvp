# app/search_index.py
"""
Search FAISS indices with text or image queries
"""

import argparse, json
from pathlib import Path
import faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def l2_normalize(vectors):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norm

def load_meta(path: Path):
    return [json.loads(l) for l in open(path,"r",encoding="utf-8")]

def search(index_dir, query_text=None, query_image=None, top_k=5, device="cpu",
           text_model="sentence-transformers/all-MiniLM-L6-v2",
           image_model="openai/clip-vit-base-patch32"):
    meta=load_meta(Path(index_dir)/"meta.jsonl")
    results=[]
    if query_text:
        model=SentenceTransformer(text_model, device=device)
        vec=model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        idx=faiss.read_index(str(Path(index_dir)/"faiss_text.index"))
        scores,ids=idx.search(vec,top_k)
        for s,i in zip(scores[0],ids[0]):
            if i>=0: results.append({"score":float(s),"entry":meta[i]})
    elif query_image:
        model=CLIPModel.from_pretrained(image_model).to(device)
        proc=CLIPProcessor.from_pretrained(image_model)
        img=Image.open(query_image).convert("RGB")
        inputs={k:v.to(device) for k,v in proc(images=img, return_tensors="pt").items()}
        with torch.no_grad(): out=model.get_image_features(**inputs)
        vec=l2_normalize(out.cpu().numpy().astype("float32"))
        idx=faiss.read_index(str(Path(index_dir)/"faiss_image.index"))
        scores,ids=idx.search(vec,top_k)
        for s,i in zip(scores[0],ids[0]):
            if i>=0:
                for m in meta:
                    if m.get("image_index_id")==i: results.append({"score":float(s),"entry":m}); break
    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index_dir", required=True)
    p.add_argument("--query_text")
    p.add_argument("--query_image")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    args=p.parse_args()
    res=search(args.index_dir, args.query_text, args.query_image, args.top_k, args.device)
    for r in res:
        print(f"[{r['score']:.4f}] {r['entry']['file_path']}")
        if r['entry'].get("text_preview"): print("  Preview:", r['entry']['text_preview'][:150], "...")
