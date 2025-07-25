#!/usr/bin/env python3
"""
Round‑1B scorer (semantic + lexical + cross‑encoder rerank)

• Base encoder : gtr‑t5‑base  (≈540 MB, CPU)
• Reranker     : cross‑encoder/ms-marco-MiniLM-L-6-v2 (≈65 MB)
• BM25 fallback for lexical recall
• Adaptive cut‑off + top‑N cap
• Pure‑Python / CPU, no internet during run
"""

import argparse, json, time, pathlib, os, re, datetime, multiprocessing as mp
from collections import defaultdict, Counter

import numpy as np
import nltk, fitz, networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─────────────── initialisation ────────────────
nltk.download("punkt", quiet=True)

GENERIC_HEADINGS = {
    "introduction","intro","overview","background",
    "conclusion","summary","agenda","contents","table of contents",
    "toc","references","appendix","faq","index","preface"
}

# -------- helpers --------
def normal(s:str)->str: return re.sub(r"\s+"," ",s.lower().strip())

def lead(text:str,max_words:int=130)->str:
    sents=[s.strip() for s in nltk.sent_tokenize(text)]
    out,used=[],0
    for s in sents:
        w=len(s.split())
        if used+w>max_words: break
        out.append(s); used+=w
    return " ".join(out) if out else text[:max_words]

# -------- PDF slicing (outline‑aware, very light) --------
def chunk_by_outline(pages, outline):
    if not outline:
        for i,txt in enumerate(pages):
            if len(txt)>120:
                yield {"title":f"page_{i+1}","page":i+1,"raw":txt[:2000],"level":"H6"}
        return
    by_page=defaultdict(list)
    for h in outline: by_page[h["page"]].append(h)
    sorted_pages=sorted(by_page)
    for k,pg in enumerate(sorted_pages):
        start=pg-1
        end  =sorted_pages[k+1]-2 if k+1<len(sorted_pages) else len(pages)-1
        span =" ".join(pages[start:end+1])[:2000]
        hdr  =by_page[pg][0]
        yield {"title":hdr["text"].strip(),
               "page":pg,
               "raw":span,
               "level":hdr.get("level","H6")}

def slice_pdf(args):
    pdf_path, outline_map = args
    pages   = [re.sub(r"\s+"," ",(p.get_text("text") or "")).strip()
               for p in fitz.open(pdf_path)]
    outline = outline_map.get(pdf_path.name,[])
    out=[]
    for c in chunk_by_outline(pages, outline):
        if len(c["title"].split())<=2 and normal(c["title"]) in GENERIC_HEADINGS:
            continue
        c["doc"]=os.path.basename(pdf_path)
        out.append(c)
    return out

# -------- adaptive elbow --------
def elbow(scores,drop=0.12,floor=0.45,min_keep=6,max_keep=30):
    keep=len(scores)
    for i,(a,b) in enumerate(zip(scores[:-1],scores[1:]),1):
        if a-b>drop or b<floor:
            keep=i+1; break
    return max(min_keep,min(keep,max_keep))

# -------- main pipeline --------
def run_pipeline(cfg, pdf_dir: pathlib.Path, model_path:str,
                 cross_name:str="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_out:int=5):

    tic=time.time()

    persona = cfg["persona"]["role"]
    job_blk = (cfg.get("job_to_be_done") or cfg.get("job_to_be") or {})
    task    = job_blk.get("task","").strip()
    if not task:
        raise ValueError("JSON missing task field")
    intent  = f"{persona} needs to {task}"

    # --- models
    encoder = SentenceTransformer(model_path, device="cpu")
    reranker = CrossEncoder(cross_name, device="cpu")   # ±65 MB

    intent_vec = encoder.encode(intent, normalize_embeddings=True)

    # --- outlines from Round‑1A
    from process_pdfs import extract as extract_pdf
    outline_map = {d["filename"]:
                   extract_pdf(str(pdf_dir/d["filename"]))["outline"]
                   for d in cfg["documents"]}

    # --- slice PDFs (multiprocessing)
    with mp.Pool(min(4, mp.cpu_count())) as pool:
        chunks=sum(pool.map(
            slice_pdf,
            [(pdf_dir/d["filename"], outline_map) for d in cfg["documents"]]
        ),[])

    if not chunks: raise RuntimeError("no chunks extracted")

    # --- BM25 recall
    tokenized=[re.findall(r"\w+",c["raw"].lower()) for c in chunks]
    bm25=BM25Okapi(tokenized)
    bm_scores=bm25.get_scores(re.findall(r"\w+", intent.lower()))
    recall_idx=np.argsort(-bm_scores)[:300]

    # --- SBERT embeddings for recall set
    previews=[
        (c["title"]+" – "+" ".join(c["raw"].split()[:80])
        if len(c["title"].split())<=3 else c["title"])
        for c in (chunks[i] for i in recall_idx)
    ]
    emb = encoder.encode(previews, batch_size=48, normalize_embeddings=True, show_progress_bar=False)
    dense = np.dot(emb, intent_vec)

    bm_norm=bm_scores[recall_idx]/(bm_scores[recall_idx].max()+1e-9)
    hybrid = 0.9*dense + 0.1*bm_norm
    order  = np.argsort(-hybrid)

    keep_n = elbow(hybrid[order].tolist())
    keep_idxs = order[:keep_n]

    # --- Cross‑encoder rerank on kept candidates
    pairs=[(intent, previews[i]) for i in keep_idxs]
    ce_scores = np.array(reranker.predict(pairs, batch_size=32))

    final_scores = 0.6*ce_scores + 0.4*hybrid[keep_idxs]
    finals = sorted(zip(final_scores, keep_idxs), key=lambda x:-x[0])

    # --- build JSON
    out_sections=[]
    out_analysis=[]
    for rank,(sc,i) in enumerate(finals[:top_out],1):
        c=chunks[ recall_idx[ i ] ]
        if len(c["title"].split())<2: continue   # safety
        out_sections.append(dict(
            document=c["doc"],
            section_title=c["title"],
            importance_rank=rank,
            page_number=c["page"]
        ))
        out_analysis.append(dict(
            document=c["doc"],
            refined_text=lead(c["raw"]),
            page_number=c["page"]
        ))

    result=dict(
        metadata=dict(
            input_documents=[d["filename"] for d in cfg["documents"]],
            persona=persona,
            job_to_be_done=task,
            processing_timestamp=datetime.datetime.utcnow().isoformat()
        ),
        extracted_sections=out_sections,
        subsection_analysis=out_analysis
    )

    print(f"[INFO] done in {time.time()-tic:.1f}s")
    return result

# -------- CLI --------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--challenge_json", required=True, help="File location containing challenge 1B input json")
    ap.add_argument("--pdf_dir",      required=True, help="directory containing PDFs")
    ap.add_argument("--model_path",   default="./models/sentence-transformers/gtr-t5-base", help="model path")
    ap.add_argument("--cross_model",  default="./models/cross-encoder/ms-marco-MiniLM-L6-v2", help="cross model path")
    ap.add_argument("--top_out",      type=int, default=15, help="max sections in output")
    ap.add_argument("--output",       required=True,default="challenge1b_output.json", help="location to store output json")
    args=ap.parse_args()

    cfg=json.loads(pathlib.Path(args.challenge_json).read_text(encoding="utf-8"))
    res=run_pipeline(cfg, pathlib.Path(args.pdf_dir),
                     args.model_path,
                     cross_name=args.cross_model,
                     top_out=args.top_out)

    pathlib.Path(args.output).write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"✓ wrote {args.output}")

if __name__=="__main__":
    main()
