# Round‑1B PDF Scorer  
Semantic + Lexical + Cross‑Encoder Rerank
========================================

This repository contains a **single‑file pipeline (`main.py`)** that selects the
most relevant PDF sections for a given *persona + task* (Challenge 1 B).

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Base encoder | `gtr‑t5‑base` | ≈ 540 MB | dense semantic recall |
| Reranker | `cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2` | ≈ 65 MB | pair‑wise relevance rerank |
| Lexical | BM25 (pure Python) | – | key‑term recall + tie‑breaker |

The code runs **entirely on CPU, without Internet access**, once the models are
embedded in the Docker image.

---

## 1  🏗 Build the Docker image (first time only)

> **Heads‑up:** the first `docker build` **is slow (~8 min)** because it must:
>
> * download ≈ 300 MB of Python wheels (`torch`, `transformers`, …)  
> * download and unpack ≈ 600 MB of HF model tarballs
>

```bash
# At repo root
docker build -t round1b .
````

*Inside the Dockerfile we fetch the 2 model tarballs from the project’s
GitHub → Releases page; they do **not** live in git, so the repo stays small.*

---

### 2  Run the scorer
##### Note: Please use /data with output else your output will be lost

> **Mounting your files**
> `-v <host‑path>:/data` tells Docker to **treat whatever folder you point at as `/data` inside the container**.
> Just replace `<host‑path>` with the folder that already contains
> `challenge1b_input.json` and the `PDFs/` sub‑directory.

#### 2.1 Linux / macOS example

```bash
# Inside the repo root – adjust if your files live elsewhere
docker run --rm \
  -v "$PWD/Challenge_1b/Collection_2":/data \   # <‑‑ your host folder
  round1b \
  --challenge_json /data/challenge1b_input.json \
  --pdf_dir        /data/PDFs \
  --output         /data/challenge1b_output.json
```

#### 2.2 Windows (PowerShell or CMD) example

```powershell
REM Replace the path before the colon with YOUR folder:
docker run --rm ^
  -v "C:\Users\you\Challenge_1b\Collection_2":/data ^
  round1b ^
  --challenge_json /data/challenge1b_input.json ^
  --pdf_dir        /data/PDFs ^
  --output         /data/challenge1b_output.json
```

If everything is mounted correctly the container will report:

```
[INFO] done in 38.7s –  kept 5 sections
✓ wrote /data/challenge1b_output.json
```

— and `challenge1b_output.json` will appear inside the same host folder you
mounted.

## 3  CLI options (inside the container)

```
--challenge_json   required   path to the input JSON
--pdf_dir          required   directory containing the PDFs
--output           default=challenge1b_output.json
--model_path       default=/app/models/sentence-transformers/gtr-t5-base
--cross_model      default=/app/models/cross-encoder/ms-marco-MiniLM-L-6-v2
--top_out          default=15  hard cap on #sections in output
```

You normally **don’t touch `--model_path` / `--cross_model`** because the models
are baked into the image.

---

## 4  🚀 How the pipeline works (plain English)

1. **Turn the prompt into a query**
   *“HR professional needs to create and manage fillable forms …”* becomes a
   single sentence.

2. **Slice every PDF into candidate sections**
   Using the outline headings we grab each heading + the next 1‑2 pages of text.
   Generic headings like *“Introduction”* are discarded.

3. **Recall phase**

   * 300 candidates are fetched by **BM25** (keyword overlap).
   * Each gets a 768‑d embedding from **gtr‑t5‑base** and we compute cosine
     similarity → hybrid score = `0.9*dense + 0.1*BM25`.

4. **Adaptive elbow cut**
   We keep candidates until the score curve drops sharply or falls below
   0.45 (similarity). Usually 15‑30 remain.

5. **Rerank phase**
   Each remaining `(query, section‑preview)` pair is passed through the
   **cross‑encoder**, which reads both texts and emits a fine‑grained relevance
   score.
   Final score = `0.6*cross + 0.4*hybrid`.

6. **Output**
   The top‑`N` (default 5) sections are dumped to JSON:

   * `extracted_sections` → doc name, heading, rank, page
   * `subsection_analysis` → 1‑paragraph “lead‑summary” for quick preview.

Total runtime on 6 × 200‑page PDFs: **35‑50 s** on an 8‑vCPU laptop, CPU only.
