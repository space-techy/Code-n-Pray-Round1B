#!/usr/bin/env python3
# adaptive_heading_extractor_full_v30_introfix.py
#
# v30 vs v29:
# - Prevents "Introduction/Overview/Background/..." from merging into the title:
#   (a) trims such tokens out of the chosen title block
#   (b) blocks merge_wrapped_headings() from joining them to the previous line
# - Everything else unchanged to preserve good behavior on other PDFs.

import fitz
import json, math, re
import numpy as np
from collections import Counter, defaultdict

# ─────────────────────────────── regexes ─────────────────────────────── #
NUM_PATTERN         = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")
NUMBERING_RE        = re.compile(r"^\s*\d+(?:\.\d+)*\b")
PROMO_NUMBERING_RE  = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")
ENDS_COLON_RE       = re.compile(r".*:\s*$")
ROMAN_PHASE_RE      = re.compile(r"^\s*Phase\s+[IVXLCM]+\b", re.I)
PUNCT_ONLY_RE       = re.compile(r"^[\W_]+$")

DATE_RE             = re.compile(
    r"^\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)(?:\s+\d{1,2})?(?:,?\s+\d{2,4})?\s*$",
    re.IGNORECASE
)
DIGIT_DATE_RE       = re.compile(r"^\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s*$")

APPENDIX_RE         = re.compile(r"^\s*Appendix\s+([A-Z])\b", re.I)
FOR_EACH_RE         = re.compile(r"^\s*(For\s+(each|the)\s+[A-Z][^\:]*\s+it\s+could\s+mean:)\s*$", re.I)
TOC_ENTRY_RE        = re.compile(r"\.{2,}\s*\d+\s*$")     # dot leaders .. 12

COVER_NOISE_RE      = re.compile(
    r"^\s*(version\s+\d|draft|confidential|copyright|©|\(c\)|isbn|issn|"
    r"international\b.*(board|association|committee)|prepared\s+by|published\s+by)\b",
    re.I
)
NARRATIVE_VERBS_RE  = re.compile(r"\b(include|should|must|expected|approved|completed|distribution)\b", re.I)

SPLIT_MULTI_NUM_RE  = re.compile(r"(?<!^)\s+(?=\d+(?:\.\d+)*[.)]?\s)")
NUM_START_RE        = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s")

# list/bullet detection
BULLET_LEADER_RE    = re.compile(r"^\s*([•\u2022\u25E6\u25AA\u25CF]|[-–—])\s+")
ENUM_LIST_RE        = re.compile(r"^\s*(?:\(?[ivx]+\)|\(?[a-zA-Z]\)|\(?\d+\)|\d+[.)])\s+", re.I)
ENUMERATOR_RE       = re.compile(r"^\s*(?:\(?\d+(?:\.\d+)*|\(?[ivxlcdm]+|\(?[A-Za-z])[\.)]?\s*$", re.I)
NUM_LIST_RE         = re.compile(r"^\s*(?:\d+\s+){2,}\d+\s*$")

# Faux bullets 'o'
FAUX_BULLET_O_START_RE  = re.compile(r"^\s*[oO](?=\d|[A-Z])")
FAUX_BULLET_O_INLINE_RE = re.compile(r"(?:^|[\s,;:()\[\]—\-])o(?=\d|[A-Z])")

# Recipe subheads we keep even if colon-ended
RECIPE_SUBHEADS = {"ingredients:", "instructions:", "directions:", "method:", "steps:", "notes:"}

# Units / ingredient / instruction signals (used only inside recipe mode)
INGR_UNIT_RE = re.compile(
    r"\b(\d+(/\d+)?|\d+\.\d+)\s*(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|"
    r"g|gram|grams|kg|ml|l|oz|ounce|ounces|lb|pound|pounds|clove|cloves|slice|slices|pinch|dash)\b",
    re.I
)
INGR_CORE_RE = re.compile(
    r"\b(salt|pepper|sugar|onion|onions|garlic|egg|eggs|tomato|tomatoes|potato|potatoes|"
    r"milk|butter|flour|oil|olive\s+oil|water|vinegar|cheese|cream|yogurt|cilantro|parsley|"
    r"cumin|turmeric|ginger|soy\s*sauce|mirin|sesame|dill|oregano|basil|paprika|rice|beans|lentils|"
    r"chickpea|chickpeas|spinach|cucumber|feta|gruy|gruy\u00E8re|gruyere)\b",
    re.I
)
INSTR_VERB_RE = re.compile(
    r"^\s*(preheat|mix|add|bake|cook|chop|slice|whisk|stir|pour|bring|simmer|boil|reduce|"
    r"serve|heat|layer|beat|steam|toss|drizzle|rub|rinse|drain|grind|mash)\b",
    re.I
)

STYLE_TOKENS = ("bold","black","medium","demi","semibold","heavy","extrabold","italic","oblique","roman","regular")

# Section tokens that must NOT be glued to titles or previous headings
TAIL_BAN = {
    "introduction", "overview", "background", "contents", "table of contents",
    "acknowledgements", "abstract", "foreword", "preface"
}

# ───────────────────────────── text helpers ───────────────────────────── #
def fix_encoding(s: str) -> str:
    s = re.sub(r"([A-Za-z])\uFFFD([A-Za-z])", r"\1'\2", s)
    s = s.replace("\u2019", "’")
    s = re.sub(r"\s{2,}", " ", s)
    return s

def squeeze_repeats_soft(s: str) -> str:
    protected = {}
    def protect(m):
        tok = m.group(0); key = f"__ROM__{len(protected)}__"
        protected[key] = tok; return key
    s2 = re.sub(r"\bI{3,}\b", protect, s)
    s2 = re.sub(r"\bX{3,}\b", protect, s2)
    s2 = re.sub(r"\bC{3,}\b", protect, s2)
    s2 = re.sub(r"\bM{3,}\b", protect, s2)
    s2 = re.sub(r"(.)\1{2,}", r"\1\1", s2)
    for key, tok in protected.items():
        s2 = s2.replace(key, tok)
    return s2

def starts_capital(t: str) -> bool:
    t = t.strip()
    return bool(t and t[0].isalpha() and t[0].upper() == t[0])

def word_count(t: str) -> int:
    return len(t.strip().split())

def uppercase_ratio(t: str) -> float:
    t = re.sub(r"[^A-Za-z]", "", t)
    if not t: return 0.0
    return sum(ch.isupper() for ch in t) / len(t)

def alpha_ratio(t: str) -> float:
    nonspace = re.sub(r"\s+", "", t)
    if not nonspace:
        return 0.0
    letters = re.sub(r"[^A-Za-z]", "", nonspace)
    return len(letters)/len(nonspace)

def numeric_depth(text: str):
    m = NUM_PATTERN.match(text)
    if not m:
        return None
    return 1 + m.group(1).count('.')

# ───────────────────────────── font helpers ───────────────────────────── #
def normalize_family(font_name: str) -> str:
    if not font_name:
        return "unknown"
    name = font_name
    if '+' in name:
        name = name.split('+', 1)[1]
    low = name.lower()
    for tok in STYLE_TOKENS:
        low = low.replace(tok, "")
    low = low.replace("-", "").replace("_", "")
    return low or "unknown"

def is_bold(font_name: str) -> int:
    fn = (font_name or "").lower()
    return int(any(k in fn for k in ("bold","black","demi","semibold","heavy","extrabold","medium")))

def is_italic(font_name: str) -> int:
    fn = (font_name or "").lower()
    return int(any(k in fn for k in ("italic","oblique")))

# ─────────────────────── merging & de-ghosting ───────────────────────── #
def merge_line_spans_overlap_trim(spans, page_w):
    kept_parts, families = [], []
    last_x1 = -1e9
    max_size = 0.0
    bold = 0; italic = 0
    x0 = float("inf"); x1 = -float("inf")
    y0 = float("inf"); y1 = -float("inf")

    for sp in spans:
        t = sp["text"]
        plain = t.replace(" ", "")
        width = max(1.0, sp["x1"] - sp["x0"])
        cw = width / max(1, len(plain))
        tol = max(0.5, 0.01 * page_w, 0.5 * cw)
        if kept_parts:
            overlap = max(0.0, last_x1 - sp["x0"])
            if overlap > tol:
                est_chars = int(round((overlap / width) * len(t)))
                est_chars = max(0, min(est_chars, len(t)-1))
                t = t[est_chars:]
        if t:
            kept_parts.append(t)
            last_x1 = max(last_x1, sp["x1"])
            max_size = max(max_size, sp["size"])
            bold   = max(bold, is_bold(sp.get("font","")) or sp.get("bold",0))
            italic = max(italic, is_italic(sp.get("font","")) or sp.get("italic",0))
            families.append(normalize_family(sp.get("font","")))
            x0 = min(x0, sp["x0"]); x1 = max(x1, sp["x1"])
            y0 = min(y0, sp["y0"]); y1 = max(y1, sp["y1"])

    if not kept_parts:
        return "", 0.0, 0, 0, "unknown", 0.0, 0.0, 0.0, 0.0, 0.0

    fam_counts = Counter(families)
    family, cnt = fam_counts.most_common(1)[0]
    fam_conf = cnt / max(1, len(families))

    text = "".join(kept_parts)
    return text, max_size, bold, italic, family, fam_conf, x0, x1, y0, y1

# ─────────────────────────────── parsing ─────────────────────────────── #
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    lines, size_freq, family_freq = [], Counter(), Counter()
    uid = 0

    for pg, page in enumerate(doc, 1):
        w, h = page.rect.width, page.rect.height
        spans = []
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    txt = sp["text"].replace("\n", " ")
                    if not txt.strip():
                        continue
                    spans.append({
                        "text": txt,
                        "size": round(sp["size"], 2),
                        "font": sp["font"],
                        "bold": is_bold(sp["font"]),
                        "italic": is_italic(sp["font"]),
                        "x0": sp["bbox"][0], "x1": sp["bbox"][2],
                        "y0": sp["bbox"][1], "y1": sp["bbox"][3],
                    })

        spans.sort(key=lambda s: (s["y0"], s["x0"]))
        groups = defaultdict(list)
        for sp in spans:
            groups[round(sp["y0"])].append(sp)

        for _, grp in groups.items():
            grp.sort(key=lambda s: s["x0"])
            text, size, bold, italic, family, fam_conf, x0, x1, y0, y1 = merge_line_spans_overlap_trim(grp, w)
            text = squeeze_repeats_soft(fix_encoding(text))
            if not text.strip() or PUNCT_ONLY_RE.match(text.strip()):
                continue
            line = {
                "id": uid, "text": text, "size": size,
                "bold": bold, "italic": italic, "family": family, "fam_conf": fam_conf,
                "font_orig": grp[0]["font"],
                "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                "page": pg, "w": w, "h": h
            }
            lines.append(line); uid += 1
            size_freq[size] += 1
            family_freq[family] += 1

    lines.sort(key=lambda l: (l["page"], l["y0"]))
    return lines, size_freq, family_freq, doc.page_count

# ───────────────────────────── statistics ────────────────────────────── #
def derive_paragraph_band(size_freq):
    sizes = sorted(size_freq)
    counts = [size_freq[s] for s in sizes]
    if not sizes:
        return set()
    cum = np.cumsum(counts) / sum(counts)
    idx60 = next((i for i,c in enumerate(cum) if c >= 0.60), len(sizes)-1)
    if len(sizes) > 1:
        diffs = np.diff(sizes)
        jump_idx = int(np.argmax(diffs))
        cut = min(idx60, jump_idx)
    else:
        cut = 0
    return set(sizes[:cut+1])

def size_step_tolerance(size_freq):
    sizes = sorted(size_freq)
    if len(sizes) < 2:
        return 0.5
    diffs = np.diff(sizes)
    med = float(np.median(diffs))
    return max(0.5, 0.5 * med)

def add_line_gaps(lines):
    prev = {}
    per_page_gaps = defaultdict(list)
    for l in lines:
        pg = l["page"]
        gap = (l["y0"] - prev[pg]["y1"]) if pg in prev else float("inf")
        l["gap_above"] = gap
        prev[pg] = l
        if gap != float("inf"):
            per_page_gaps[pg].append(gap)
    page_gap_q40 = {pg: (float(np.percentile(gs, 40)) if gs else 0.0) for pg, gs in per_page_gaps.items()}
    page_gap_q60 = {pg: (float(np.percentile(gs, 60)) if gs else 0.0) for pg, gs in per_page_gaps.items()}
    page_gap_q75 = {pg: (float(np.percentile(gs, 75)) if gs else 0.0) for pg, gs in per_page_gaps.items()}
    return page_gap_q40, page_gap_q60, page_gap_q75

def body_indent_stats(lines, body_band):
    xs = [l["x0"] for l in lines if l["size"] in body_band]
    if not xs:
        return {"q10": 0.0, "q25": 0.0, "q50": 0.0}
    xs = np.array(xs, dtype=float)
    return {"q10": float(np.percentile(xs, 10)),
            "q25": float(np.percentile(xs, 25)),
            "q50": float(np.percentile(xs, 50))}

def per_page_size_p80(lines):
    by_page = defaultdict(list)
    for l in lines:
        by_page[l["page"]].append(l["size"])
    return {pg: float(np.percentile(sizes, 80)) for pg, sizes in by_page.items()}

def family_stats(family_freq):
    total = sum(family_freq.values()) or 1
    shares = {fam: cnt/total for fam, cnt in family_freq.items()}
    if not shares:
        return {"shares": {}, "q25": 0.0, "major": None}
    q25 = float(np.percentile(list(shares.values()), 25))
    major = max(shares.items(), key=lambda kv: kv[1])[0]
    return {"shares": shares, "q25": q25, "major": major}

def body_shape_stats(lines, body_band):
    bodies = [l for l in lines if l["size"] in body_band]
    if not bodies:
        return {"wc_q35": 8, "wc_q50": 12, "char_q35": 40, "fill_q40": 0.55, "fill_q60": 0.70}
    wcs   = [len(l["text"].split()) for l in bodies]
    chars = [len(l["text"].strip()) for l in bodies]
    fills = [ (l["x1"]-l["x0"])/max(1.0, l["w"]) for l in bodies ]
    return {
        "wc_q35":  float(np.percentile(wcs, 35)) if wcs else 8.0,
        "wc_q50":  float(np.median(wcs)) if wcs else 12.0,
        "char_q35":float(np.percentile(chars, 35)) if chars else 40.0,
        "fill_q40":float(np.percentile(fills, 40)) if fills else 0.55,
        "fill_q60":float(np.percentile(fills, 60)) if fills else 0.70
    }

def center_deviation(l):
    return abs(((l["x0"]+l["x1"])/2) - l["w"]/2)/max(1.0, l["w"])

def center_stats_for_top(size_cands):
    if not size_cands:
        return {"c_q75": 0.5, "c_median": 0.5}
    sizes_desc = sorted({c["size"] for c in size_cands}, reverse=True)
    top = sizes_desc[0]
    topset = [c for c in size_cands if math.isclose(c["size"], top, abs_tol=0.5)] or size_cands
    cdevs = [center_deviation(c) for c in topset]
    return {"c_q75": float(np.percentile(cdevs, 75)) if cdevs else 0.5,
            "c_median": float(np.median(cdevs)) if cdevs else 0.5}

def doc_modes(size_freq, lines):
    sizes = sorted(size_freq)
    uniform = False
    if sizes:
        if len(sizes) == 1:
            uniform = True
        else:
            spread = max(sizes) - min(sizes)
            uniform = spread <= max(0.75, 0.40 * np.median(np.diff(sizes))) if len(sizes) > 1 else True
    raw_list_hits = sum(bool(BULLET_LEADER_RE.match(l["text"])) or bool(ENUM_LIST_RE.match(l["text"])) for l in lines)
    list_density = raw_list_hits / max(1, len(lines))
    return {"uniform_sizes": uniform, "list_density": list_density}

# ────────────────────────── heading shape check ───────────────────────── #
def is_heading_shape(line, shape):
    t = line["text"].strip()
    wc = len(t.split())
    fill = (line["x1"] - line["x0"]) / max(1.0, line["w"])
    short_ok = (wc <= max(6, shape["wc_q35"])) or (len(t) <= max(20, shape["char_q35"]))
    narrow_ok = fill <= shape["fill_q40"]
    return (short_ok or narrow_ok) and not t.endswith(".")

# ─────────────────────── Recipe-mode detection & dish heads ─────────────────────── #
def detect_recipe_mode(lines, body_band, shape):
    subheads = 0
    head_like = 0
    for l in lines:
        low = l["text"].strip().lower()
        if ENDS_COLON_RE.match(low) and low in RECIPE_SUBHEADS:
            subheads += 1
        if (not NUMBERING_RE.match(l["text"])) and (not ENDS_COLON_RE.match(l["text"])) and is_heading_shape(l, shape):
            if 1 <= word_count(l["text"]) <= 4 and uppercase_ratio(l["text"]) < 0.85:
                head_like += 1
    return (subheads >= 3) or (head_like >= 15)

def find_dish_heads(lines, body_band, shape):
    heads = []
    for l in lines:
        t = l["text"]
        if NUMBERING_RE.match(t) or ENDS_COLON_RE.match(t):
            continue
        if not is_heading_shape(l, shape):
            continue
        big_enough = l["size"] >= (max(body_band) if body_band else 0.0) - 0.25
        if (l["bold"] or big_enough) and (word_count(t) <= 5) and not t.strip().endswith("."):
            heads.append(l["id"])
    return set(heads)

# ───────────────────────── Stage 1: candidates ───────────────────────── #
def size_candidates(lines, body_band, size_tol, page_size_p80_map, fam_stats, shape, page_gap_q60, modes, body_indent):
    doc_body_top = max(body_band) if body_band else 0.0
    out = []
    shares = fam_stats["shares"]; q25_share = fam_stats["q25"]; major = fam_stats["major"]

    q25 = body_indent.get("q25", 0.0); q50 = body_indent.get("q50", 0.0)
    indent_tol = max(0.0, (q50 - q25) * 0.5)

    for idx, l in enumerate(lines):
        t = l["text"]; sz = l["size"]; pg = l["page"]
        is_num   = bool(NUMBERING_RE.match(t))
        is_colon_end = bool(ENDS_COLON_RE.match(t))
        is_phase = bool(ROMAN_PHASE_RE.match(t))
        is_question = t.strip().endswith("?")
        styled   = bool(l.get("bold",0) or l.get("italic",0))
        fam_share = shares.get(l["family"], 0.0)
        rare_family = (l["family"] != major) and (fam_share <= q25_share)

        cond_A = sz >= (doc_body_top + 0.5*size_tol)
        cond_B = (is_num or is_colon_end) and styled and (sz >= (doc_body_top + 0.25*size_tol))
        cond_C = (is_num or is_colon_end) and styled and (sz >= (page_size_p80_map.get(pg, 0.0) + 0.25*size_tol))
        cond_D = styled and rare_family and is_heading_shape(l, shape)
        cond_E = is_phase and (is_heading_shape(l, shape) or (":" in t)) and (sz >= (doc_body_top - 0.9*size_tol))
        cond_F = is_question and is_heading_shape(l, shape) and (l.get("gap_above", 0.0) >= 0.9 * page_gap_q60.get(pg, 0.0))

        # uniform-size rescue — bold + short + left + (big gap or next line more-indented body)
        cond_G = False
        if modes.get("uniform_sizes", False) and not any([cond_A, cond_B, cond_C, cond_D, cond_E, cond_F]):
            left_anchor = l["x0"] <= (q25 + indent_tol)
            shortish    = is_heading_shape(l, shape)
            bold_only   = bool(l.get("bold",0)) and not l.get("italic",0)
            big_gap     = l.get("gap_above", 0.0) >= 0.9 * page_gap_q60.get(pg, 0.0)
            nxt = None
            for j in range(idx+1, len(lines)):
                if lines[j]["page"] != pg: break
                nxt = lines[j]; break
            nxt_more_indented_body = bool(nxt) and (nxt["size"] in body_band) and (nxt["x0"] >= (q50 - indent_tol))
            cond_G = bold_only and shortish and left_anchor and (big_gap or nxt_more_indented_body)

        if cond_A or cond_B or cond_C or cond_D or cond_E or cond_F or cond_G:
            out.append(l)

    out.sort(key=lambda x: (x["page"], x["y0"]))
    return out

# ───────────────────── Promote colon/number run-ins ──────────────────── #
def promote_runins(lines, body_band, body_indent, shape, size_tol, existing_ids):
    q50 = body_indent.get("q50", 0.0)
    q25 = body_indent.get("q25", 0.0)
    indent_tol = max(0.0, (q50 - q25) * 0.5)
    promoted = []

    for idx, l in enumerate(lines):
        if l["id"] in existing_ids:
            continue
        t = l["text"]
        is_colon_end = bool(ENDS_COLON_RE.match(t))
        is_true_num  = bool(PROMO_NUMBERING_RE.match(t))
        is_runin = is_colon_end or is_true_num
        if not is_runin:
            continue

        styled_or_cap = bool(l.get("bold",0) or l.get("italic",0) or starts_capital(t))
        if not styled_or_cap:
            continue
        if not is_heading_shape(l, shape):
            continue

        pg = l["page"]
        nxt = None
        for j in range(idx+1, len(lines)):
            if lines[j]["page"] != pg:
                break
            nxt = lines[j]; break
        if not nxt:
            continue

        nxt_is_body   = nxt["size"] in body_band
        nxt_more_ind  = nxt["x0"] >= (q50 - indent_tol)
        if nxt_is_body and nxt_more_ind:
            promoted.append(l)

    return promoted

# ─────────────── merge wrapped headings (robust to gaps) ─────────────── #
def _has_repeated_upper_token(t, min_len=4):
    toks = re.findall(r"[A-Z]{%d,}" % min_len, t)
    if not toks:
        return False
    cnt = Counter(toks)
    return any(v >= 2 for v in cnt.values())

def _has_repeated_upper_substring(t, min_len=4):
    up = re.sub(r"[^A-Z]", " ", t.upper())
    toks = [tok for tok in up.split() if len(tok) >= min_len]
    up_no_spaces = re.sub(r"\s+", " ", up)
    for tok in set(toks):
        if up_no_spaces.count(tok) >= 2:
            return True
    return False

def merge_wrapped_headings(cands, body_band, page_gap_q60, page_gap_q75, body_indent, size_tol, shape):
    if not cands:
        return []
    out = []
    i = 0
    while i < len(cands):
        cur = cands[i]
        parts = [cur["text"]]
        x0 = cur["x0"]; x1 = cur["x1"]
        y0 = cur["y0"]; y1 = cur["y1"]
        bold = cur["bold"]; italic = cur["italic"]
        fam_counts = Counter([cur["family"]])
        pg = cur["page"]

        j = i + 1
        while j < len(cands):
            nxt = cands[j]
            if nxt["page"] != pg:
                break

            # NEW: do not merge canonical section tokens (prevents "… Introduction")
            if nxt["text"].strip().lower() in TAIL_BAN:
                break

            # two-panel guard
            fill_cur = (cur["x1"] - cur["x0"]) / max(1.0, cur["w"])
            fill_nxt = (nxt["x1"] - nxt["x0"]) / max(1.0, nxt["w"])
            cur_narrow = fill_cur <= max(0.35, shape.get("fill_q40", 0.30) * 0.95)
            nxt_wide   = fill_nxt >= 0.45 and (fill_nxt >= fill_cur + 0.20)
            nxt_upper  = uppercase_ratio(nxt["text"]) >= 0.60
            nxt_repeat = _has_repeated_upper_token(nxt["text"]) or _has_repeated_upper_substring(nxt["text"])
            if cur_narrow and nxt_wide and nxt_upper and nxt_repeat:
                break

            if _has_repeated_upper_token(nxt["text"]) and uppercase_ratio(cur["text"]) >= 0.7:
                break
            if NUM_START_RE.match(nxt["text"]):
                break
            if cur["text"].strip().endswith("?"):
                break

            size_close  = math.isclose(nxt["size"], cur["size"], abs_tol=max(0.5, 0.5*size_tol))
            gap         = nxt["y0"] - y1
            gap_cap     = max(2.2 * max(cur["size"], 1.0), 4.0 * page_gap_q75.get(pg, 0.0))
            gap_ok      = 0 <= gap <= gap_cap

            cdev_cur = center_deviation(cur)
            cdev_nxt = center_deviation(nxt)
            center_pair = (abs(cdev_cur - cdev_nxt) <= 0.03)
            q25 = body_indent.get("q25", 0.0); q50 = body_indent.get("q50", 0.0)
            indent_tol  = max(0.0, (q50 - q25) * 0.5)
            left_pair   = abs(nxt["x0"] - cur["x0"]) <= (indent_tol or 2.0)

            nxt_headingish = is_heading_shape(nxt, shape)
            runin_stop = ENDS_COLON_RE.match(cur["text"]) and (nxt["size"] in body_band) and (nxt["x0"] >= body_indent.get("q50", 0.0) - (indent_tol or 2.0))
            short_tail = (word_count(nxt["text"]) <= 2)

            same_block = size_close and gap_ok and (center_pair or left_pair) and nxt_headingish and not runin_stop
            if not same_block and not (size_close and gap_ok and center_pair and short_tail):
                break

            parts.append(nxt["text"])
            x0 = min(x0, nxt["x0"]); x1 = max(x1, nxt["x1"])
            y1 = max(y1, nxt["y1"])
            bold = max(bold, nxt["bold"]); italic = max(italic, nxt["italic"])
            fam_counts.update([nxt["family"]]); j += 1

        family, _ = fam_counts.most_common(1)[0]
        text = " ".join(p.strip() for p in parts if p.strip())
        out.append({**cur, "text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                    "bold": bold, "italic": italic, "family": family})
        i = j
    return out

# ───────────────────── Stage 2: soft-OR (layout) ─────────────────────── #
def soft_or_filter(lines, size_cands, page_gap_q40, page_gap_q60, body_indent, center_stats):
    out = []
    first_on_page = {}
    for l in lines:
        pg = l["page"]
        if pg not in first_on_page or l["y0"] < first_on_page[pg]["y0"]:
            first_on_page[pg] = l

    for c in size_cands:
        pg = c["page"]
        q40 = page_gap_q40.get(pg, 0.0); q60 = page_gap_q60.get(pg, 0.0)
        gap_tol    = max(0.0, (q60 - q40) * 0.5)
        gap_thresh = q60 - gap_tol
        gap_ok     = (c is first_on_page.get(pg)) or (c["gap_above"] >= gap_thresh)

        q25 = body_indent.get("q25", 0.0); q50 = body_indent.get("q50", 0.0)
        indent_tol = max(0.0, (q50 - q25) * 0.5)
        indent_ok  = (c["x0"] <= (q25 + indent_tol))

        cdev = center_deviation(c)
        c_q75 = center_stats["c_q75"]; c_med = center_stats["c_median"]
        center_tol = max(0.01, 0.5 * abs(c_q75 - c_med))
        center_ok  = cdev <= (c_q75 + center_tol)

        if c.get("is_list_item", False):
            if gap_ok and indent_ok:
                out.append(c)
            continue

        if pg == 1:
            if gap_ok or indent_ok or center_ok or (c is first_on_page.get(pg)):
                out.append(c)
        else:
            if gap_ok or indent_ok or center_ok:
                out.append(c)
    return out

# ──────────────────── Stage 3: light sanity + dates ──────────────────── #
def is_date_like(t: str) -> bool:
    s = t.strip()
    if DATE_RE.match(s) or DIGIT_DATE_RE.match(s):
        return True
    if re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s, re.I) and re.search(r"\b(19|20)\d{2}\b", s, re.I):
        return True
    return False

def compute_page_top_sizes(cands, k=2):
    by_pg = defaultdict(set)
    sizes_by_pg = defaultdict(list)
    for c in cands:
        sizes_by_pg[c["page"]].append(c["size"])
    for pg, arr in sizes_by_pg.items():
        uniq = sorted(set(arr), reverse=True)
        topk = uniq[:min(k, len(uniq))]
        by_pg[pg] = set(topk)
    return by_pg

# ───────────────── Appendix-aware Phase injection ────────────────────── #
def inject_phases_under_appendix(stage2, lines, page_gap_q60, shape):
    appendix_pages = []
    for h in stage2:
        if APPENDIX_RE.match(h["text"] or ""):
            appendix_pages.append((h["page"], h["y1"], h["text"].strip().lower()))

    added = []
    if not appendix_pages:
        return stage2, added

    stage2_ids = {h["id"] for h in stage2}
    for pg, y_min, app_txt in appendix_pages:
        if not app_txt.startswith("appendix a"):
            continue
        phase_lines = [l for l in lines
                       if l["page"] == pg and l["y0"] >= (y_min - 2.0*max(1.0, l["size"]))
                       and ROMAN_PHASE_RE.match(l["text"])]
        for l in phase_lines:
            if l["id"] in stage2_ids:
                continue
            if not is_heading_shape(l, shape):
                if not (l["bold"] and len(l["text"].split()) <= 10):
                    continue
            if l["gap_above"] < 0.6 * page_gap_q60.get(pg, 0.0):
                if not l["bold"]:
                    continue
            h = {**l, "src": "phase", "force": True}
            stage2.append(h); stage2_ids.add(l["id"]); added.append(l["text"])

    stage2.sort(key=lambda x: (x["page"], x["y0"]))
    return stage2, added

# ───────────────── Enhanced recipe tagging ───────────────── #
def tag_list_items(lines, body_indent, page_gap_q60, body_band, modes, shape):
    q25 = body_indent.get("q25", 0.0); q50 = body_indent.get("q50", 0.0)
    indent_tol = max(0.0, (q50 - q25) * 0.5)

    recipe_mode = detect_recipe_mode(lines, body_band, shape)
    dish_heads = find_dish_heads(lines, body_band, shape) if recipe_mode else set()

    for l in lines:
        t = l["text"]
        has_bullet = bool(BULLET_LEADER_RE.match(t))
        has_enum   = bool(ENUM_LIST_RE.match(t))
        faux_o_start  = bool(FAUX_BULLET_O_START_RE.match(t))
        faux_o_inline = bool(FAUX_BULLET_O_INLINE_RE.search(t))
        deeper     = l["x0"] >= (q50 - indent_tol)
        small_gap  = l.get("gap_above", 0.0) <= 0.8 * page_gap_q60.get(l["page"], 0.0)
        true_num_head = bool(NUMBERING_RE.match(t)) and (l["x0"] <= (q25 + indent_tol))

        l["is_list_item"] = (
            (not true_num_head) and (
                (deeper and (has_bullet or has_enum) and small_gap) or
                (has_bullet and small_gap) or
                faux_o_start or
                (recipe_mode and faux_o_inline)
            )
        )

        if recipe_mode:
            if (INGR_UNIT_RE.search(t) or INGR_CORE_RE.search(t) or INSTR_VERB_RE.match(t)) and not (l["bold"] or l["italic"]) and not true_num_head:
                l["is_list_item"] = True
            if re.match(r"^\s*\d+(?:\s+\d/\d)?\s+\w+", t) and (word_count(t) <= 8) and not (l["bold"] or l["italic"]) and not true_num_head:
                l["is_list_item"] = True

    if recipe_mode:
        by_page = defaultdict(list)
        for l in lines: by_page[l["page"]].append(l)
        for pg, arr in by_page.items():
            arr.sort(key=lambda x: x["y0"])
            in_block = False
            for i, l in enumerate(arr):
                low = l["text"].strip().lower()
                if (l["id"] in dish_heads) or (ENDS_COLON_RE.match(low) and low in RECIPE_SUBHEADS):
                    in_block = True
                    continue
                if not in_block:
                    continue
                stop = False
                if (l["id"] in dish_heads):
                    stop = True
                if ENDS_COLON_RE.match(l["text"]) and word_count(l["text"]) <= 4:
                    stop = True
                if l.get("gap_above", 0.0) >= 1.35 * page_gap_q60.get(pg, 0.0):
                    stop = True
                q25 = body_indent.get("q25", 0.0); q50 = body_indent.get("q50", 0.0)
                indent_tol  = max(0.0, (q50 - q25) * 0.5)
                leftish = l["x0"] <= (q25 + indent_tol)
                if l["bold"] and leftish and is_heading_shape(l, body_shape_stats(lines, body_band)):
                    stop = True
                if stop:
                    in_block = False
                else:
                    true_num_head = bool(NUMBERING_RE.match(l["text"])) and (l["x0"] <= (q25 + indent_tol))
                    if not (l["bold"] or l["italic"] or true_num_head or ENDS_COLON_RE.match(l["text"])):
                        l["is_list_item"] = True

    return lines

def final_headings_keep_recall(cands, shape, title_ids, doc_body_top, size_tol, fam_stats, page_top_sizes_map):
    out, seen = [], set()
    for l in cands:
        t = l["text"].strip()
        if not t or PUNCT_ONLY_RE.match(t):
            continue
        if (l["id"] not in title_ids) and is_date_like(t):
            continue
        if TOC_ENTRY_RE.search(t):
            continue
        if ENUMERATOR_RE.match(t):
            continue
        if re.match(r"^\s*(?:19|20)\d{2}\.?\s*$", t):
            continue

        low = t.lower()
        if l.get("is_list_item", False) and low not in RECIPE_SUBHEADS:
            continue

        if l.get("src") == "phase":
            pass_ok = True
        else:
            if l["size"] in page_top_sizes_map.get(l["page"], set()):
                pass_ok = (
                    is_heading_shape(l, shape) or l["bold"] or l["italic"] or
                    NUMBERING_RE.match(t) or ENDS_COLON_RE.match(t) or t.endswith("?")
                )
            else:
                pass_ok = False
                if ENDS_COLON_RE.match(t):
                    styled_or_cap = (l["bold"] or l["italic"] or starts_capital(t))
                    if styled_or_cap or (l["size"] >= (doc_body_top + 0.25*size_tol)):
                        pass_ok = True
                if not pass_ok and (is_heading_shape(l, shape) or l["bold"] or l["italic"] or NUMBERING_RE.match(t) or t.endswith("?")):
                    pass_ok = True

        if not pass_ok:
            continue

        key = (t.lower(), l["page"])
        if key in seen:
            continue
        seen.add(key)
        out.append(l)

    return out

# ─────────────────────────── Level assignment ────────────────────────── #
def assign_levels_mixed(heads, max_levels=6):
    if not heads:
        return []
    by_depth = defaultdict(list)
    for h in heads:
        d = numeric_depth(h["text"])
        if d is not None:
            by_depth[min(d, max_levels)].append(h["size"])
    depth_median = {d: float(np.median(sizes)) for d, sizes in by_depth.items() if sizes}

    def map_depth_to_level(d):
        d = max(1, min(d, max_levels))
        return f"H{d}"

    if depth_median:
        for h in heads:
            d = numeric_depth(h["text"])
            if d is not None:
                h["level"] = map_depth_to_level(d)
        undecided = [h for h in heads if "level" not in h]
        if undecided:
            med_items = sorted(depth_median.items())
            for h in undecided:
                diffs = [(abs(h["size"] - ms), d) for d, ms in med_items]
                _, best_d = min(diffs, key=lambda x: x[0])
                h["level"] = map_depth_to_level(best_d)
        return heads

    sizes_desc = sorted({h["size"] for h in heads}, reverse=True)
    if len(sizes_desc) == 1:
        s2lvl = {sizes_desc[0]: "H1"}
    else:
        diffs = [sizes_desc[i] - sizes_desc[i+1] for i in range(len(sizes_desc)-1)]
        k = min(6-1, len(diffs))
        cut_idxs = sorted(np.argsort(diffs)[-k:])
        boundaries = set(i+1 for i in cut_idxs)
        level = 1
        s2lvl = {}
        for i, s in enumerate(sizes_desc):
            s2lvl[s] = f"H{level}"
            if (i+1) in boundaries:
                level += 1
    for h in heads:
        h["level"] = s2lvl[h["size"]]
    return heads

# ─────────── First content page calibration ─────────── #
def calibrate_first_content_page(heads, cover_offset_guess, h2_abs=1.8, h2_rel=0.14, cdev_thresh=0.22):
    if not heads:
        return set(), set()

    content_page = 1 + (cover_offset_guess or 0)
    P = [h for h in heads if h["page"] == content_page]
    if not P:
        return set(), set()

    C = [h for h in P if numeric_depth(h["text"]) is None
         and not ENDS_COLON_RE.match(h["text"])
         and not h["text"].strip().lower().startswith(("appendix", "phase "))]
    C.sort(key=lambda h: h["y0"])

    h1_ids = []
    if C:
        a = C[0]
        b = None
        for k in range(1, len(C)):
            cand = C[k]
            vgap = cand["y0"] - a["y1"]
            if vgap < 0 or vgap > (3.0 * max(a["size"], cand["size"])): break
            if center_deviation(a) <= cdev_thresh and center_deviation(cand) <= cdev_thresh: b = cand
            break
        h1_ids = [a["id"]] + ([b["id"]] if b else [])
        for h in heads:
            if h["id"] in h1_ids:
                h["level"] = "H1"
                h["co_h1_protect"] = True

    h2_sizes = set()
    if C:
        start_y = None
        if h1_ids:
            last = max((h for h in heads if h["id"] in h1_ids and h["page"] == content_page), key=lambda z: z["y1"], default=None)
            if last: start_y = last["y1"]
        H2C = [h for h in P if numeric_depth(h["text"]) is None
               and not ENDS_COLON_RE.match(h["text"])
               and (start_y is None or h["y0"] >= start_y + 0.1)]
        if H2C:
            sz_counts = Counter(h["size"] for h in H2C)
            s2 = max(sz_counts.keys(), key=lambda s: (sz_counts[s], s))
            h2_sizes.add(s2)
            for h in P:
                if ENDS_COLON_RE.match(h["text"]): continue
                if abs(h["size"] - s2) <= max(h2_abs, h2_rel * s2):
                    if h.get("level") != "H1":
                        h["level"] = "H2"

    h1_sizes = {h["size"] for h in heads if h.get("co_h1_protect")}
    return h1_sizes, h2_sizes

def reassert_numeric_levels(heads, max_levels=6):
    for h in heads:
        d = numeric_depth(h["text"])
        if d is not None:
            d = max(1, min(d, max_levels))
            h["level"] = f"H{d}"
    return heads

def enforce_top_size_h1_and_demote_others(heads):
    if not heads:
        return heads
    if any(numeric_depth(h["text"]) == 1 for h in heads):
        return heads
    counts = Counter(h["size"] for h in heads)
    s_max = max(counts) if counts else None
    if s_max is None: return heads
    if counts[s_max] <= 3:
        for h in heads:
            if getattr(h, "co_h1_protect", False):
                continue
            if math.isclose(h["size"], s_max, abs_tol=0.25):
                h["level"] = "H1"
            elif h.get("level") == "H1" and h["size"] < (s_max - 0.75):
                h["level"] = "H2"
    return heads

def normalize_primary_H2_band(heads):
    if not heads:
        return heads
    h1_sizes = [h["size"] for h in heads if h.get("level") == "H1"]
    h1_top = max(h1_sizes) if h1_sizes else None

    cands = []
    for h in heads:
        txt = h["text"].strip(); low = txt.lower()
        if h["page"] > 3: continue
        if ENDS_COLON_RE.match(txt) or txt.endswith("?") or NUM_PATTERN.match(txt): continue
        if low.startswith("appendix") or low.startswith("phase "): continue
        if h.get("level") == "H1": continue
        cands.append(h)

    if not cands:
        return heads

    size_counts = Counter(h["size"] for h in cands)
    below = [s for s in size_counts if (h1_top is None or s < (h1_top - 0.25))]
    if not below: below = list(size_counts.keys())
    h2_size = max(below, key=lambda s: (size_counts[s], s))

    for h in cands:
        if math.isclose(h["size"], h2_size, abs_tol=0.8):
            if h.get("level") not in ("H1", "H2"):
                h["level"] = "H2"
    return heads

def demote_runins_by_shape(heads, body_indent, doc_body_top, size_tol, page_gap_q60):
    q25 = body_indent.get("q25", 0.0); q50 = body_indent.get("q50", 0.0)
    indent_tol = max(0.0, (q50 - q25) * 0.5)
    for h in heads:
        t = h["text"].strip()
        low = t.lower()

        if low.startswith("appendix"):
            h["level"] = "H2"; continue
        if low.startswith("phase "):
            h["level"] = "H3"; continue

        if ENDS_COLON_RE.match(t):
            short = len(t.replace(":", "").split()) <= 3
            strong_gap = h.get("gap_above", 0.0) >= 0.8 * page_gap_q60.get(h["page"], 0.0)
            if short and strong_gap:
                h["level"] = "H3"; continue
            deep = h["x0"] >= (q50 - indent_tol)
            h["level"] = "H4" if deep else "H3"
            continue
    return heads

def demote_between_H2_corridor(heads):
    idxs_H2 = [i for i,h in enumerate(heads) if h.get("level") == "H2" and not h["text"].lower().startswith("appendix")]
    if not idxs_H2:
        return heads
    for k in range(len(idxs_H2)-1):
        i0, i1 = idxs_H2[k], idxs_H2[k+1]
        for j in range(i0+1, i1):
            h = heads[j]
            txt = h["text"].strip(); low = txt.lower()
            if low.startswith("appendix") or low.startswith("phase "): continue
            if txt.endswith("?") and h.get("level") == "H2": h["level"] = "H3"
            if (ENDS_COLON_RE.match(txt) or word_count(txt) <= 2) and h.get("level") == "H2": h["level"] = "H3"
    return heads

def demote_isolated_short_H2(heads):
    idxs_H2 = [i for i,h in enumerate(heads) if h.get("level") == "H2"]
    for i in idxs_H2:
        h = heads[i]
        low = h["text"].lower()
        if low.startswith("appendix") or low.startswith("phase "): continue
        if not (ENDS_COLON_RE.match(h["text"]) or word_count(h["text"]) <= 2): continue
        prev = next((j for j in range(i-1, -1, -1) if heads[j].get("level") == "H2"), None)
        nxt  = next((j for j in range(i+1, len(heads)) if heads[j].get("level") == "H2"), None)
        if prev is not None and nxt is not None: h["level"] = "H3"
    return heads

def appendix_numeric_level_fix(heads):
    current_appendix = None
    for h in heads:
        txt = h["text"].strip()
        low = txt.lower()
        m = re.match(r"^\s*appendix\s+([a-z])\b", low, re.I)
        if m:
            current_appendix = m.group(1).upper()
            h["level"] = "H2"; continue

        if current_appendix:
            mnum = re.match(r"^\s*(\d+(?:\.\d+)*)\b", txt)
            if mnum:
                depth = 1 + mnum.group(1).count(".")
                level = 2 + depth
                level = min(level, 6)
                h["level"] = f"H{level}"
    return heads

def smooth_levels_by_size_ranges(heads, h1_anchor_sizes, h2_anchor_sizes, h1_rel=0.14, h2_rel=0.14, h1_abs=1.8, h2_abs=1.8):
    def in_any(sz, anc, rel, abspt):
        for s in anc:
            if abs(sz - s) <= max(abspt, rel*s):
                return True
        return False

    max_page_for_h1 = min([h["page"] for h in heads], default=1) + 1
    for h in heads:
        t = h["text"]
        if numeric_depth(t) is not None or ENDS_COLON_RE.match(t):
            continue
        if h["page"] <= max_page_for_h1 and in_any(h["size"], h1_anchor_sizes, h1_rel, h1_abs):
            h["level"] = "H1"
            h["co_h1_protect"] = True
        elif in_any(h["size"], h2_anchor_sizes, h2_rel, h2_abs):
            if h.get("level") not in ("H1","H2"):
                h["level"] = "H2"
    return heads

def _collapse_many_double_letters(s: str, min_hits: int = 3) -> str:
    if len(re.findall(r"([A-Za-z])\1", s)) >= min_hits:
        return re.sub(r"([A-Za-z])\1+", r"\1", s)
    return s

def _canonicalize_rfp_line(s: str) -> str:
    if re.match(r"^\s*R\s*F\s*P\s*:\s*", s, re.I):
        return "RFP:Request for Proposal"
    return s

# ───────────────────────────── Title (p1 block) ───────────────────────────── #
def build_title_and_ids(lines, body_band):
    p1 = [l for l in lines if l["page"] == 1]
    if not p1:
        return "", set(), float("inf")
    thr = max(body_band) if body_band else 0.0
    big = [l for l in p1 if l["size"] > thr]

    # fallback — bold+centered short lines in top half
    if not big:
        top_half = [l for l in p1 if l["y0"] <= 0.55 * l["h"]]
        cand = []
        for l in top_half:
            t = l["text"].strip()
            if not t or t.endswith("."): continue
            if word_count(t) > 12: continue
            bold_or_caps = l["bold"] or uppercase_ratio(t) >= 0.35
            if not bold_or_caps: continue
            cand.append(l)
        big = cand

    if not big:
        return "", set(), float("inf")

    # group contiguous display lines into blocks (same tier)
    blocks, cur = [], []
    for l in sorted(big, key=lambda z: (z["y0"], z["x0"])):
        if not cur:
            cur = [l]; continue
        prev = cur[-1]
        near = (0 <= (l["y0"] - prev["y1"]) <= 1.6 * max(prev["size"], 1.0))
        same_tier = math.isclose(l["size"], prev["size"], abs_tol=1.5) or l["size"] >= prev["size"] - 1.2
        if near and same_tier and not ENDS_COLON_RE.match(l["text"]) and word_count(l["text"]) <= 10 and not l["text"].strip().endswith("."):
            cur.append(l)
        else:
            blocks.append(cur); cur = [l]
    if cur: blocks.append(cur)

    def centeredness(line):
        return abs(((line["x0"]+line["x1"])/2) - line["w"]/2) / line["w"]

    def block_score(B):
        first = B[0]
        txt = " ".join(b["text"].strip() for b in B if b["text"].strip())
        wc = word_count(txt)
        ends_dot = txt.endswith(".")
        colon    = ENDS_COLON_RE.match(txt) is not None
        ucr      = uppercase_ratio(txt)
        size_max = max(b["size"] for b in B)
        y0_min   = min(b["y0"] for b in B)
        cdev     = centeredness(first)

        score = 0.0
        score += 3.0 * size_max
        score += 2.0 * (1.0 - min(1.0, cdev*8.0))
        score += 1.5 * (1.0 if (2 <= wc <= 12) else 0.0)
        score += 0.7 * (1.0 if ucr >= 0.25 else 0.0)
        score -= 3.0 * (1.0 if ends_dot else 0.0)
        score -= 2.0 * (1.0 if colon else 0.0)
        score -= 0.004 * y0_min
        return score

    best = max(blocks, key=block_score)

    # allow aligned line just above (subtitle)
    first = best[0]
    prev_candidates = [l for l in big if l["y1"] <= first["y0"]]
    if prev_candidates:
        prev = max(prev_candidates, key=lambda z: z["y1"])
        c_first = (first["x0"]+first["x1"])/(2*first["w"])
        c_prev  = (prev ["x0"]+prev ["x1"])/(2*prev ["w"])
        if 0 <= (first["y0"] - prev["y1"]) <= 2.0 * max(prev["size"], 1.0) and abs(c_first - c_prev) <= 0.08:
            if not ENDS_COLON_RE.match(prev["text"]) and word_count(prev["text"]) <= 10 and not prev["text"].strip().endswith("."):
                best = [prev] + best

    # allow 1–2 centered tails just below — but don't take section tokens
    last = best[-1]
    c_block = (last["x0"]+last["x1"]) / (2*last["w"])
    followers = [l for l in sorted(big, key=lambda z: (z["y0"], z["x0"])) if l["y0"] >= last["y1"]]
    tails = []
    for f in followers:
        vgap = f["y0"] - last["y1"]
        if vgap < 0 or vgap > 2.2 * max(last["size"], f["size"]):
            break
        c_f = (f["x0"]+f["x1"]) / (2*f["w"])
        if abs(c_f - c_block) <= 0.08 and not is_date_like(f["text"]) and not f["text"].strip().endswith("."):
            low = f["text"].strip().lower()
            if low in TAIL_BAN or NUMBERING_RE.match(f["text"]):
                break
            if word_count(f["text"]) <= 22:
                tails.append(f)
                last = f
                if len(tails) >= 2:
                    break
            else:
                break
        else:
            break
    if tails:
        best = best + tails

    # NEW: trim any lines that are exactly section tokens (e.g., "Introduction")
    trimmed = [b for b in best if b["text"].strip().lower() not in TAIL_BAN]
    if trimmed:
        best = trimmed

    # plausibility + cleanup
    page_h = best[0]["h"]
    y_top  = min(b["y0"] for b in best)
    y_bot  = max(b["y1"] for b in best)
    y_ctr_frac = ((y_top + y_bot) / 2.0) / max(1.0, page_h)
    bottomish = y_ctr_frac >= 0.68
    bodies_center = [l for l in p1 if (l["size"] in body_band and 0.35*page_h <= l["y0"] <= 0.72*page_h)]
    has_center_body = len(bodies_center) >= 3
    page_max_size = max(l["size"] for l in p1) if p1 else 0.0
    block_max_size = max(b["size"] for b in best)
    text_lines = []

    for k, b in enumerate(best):
        tt = b["text"].strip()
        if k == 0 and re.match(r"^\s*R\s*F\s*P\s*:\s*", tt, re.I):
            tt = _collapse_many_double_letters(tt, min_hits=2)
            tt = _canonicalize_rfp_line(tt)
        else:
            tt = _collapse_many_double_letters(tt, min_hits=4)
        text_lines.append(tt)

    text_block = " ".join(t for t in text_lines if t)
    ends_bang  = text_block.strip().endswith("!")
    biggest_bottom = bottomish and (block_max_size >= page_max_size - 0.01) and has_center_body
    if ends_bang or biggest_bottom:
        return "", set(), float("inf")

    title = squeeze_repeats_soft(fix_encoding(text_block.strip()))
    ids   = {b["id"] for b in best}
    top_y = y_top
    return title, ids, top_y

# ─────────────────────────── split multi-numbered ─────────────────────────── #
def split_multi_numbered(heads):
    out = []
    for h in heads:
        t = h["text"].strip()
        if re.search(r"\s\d+(?:\.\d+)*[.)]?\s", t):
            parts = [p.strip() for p in SPLIT_MULTI_NUM_RE.split(t) if p.strip()]
            if len(parts) > 1 and NUM_START_RE.match(parts[0]):
                for p in parts:
                    out.append({**h, "text": p})
                continue
        out.append(h)
    return out

def detect_cover_offset(page_p80):
    if 1 not in page_p80 or len(page_p80) <= 1:
        return 0
    p1 = page_p80[1]
    others = [v for k,v in page_p80.items() if k != 1]
    if not others:
        return 0
    if p1 >= (max(others) + 6.0):
        return 1
    return 0

# ────────────────────────────── Post filters ─────────────────────────────── #
def suppress_narrative_colon(heads):
    keep = []
    for h in heads:
        t = h["text"].strip()
        if ENDS_COLON_RE.match(t):
            if word_count(t) >= 6 and NARRATIVE_VERBS_RE.search(t):
                continue
        keep.append(h)
    return keep

def normalize_for_each_levels(heads):
    for h in heads:
        if FOR_EACH_RE.match(h["text"]):
            h["level"] = "H4"
    return heads

def suppress_appendix_noise(heads):
    out = []
    current_app = None
    for h in heads:
        low = h["text"].strip().lower()
        m = re.match(r"^\s*appendix\s+([a-z])\b", low)
        if m:
            current_app = m.group(1).upper()
            out.append(h)
            continue

        if not current_app:
            out.append(h); continue

        if current_app == "A":
            if low.startswith("phase "):
                out.append(h)
            continue

        if current_app == "B":
            if re.match(r"^\s*\d+(?:\.\d+)*\b", h["text"]):
                out.append(h)
            continue

        if current_app == "C":
            continue

        out.append(h)
    return out

def suppress_toc_entries(heads):
    return [h for h in heads if not TOC_ENTRY_RE.search(h["text"].strip())]

def promote_known_section_heads(heads):
    KEY = {"revision history", "table of contents", "acknowledgements"}
    for h in heads:
        low = h["text"].strip().lower()
        if h["page"] <= 5 and low in KEY:
            h["level"] = "H1"
    return heads

def suppress_cover_heads(heads, cover_offset, title_text):
    if cover_offset != 1:
        return heads
    out = []
    for h in heads:
        if h["page"] != 1:
            out.append(h); continue
        t = h["text"].strip()
        low = t.lower()
        if t == title_text:
            out.append(h); continue
        if NUMBERING_RE.match(t):
            out.append(h); continue
        if low in {"table of contents","revision history","acknowledgements","abstract","executive summary"}:
            out.append(h); continue
        if COVER_NOISE_RE.match(t):
            continue
        out.append(h)
    return out

REV_TABLE_HEADERS = {"version","date","remarks","identifier","reference"}

def suppress_table_column_headers(heads):
    by_page = defaultdict(list)
    for h in heads: by_page[h["page"]].append(h)
    keep = []
    for pg, arr in by_page.items():
        pure_nums = sum(bool(re.fullmatch(r"\d+(?:\.\d+)*", x["text"].strip())) for x in arr)
        is_rev_table = pure_nums >= 3
        for h in arr:
            low = h["text"].strip().lower()
            if is_rev_table and low in REV_TABLE_HEADERS and word_count(low) <= 2:
                continue
            keep.append(h)
    return keep

def suppress_running_headers(heads, npages, top_frac=0.22, min_pages_ratio=0.3):
    by_text = defaultdict(list)
    for h in heads:
        by_text[h["text"].strip().lower()].append(h)

    keep = []
    threshold = max(3, int(math.ceil(min_pages_ratio * max(1, npages))))
    for txt, occs in by_text.items():
        wc = word_count(txt)
        if wc > 2:
            keep.extend(occs); continue
        if len(occs) < threshold:
            keep.extend(occs); continue
        tops = [o["y0"]/max(1.0, o["h"]) for o in occs]
        if np.median(tops) <= top_frac:
            continue
        keep.extend(occs)
    keep.sort(key=lambda h: (h["page"], h["y0"]))
    return keep

# ───────────────── Single-page headline rescue ───────────────── #
def rescue_headline_singlepage(lines, page_gap_q60, doc_body_top, size_tol):
    best = None; best_score = -1e9
    for l in lines:
        t = l["text"].strip()
        if not t or len(t) > 80:
            continue
        if l["size"] < (doc_body_top + 0.35 * max(0.5, size_tol)):
            continue
        alpha = alpha_ratio(t)
        if alpha < 0.45:
            continue
        g = max(0.0, l.get("gap_above", 0.0))
        g_ref = max(1.0, page_gap_q60.get(l["page"], 1.0))
        cdev = center_deviation(l)
        wc = word_count(t)
        if wc > 10:
            continue
        ends_bang = 1.0 if t.endswith("!") else 0.0
        punct_ratio = (len(re.findall(r"[^\w\s’'-]", t)) / max(1, len(t)))
        long_tok = any(len(tok) >= 14 for tok in t.split())
        wbonus = 0.8 if wc <= 3 else (0.4 if wc <= 5 else 0.1)
        score = (
            (g / g_ref) +
            0.75 * l["size"] +
            0.85 * (1.0 - min(1.0, cdev*6.0)) +
            wbonus +
            0.9 * ends_bang +
            0.8 * max(0.0, alpha - 0.45) - 
            3.5 * punct_ratio - 
            (0.45 if long_tok else 0.0)
        )
        if score > best_score:
            best_score, best = score, l
    if not best:
        return []
    return [{**best, "level": "H1"}]

def flyer_consolidation(heads, npages):
    if npages != 1 or not heads:
        return heads
    def is_allcaps(h):
        t = h["text"].strip()
        alpha = re.sub(r"[^A-Za-z]", "", t)
        return bool(alpha) and uppercase_ratio(t) >= 0.85
    H1s = [h for h in heads if h.get("level") == "H1" and not ENDS_COLON_RE.match(h["text"].strip())]
    ac_H1 = [h for h in H1s if is_allcaps(h)]
    if not ac_H1:
        return [h for h in heads if not (ENDS_COLON_RE.match(h["text"].strip()) and word_count(h["text"]) <= 2)]
    def banner_score(h):
        t = h["text"].strip()
        wc = word_count(t)
        fill = (h["x1"] - h["x0"]) / max(1.0, h["w"])
        cdev = center_deviation(h)
        toks = [tok for tok in re.findall(r"[A-Za-z]+", t.upper())]
        has_repeat = any(toks.count(tok) >= 2 for tok in set(toks))
        return (
            2.2 * (1.0 - min(1.0, cdev * 8.0)) +
            1.4 * (1.0 - min(1.0, fill)) +
            (0.9 if wc <= 3 else 0.4 if wc <= 5 else 0.0) - 
            (1.6 if has_repeat else 0.0)
        )
    banner = max(ac_H1, key=banner_score)
    kept = []
    for h in heads:
        t = h["text"].strip()
        if h is banner:
            kept.append(h); continue
        if h.get("level") == "H1" and is_allcaps(h) and h["page"] == banner["page"]:
            continue
        if ENDS_COLON_RE.match(t) and word_count(t) <= 2:
            continue
        kept.append(h)
    return kept

# ────────────────────────────────── main ───────────────────────────────── #
def extract(pdf_path):
    lines, size_freq, family_freq, npages = parse_pdf(pdf_path)
    if not lines:
        return {"title":"", "outline": [], "_debug": {}}

    page_gap_q40, page_gap_q60, page_gap_q75 = add_line_gaps(lines)
    band        = derive_paragraph_band(size_freq)
    size_tol    = size_step_tolerance(size_freq)
    body_indent = body_indent_stats(lines, band)
    page_p80    = per_page_size_p80(lines)
    fam_stat    = family_stats(family_freq)
    shape       = body_shape_stats(lines, band)
    doc_body_top= max(band) if band else 0.0
    modes       = doc_modes(size_freq, lines)

    lines       = tag_list_items(lines, body_indent, page_gap_q60, band, modes, shape)

    cover_offset_guess = detect_cover_offset(page_p80)
    title, title_ids, title_top_y0 = build_title_and_ids(lines, band)

    stage1     = size_candidates(lines, band, size_tol, page_p80, fam_stat, shape, page_gap_q60, modes, body_indent)
    stage1_ids = {l["id"] for l in stage1}
    promoted   = promote_runins(lines, band, body_indent, shape, size_tol, stage1_ids)
    stage1_all = sorted(stage1 + promoted, key=lambda l: (l["page"], l["y0"]))
    stage1m    = merge_wrapped_headings(stage1_all, band, page_gap_q60, page_gap_q75, body_indent, size_tol, shape)

    center_stat= center_stats_for_top(stage1m)
    stage2     = soft_or_filter(lines, stage1m, page_gap_q40, page_gap_q60, body_indent, center_stat)
    stage2     = [h for h in stage2 if h["id"] not in title_ids]
    stage2     = [h for h in stage2 if not (h["page"] == 1 and h["y1"] <= title_top_y0)]

    stage2, phase_added = inject_phases_under_appendix(stage2, lines, page_gap_q60, shape)

    page_top_sizes_map = compute_page_top_sizes(stage2, k=2)
    stage3     = final_headings_keep_recall(stage2, shape, title_ids, doc_body_top, size_tol, fam_stat, page_top_sizes_map)

    if not stage3 and npages == 1:
        stage3 = rescue_headline_singlepage(lines, page_gap_q60, doc_body_top, size_tol)

    heads      = assign_levels_mixed(stage3, max_levels=6)
    h1_anchor_sizes, h2_anchor_sizes = calibrate_first_content_page(heads, cover_offset_guess)

    heads      = enforce_top_size_h1_and_demote_others(heads)
    heads      = reassert_numeric_levels(heads)
    heads      = normalize_primary_H2_band(heads)
    heads      = demote_runins_by_shape(heads, body_indent, doc_body_top, size_tol, page_gap_q60)
    heads      = appendix_numeric_level_fix(heads)
    heads      = split_multi_numbered(heads)
    heads      = smooth_levels_by_size_ranges(heads, h1_anchor_sizes, h2_anchor_sizes,
                                              h1_rel=0.14, h2_rel=0.14, h1_abs=1.8, h2_abs=1.8)
    heads      = demote_between_H2_corridor(heads)
    heads      = demote_isolated_short_H2(heads)

    heads      = suppress_table_column_headers(heads)
    heads      = suppress_running_headers(heads, npages)

    heads      = suppress_narrative_colon(heads)
    heads      = normalize_for_each_levels(heads)
    heads      = suppress_appendix_noise(heads)
    heads      = promote_known_section_heads(heads)
    heads      = suppress_toc_entries(heads)
    heads      = suppress_cover_heads(heads, cover_offset_guess, title)

    heads      = flyer_consolidation(heads, npages)

    if npages == 1 and not heads:
        heads = rescue_headline_singlepage(lines, page_gap_q60, doc_body_top, size_tol)

    if cover_offset_guess:
        for h in heads:
            if h["page"] > 1:
                h["page"] = h["page"] - cover_offset_guess

    if npages == 1:
        for h in heads:
            h["page"] = 0

    seen = set()
    final_heads = []
    for h in heads:
        k = (h["text"].strip().lower(), h["page"])
        if k in seen: continue
        seen.add(k)
        final_heads.append(h)

    outline = [{
        "level": h["level"],
        "text":  h["text"].strip(),
        "page":  h["page"]
    } for h in final_heads if h["text"].strip() != title]

    debug = {
        "body_band_sizes": sorted(band),
        "size_tol": size_tol,
        "page_gap_q40": page_gap_q40,
        "page_gap_q60": page_gap_q60,
        "body_indent": body_indent,
        "page_size_p80": page_p80,
        "family_major_q25": {"q25": fam_stat["q25"], "major": fam_stat["major"]},
        "shape": shape,
        "modes": modes,
        "recipe_mode": detect_recipe_mode(lines, band, shape),
        "size_candidates": [s["text"] for s in stage1],
        "promoted_runins": [s["text"] for s in promoted],
        "stage1_merged":   [s["text"] for s in stage1m],
        "stage2_soft_or":  [s["text"] for s in stage2],
        "final_headings":  [s["text"] for s in final_heads],
        "title": title,
        "phase_injected": phase_added,
        "cover_offset": cover_offset_guess,
        "h1_anchor_sizes": list(map(float, h1_anchor_sizes)),
        "h2_anchor_sizes": list(map(float, h2_anchor_sizes))
    }
    return {"title": title, "outline": outline, "_debug": debug}

def json_safe(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, set):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.bool_,)):  # recent numpy
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o

# ───── NEW main ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, pathlib, os, json, sys

    ap = argparse.ArgumentParser(
        description="Batch‑extract headings from every *.pdf in INPUT_DIR " 
                    "and write <name>.json files to OUTPUT_DIR.")
    ap.add_argument("--input",  default="/app/input",
                    help="directory containing input PDFs (default: /app/input)")
    ap.add_argument("--output", default="/app/output",
                    help="directory for JSON outputs (default: /app/output)")
    args = ap.parse_args()

    in_dir  = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] no PDFs found in {in_dir}", file=sys.stderr)

    for pdf_path in pdfs:
        try:
            res = json_safe(extract(str(pdf_path)))
            res.pop("_debug", None) 
        except Exception as e:
            print(f"[ERROR] failed on {pdf_path.name}: {e}", file=sys.stderr)
            continue

        out_file = out_dir / f"{pdf_path.stem}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        print(f"[OK] {pdf_path.name} → {out_file.name}")

