"""
Evaluate tag-to-summary relevance with BM25 (two modes).

- tag-each: each tag is a query scored against its book summary.
- tag-merged: all tags merged into one query with configurable TF handling and
  optional length normalization.

Inputs
------
1) Excel with columns [書名, 分類, 標籤]
2) CSV with columns [書名, 作者, 書籍簡介, 書籍分類第二層]

Outputs
-------
- CSV with per-book metrics (tag-each and/or tag-merged)
- JSONL with per-tag scores and merged score per book

Usage:
python evaluation.py \
  --excel data/process/十類書名_標籤.xlsx \
  --books-csv data/process/processed_books.csv \
  --out-csv data/process/artifacts/bm25_eval_summary.csv \
  --out-jsonl data/process/artifacts/bm25_eval_details.jsonl \
  --mode both --tokenizer auto --ngram 2 --k1 1.2 --b 0.75 \
  --merged-tf binary --normalize-query true --topk-mean 5
"""
import os, re, json, math, argparse
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from bookrec.embeddings import EmbeddingClient
from scipy.spatial.distance import cosine as cosine_distance

# ─────────────────────────────────────────────────────────
# 路徑（預設：repo/data/process 與 artifacts）
# 如需覆寫，把 BOOKREC_ROOT 設成你的 repo 根目錄
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
ART  = BASE_DIR / "data" / "process"
TAG_CACHE_JSON  = ART / "tag_cache.json"

# -------- Tokenization --------

def _try_import_jieba():
    try:
        import jieba  # type: ignore
        return jieba
    except Exception:
        return None

_jieba = _try_import_jieba()

_ZH_PUNCS = set("，。！？：；、「」『』（）—…·《》〈〉-～—‧·︰、\n\t ")
_LAT_PUNCS = set(",.!?;:'\"()[]{}<>/\\|@#$%^&*-_=+`~\n\t ")

_DEF_SPLIT_RE = re.compile(r"[\s,;；，。.!?、]+")


def tokenize(text: str, mode: str = "auto", ngram: int = 2) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if mode == "jieba" or (mode == "auto" and _jieba is not None):
        return [w.strip() for w in _jieba.cut(t) if w.strip()]
    if mode == "whitespace":
        return [w for w in _DEF_SPLIT_RE.split(t) if w]
    chars = [c for c in t if c not in _ZH_PUNCS and c not in _LAT_PUNCS]
    if ngram <= 1:
        return chars
    return ["".join(chars[i:i+ngram]) for i in range(max(0, len(chars)-ngram+1))] or chars

# -------- BM25 --------
@dataclass
class BM25Params:
    k1: float = 1.2
    b: float = 0.75

class BM25:
    def __init__(self, corpus_tokens: List[List[str]], params: BM25Params = BM25Params()):
        self.params = params
        self.N = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - df_t + 0.5)/(df_t + 0.5) + 1.0) for t, df_t in df.items()}
        self.doc_tf = []
        for doc in corpus_tokens:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf.append(tf)

    def score(self, q_tokens: List[str], doc_id: int) -> float:
        tf = self.doc_tf[doc_id]
        dl = self.doc_len[doc_id]
        denom_base = self.params.k1 * (1.0 - self.params.b + self.params.b * dl / max(1e-9, self.avgdl))
        s = 0.0
        for q in q_tokens:
            f = tf.get(q, 0)
            if f == 0: continue
            idf = self.idf.get(q)
            if idf is None: continue
            num = f * (self.params.k1 + 1.0)
            denom = f + denom_base
            s += idf * (num / denom)
        return float(s)

# -------- Evaluation --------

def load_inputs(excel: str, books_csv: str):
    return pd.read_excel(excel), pd.read_csv(books_csv).drop_duplicates(subset=["書名"]).reset_index(drop=True)

def parse_tags(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = re.split(r"[,\uFF0C]+", cell)
    seen, out = set(), []
    for t in [p.strip() for p in parts if p.strip()]:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def merge_query_tokens(tags: List[str], tokenizer: str, ngram: int, merged_tf: str) -> List[str]:
    all_tokens = []
    for tag in tags:
        all_tokens.extend(tokenize(tag, tokenizer, ngram))
    if not all_tokens: return []
    tf: Dict[str, int] = {}
    for t in all_tokens: tf[t] = tf.get(t,0)+1
    if merged_tf == "raw":
        return all_tokens
    if merged_tf == "binary":
        return list(tf.keys())
    if merged_tf == "log":
        out = []
        for t,f in tf.items():
            out.extend([t]*max(1,int(round(math.log(1+f)))))
        return out
    return list(tf.keys())

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u3000\s]+", " ", text)
    return text.strip()


def evaluate_bm25(
    excel: str, books_csv: str, out_csv: str, out_jsonl: str,
    mode: str="both", tokenizer: str="auto", ngram: int=2, k1: float=1.2, b: float=0.75,
    merged_tf: str="binary", normalize_query: bool=True, topk_mean: int=5):

    tag_df, books_df = load_inputs(excel, books_csv)
    docs = []; title_to_docid = {}
    for _,row in books_df.iterrows():
        title = str(row.get("書名","")).strip()
        summary = str(row.get("書籍簡介","")).strip()
        tokens = tokenize(summary, tokenizer, ngram)
        title_to_docid[title] = len(docs)
        docs.append(tokens)
    bm25 = BM25(docs, BM25Params(k1=k1,b=b))

    details_path = Path(out_jsonl); details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows=[]
    with open(details_path,"w",encoding="utf-8") as fout:
        for _,row in tag_df.iterrows():
            title = str(row.get("書名","")).strip()
            tags=parse_tags(row.get("標籤",""))
            doc_id = title_to_docid.get(title)
            if doc_id is None: continue

            te_avg=te_max=te_cov=te_topk_mean=0.0; te_max_tag=""; te_scores=[]
            if mode in ("tag-each","both"):
                for tag in tags:
                    q_tokens=tokenize(tag,tokenizer,ngram)
                    s=bm25.score(q_tokens,doc_id)
                    te_scores.append((tag,s))
                if te_scores:
                    vals=[s for _,s in te_scores]
                    te_avg=sum(vals)/len(vals)
                    te_max_tag,te_max=max(te_scores,key=lambda x:x[1])
                    te_cov=sum(1 for v in vals if v>0)/len(vals)
                    k=max(1,min(topk_mean,len(vals)))
                    te_topk_mean=sum(sorted(vals,reverse=True)[:k])/k

            tm_score=tm_norm=0.0; tm_len=0
            if mode in ("tag-merged","both"):
                mq_tokens=merge_query_tokens(tags,tokenizer,ngram,merged_tf)
                tm_len=len(mq_tokens)
                tm_score=bm25.score(mq_tokens,doc_id) if mq_tokens else 0.0
                tm_norm=(tm_score/max(1,tm_len)) if normalize_query and tm_len>0 else tm_score

            rec={"title":title}
            if mode in ("tag-each","both"):
                rec["tag_each_scores"]=[{"tag":t,"bm25":round(s,6)} for t,s in te_scores]
            if mode in ("tag-merged","both"):
                rec.update({"tag_merged_score":round(tm_score,6),"tag_merged_len":tm_len,"tag_merged_norm":round(tm_norm,6)})
            fout.write(json.dumps(rec,ensure_ascii=False)+"\n")

            row_out={"書名":title,"標籤數":len(tags)}
            if mode in ("tag-each","both"):
                row_out.update({"TE_avg":round(te_avg,6),"TE_max":round(te_max,6),"TE_max_tag":te_max_tag,"TE_coverage":round(te_cov,6),f"TE_top{topk_mean}_mean":round(te_topk_mean,6)})
            if mode in ("tag-merged","both"):
                row_out.update({"TM_score":round(tm_score,6),"TM_len":tm_len,"TM_norm":round(tm_norm,6),"TM_tf":merged_tf,"TM_norm_on":normalize_query})
            summary_rows.append(row_out)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_csv,index=False)
    print(f"✓ Wrote summary CSV: {out_csv}")
    print(f"✓ Wrote details JSONL: {details_path}")



def evaluate_cosine(
    excel: str, books_csv: str, out_csv: str, out_jsonl: str,
    mode: str="both", embedding_level: str="sentence-max", normalize_query: bool=True, topk_mean: int=5
):
    ec = EmbeddingClient(model="text-embedding-3-small")
    tag_df, books_df = load_inputs(excel, books_csv)
    # 1. build embeddings
    docs = []
    title_to_docid = {}

    for _, row in books_df.iterrows():
        title = str(row.get("書名","")).strip()
        summary = str(row.get("書籍簡介","")).strip()
        sentences = summary.split("。")  # 中文簡單用句號切句
        sentences = [clean_text(s) for s in sentences]
        sentences = [s for s in sentences if s]
        start_idx = len(docs)
        title_to_docid[title] = list(range(start_idx, start_idx + len(sentences)))
        docs.extend(sentences)

    embeddings = ec.embed_texts(docs, batch_size=50)
    title_to_embedding = {}
    for title, idx in title_to_docid.items():
        title_to_embedding[title] = [embeddings[i] for i in idx]

    # 2. load cached tag embeddings
    cache = {}
    if TAG_CACHE_JSON.exists():
        cache = json.loads(TAG_CACHE_JSON.read_text(encoding="utf-8"))

    # 3. evaluate
    details_path = Path(out_jsonl); details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows=[]
    with open(details_path,"w",encoding="utf-8") as fout:
        for _, row in tag_df.iterrows():
            title = str(row.get("書名","")).strip()
            tags = str(row.get("標籤","")).split(",")  # 假設逗號分隔
            book_emb = title_to_embedding.get(title)
            if book_emb is None or not tags:
                continue
                
            if mode == "tag-each" or mode == "both":
                te_scores = []
                for tag in tags:
                    tag_emb = cache.get(tag.strip())
                    if tag_emb is None:
                        continue
                    sents = [np.ravel(e) for e in book_emb]
                    s_list = [1 - cosine_distance(sent_emb, tag_emb) for sent_emb in sents]
                    s = sum(s_list)/len(s_list) if embedding_level == "sentence-avg" else max(s_list)
                    te_scores.append((tag, s))
                if te_scores:
                    vals=[s for _,s in te_scores]
                    te_avg=sum(vals)/len(vals)
                    te_max_tag,te_max=max(te_scores,key=lambda x:x[1])
                    k=max(1,min(topk_mean,len(vals)))
                    te_topk_mean=sum(sorted(vals,reverse=True)[:k])/k   

            if mode in ("tag-merged", "both"):
                vecs = [np.array(cache[tag.strip()]) for tag in tags if tag.strip() in cache]
                tm_len = len(vecs)
                
                if vecs:
                    tag_emb = np.ravel(np.mean(vecs, axis=0))
                    sents = [np.ravel(e) for e in book_emb]
                    s_list = [1 - cosine_distance(sent_emb, tag_emb) for sent_emb in sents]
                    tm_score = sum(s_list)/len(s_list) if embedding_level == "sentence-avg" else max(s_list)
                    tm_norm = tm_score / max(1, tm_len) if normalize_query and tm_len > 0 else tm_score
                else:
                    tm_score = tm_norm = 0.0

            rec={"title": title}
            if mode in ("tag-each","both"):
                rec["tag_each_scores"] = [{"tag": t, "cosine": round(s,6)} for t,s in te_scores]
            if mode in ("tag-merged","both"):
                rec.update({"tag_merged_score": round(tm_score,6), "tag_merged_len": tm_len})
            fout.write(json.dumps(rec, ensure_ascii=False)+"\n")

            row_out={"書名": title, "標籤數": len(tags)}
            if mode in ("tag-each","both"):
                row_out.update({
                    "TE_avg": round(te_avg,6),
                    "TE_max": round(te_max,6),
                    "TE_max_tag": te_max_tag,
                    f"TE_top{topk_mean}_mean": round(te_topk_mean,6)
                })
            if mode in ("tag-merged","both"):
                row_out.update({"TM_score":round(tm_score,6),"TM_len":tm_len,"TM_norm":round(tm_norm,6),"TM_norm_on":normalize_query})
            summary_rows.append(row_out)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_csv,index=False)
    print(f"✓ Wrote summary CSV: {out_csv}")
    print(f"✓ Wrote details JSONL: {details_path}")

