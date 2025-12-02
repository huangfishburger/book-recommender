# bookrec/core.py
import os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import faiss

# ─────────────────────────────────────────────────────────
# path（Default：repo/data/process and artifacts）
# To override, set BOOKREC_ROOT to your repository's root directory
# ─────────────────────────────────────────────────────────
def _repo_root() -> Path:
    env = os.getenv("BOOKREC_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[1]

REPO = _repo_root()
DATA = REPO / "data" / "process"
ART  = DATA / "artifacts"
EXCEL_DEFAULT   = DATA / "books_tags.xlsx"
TAG_CACHE_JSON  = ART / "tag_cache.json"
TAG_VECS_NPY    = ART / "tag_vecs.npy"
FAISS_INDEX     = ART / "tag_index.faiss"

def _ensure_dirs():
    ART.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Helper: Read Excel + generate 'tags' column
# ─────────────────────────────────────────────────────────
def str_to_list(s):
    if isinstance(s, str):
        return [x.strip() for x in s.split(",") if x.strip()]
    return []

def _load_books_with_tags(excel_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, engine="openpyxl")
    df["tags"] = df['tags'].apply(str_to_list)
    return df[["title", "category", "tags"]]

# ─────────────────────────────────────────────────────────
# VectorStore: Build/Load FAISS + Inverted Index + IDF
# ─────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, excel_path: str | Path = EXCEL_DEFAULT):
        _ensure_dirs()
        self.excel_path = Path(excel_path)
        self.df: pd.DataFrame | None = None
        self.all_tags: List[str] = []
        self.tag2id: Dict[str, int] = {}
        self.id2tag: List[str] = []
        self.tag_vecs: np.ndarray | None = None      # L2-normalized
        self.index: faiss.IndexFlatIP | None = None  # cosine by inner product
        self.tag2books: Dict[str, List[int]] = {}
        self.idf: Dict[str, float] = {}

    # ---- offline build (requires an embedder object providing embed_texts(list)->np.ndarray) ----
    def build(self, rebuild: bool, embedder) -> "VectorStore":
        print(f"load Excel: {self.excel_path}")
        df = _load_books_with_tags(self.excel_path)
        self.df = df
        tags = sorted({t for row in df["tags"] for t in row})
        print(f"total tags count: {len(tags)}")
        self.all_tags = tags
        self.tag2id = {t: i for i, t in enumerate(tags)}
        self.id2tag = tags

        cache = {}
        if TAG_CACHE_JSON.exists() and not rebuild:
            cache = json.loads(TAG_CACHE_JSON.read_text(encoding="utf-8"))

        missing = tags if rebuild else [t for t in tags if t not in cache]
        print(f"lack of embedded: {len(missing)}（rebuild={rebuild}）")
        if missing:
            vecs = embedder.embed_texts(missing, batch_size=100, progress=True)
            for t, v in zip(missing, vecs):
                cache[t] = v.tolist()
            TAG_CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        vec_arr = np.array([cache[t] for t in tags], dtype=np.float32)
        vec_arr = vec_arr / (np.linalg.norm(vec_arr, axis=1, keepdims=True) + 1e-9)

        np.save(str(TAG_VECS_NPY), vec_arr)
        print(f"Vector matrix successfully written:{TAG_VECS_NPY}")

        self.tag_vecs = vec_arr
        self._init_faiss()
        faiss.write_index(self.index, str(FAISS_INDEX))
        print(f"FAISS index successfully written:{FAISS_INDEX}")

        self._build_inverted_and_idf()
        print("Index built successfully!")
        return self

    # ---- runtime load（do not need embedder）----
    def load(self) -> "VectorStore":
        df = _load_books_with_tags(self.excel_path)
        self.df = df
        tags = sorted({t for row in df["tags"] for t in row})
        self.all_tags = tags
        self.tag2id = {t: i for i, t in enumerate(tags)}
        self.id2tag = tags

        if not TAG_VECS_NPY.exists():
            raise RuntimeError("Vector file not found. Please run build() first.")

        vec_arr = np.load(TAG_VECS_NPY)
        if vec_arr.shape[0] != len(tags):
            raise RuntimeError("Tag count does not match the number of stored vectors. Please rebuild: build(rebuild=True)")

        # L2-normalized
        vec_arr = vec_arr / (np.linalg.norm(vec_arr, axis=1, keepdims=True) + 1e-9)
        self.tag_vecs = vec_arr
        self._init_faiss()
        self._build_inverted_and_idf()
        return self

    # ---- internal ----
    def _init_faiss(self):
        d = self.tag_vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.tag_vecs)

    def _build_inverted_and_idf(self):
        t2b = defaultdict(list)
        for i, tags in enumerate(self.df["tags"]):
            for t in tags:
                t2b[t].append(i)
        self.tag2books = t2b
        freq = Counter(t for tags in self.df["tags"] for t in tags)
        N = len(self.df)
        self.idf = {t: float(np.log(N / (1 + freq[t]))) for t in freq}
        
    # ---- utilities ----
    def similar_tags_by_vec(self, q_vec: np.ndarray, topk=20, sim_th=0.55) -> List[Tuple[str, float]]:
        sim, idx = self.index.search(q_vec[None, :], topk)
        return [(self.id2tag[i], float(s)) for i, s in zip(idx[0], sim[0]) if s >= sim_th]

    def tags_for_book(self, title: str) -> List[str]:
        m = self.df[self.df["title"] == title]
        if m.empty:
            return []
        return sorted({t for row in m["tags"] for t in row})

# ─────────────────────────────────────────────────────────
# SearchService：term→books / books→books / term→keywords / book→keywords
# (term requires 'real-time embedding' → delegated to external embedder.embed_text)
# ─────────────────────────────────────────────────────────
class SearchService:
    def __init__(self, store: VectorStore, embedder = None):
        self.store = store
        self.embedder = embedder
    
    # --- helper: Comma tokenization (supports full-width/half-width), remove whitespace, deduplicate while preserving order ---
    @staticmethod
    def _parse_terms(query_input: str):
        parts = re.split(r"[,\uFF0C]+", str(query_input))
        terms = [p.strip() for p in parts if p and p.strip()]
        seen, out = set(), []
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
    
    @staticmethod
    def _parse_titles(titles_input: str) -> list[str]:
        """
        Multiple book titles are separated by a semicolon: ; or full-width ；
        Commas (,，) are treated as part of the book title content and are not used for splitting.
        Performs NFKC normalization and trims leading/trailing whitespace; deduplicates while preserving order.
        """
        parts = re.split(r"[;\uFF1B]+", str(titles_input))
        seen, out = set(), []
        for p in parts:
            t = p.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def terms_to_books(self,
                    query_input: str,
                    tag_topk: int = 20,
                    sim_th: float = 0.55,
                    topk: int = 10,
                    use_idf: bool = False,
                    mode: str | None = None,
                    cooccur_bonus: float = 0.2) -> pd.DataFrame:
        if mode is None:
            mode = "soft_and"

        terms = self._parse_terms(query_input)
        if not terms:
            return pd.DataFrame(
                columns=["title", "category", "score", "exact_hits", "similar_hits"]
            )

        per_term_scores   = {}  # term -> {book_idx: score}
        per_term_exact    = {}  # term -> {book_idx: exact match count}
        per_term_similar  = {}  # term -> {book_idx: similar match count}

        for term in terms:
            # take term vector
            if term in self.store.tag2id:
                q_vec = self.store.tag_vecs[self.store.tag2id[term]]
            else:
                q_vec = self.embedder.embed_text(term)

            matched = self.store.similar_tags_by_vec(q_vec, topk=tag_topk, sim_th=sim_th)

            contrib_score   = defaultdict(float)
            contrib_exact   = defaultdict(int)
            contrib_similar = defaultdict(int)

            for tag, s in matched:
                w = s * (self.store.idf.get(tag, 1.0) if use_idf else 1.0)
                for bi in set(self.store.tag2books.get(tag, [])):
                    contrib_score[bi] += w
                    if tag == term:
                        contrib_exact[bi] += 1
                    else:
                        contrib_similar[bi] += 1

            per_term_scores[term]  = contrib_score
            per_term_exact[term]   = contrib_exact
            per_term_similar[term] = contrib_similar

        all_books = set().union(*(d.keys() for d in per_term_scores.values()))
        if not all_books:
            return pd.DataFrame(
                columns=["title", "category", "score", "exact_hits", "similar_hits"]
            )

        final_scores       = {}
        exact_hits_count   = {}
        similar_hits_count = {}

        for bi in all_books:
            # Aggregate scores and hits for each term
            scores_by_term = [per_term_scores[t].get(bi, 0.0) for t in terms]
            total_exact    = sum(per_term_exact[t].get(bi, 0) for t in terms)
            total_similar = sum(per_term_similar[t].get(bi, 0) for t in terms)

            exact_hits_count[bi]   = total_exact
            similar_hits_count[bi] = total_similar

            # calculate final score
            if mode == "sum":
                final = sum(scores_by_term)
            elif mode == "avg":
                final = sum(scores_by_term) / max(1, len(terms))
            elif mode == "min":
                final = min(scores_by_term)
            elif mode == "soft_and":
                # cooccur_bonus is still calculated as 'term hit count - 1'
                term_hits = sum(1 for v in scores_by_term if v > 0)
                final = sum(scores_by_term) + cooccur_bonus * max(0, term_hits - 1)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            final_scores[bi] = final

        # topk
        top = sorted(final_scores.items(), key=lambda x: -x[1])[:topk]
        df = self.store.df

        out = (
            df.loc[[i for i, _ in top], ["title", "category", "tags"]]
            .assign(
                score=[round(final_scores[i], 3) for i, _ in top],
                exact_hits=[exact_hits_count[i] for i, _ in top],
                similar_hits=[similar_hits_count[i] for i, _ in top],
            )
            .reset_index(drop=True)
        )
        return out

    # books → books
    def books_to_books(self, titles_input: str, tag_topk: int = 20, sim_th: float = 0.70, topk: int = 10, 
                      mode: str | None = None, cooccur_bonus: float = 0.2 ) -> pd.DataFrame:
        if mode is None:
            mode = "soft_and"

        src_titles = self._parse_titles(titles_input)
        # Empty results should include new columns
        if not src_titles:
            return pd.DataFrame(columns=["title", "category", "match_count", "score", "source_hits"])

        df = self.store.df

        # Collect candidate scores for each source book
        per_src_counts: dict[str, dict[int, int]]   = {}
        per_src_scores: dict[str, dict[int, float]] = {}
        per_src_books:  dict[str, set[int]]         = {}

        # Used to exclude the source book itself
        src_indices = set(df.index[df["書名"].isin(src_titles)].tolist())

        for title in src_titles:
            src_tags = self.store.tags_for_book(title)
            if not src_tags:
                per_src_counts[title] = {}
                per_src_scores[title] = {}
                per_src_books[title]  = set()
                continue

            contrib_c = defaultdict(int)
            contrib_s = defaultdict(float)

            for t in src_tags:
                t_id = self.store.tag2id.get(t)
                if t_id is None:
                    continue
                t_vec = self.store.tag_vecs[t_id]
                matched = self.store.similar_tags_by_vec(t_vec, topk=tag_topk, sim_th=sim_th)

                for tag, s in matched:
                    for bi in set(self.store.tag2books.get(tag, [])):
                        contrib_c[bi] += 1
                        contrib_s[bi] += s

            # Exclude the source book's own index
            for idx in list(contrib_c.keys()):
                if idx in src_indices:
                    contrib_c.pop(idx, None)
                    contrib_s.pop(idx, None)

            per_src_counts[title] = dict(contrib_c)
            per_src_scores[title] = dict(contrib_s)
            per_src_books[title]  = set(contrib_c.keys()) | set(contrib_s.keys())

        # Merge candidates from all sources
        all_books = set().union(*per_src_books.values()) if per_src_books else set()
        if not all_books:
            return pd.DataFrame(columns=["title", "category", "match_count", "score", "source_hits"])

        final_count: dict[int, float] = {}
        final_score: dict[int, float] = {}
        source_hits: dict[int, int] = {}

        for bi in all_books:
            counts_by_src = [per_src_counts[t].get(bi, 0)    for t in src_titles]
            scores_by_src = [per_src_scores[t].get(bi, 0.0)  for t in src_titles]
            # source_hits: The number of unique source books that matched at least once
            k = sum(1 for v in scores_by_src if v > 0.0)
            source_hits[bi] = k

            # match_count combination: sum/avg/min/soft_and
            if mode == "sum":
                c_final = sum(counts_by_src)
                s_final = sum(scores_by_src)
            elif mode == "avg":
                c_final = sum(counts_by_src) / max(1, len(src_titles))
                s_final = sum(scores_by_src) / max(1, len(src_titles))
            elif mode == "min":
                c_final = min(counts_by_src)
                s_final = min(scores_by_src)
            elif mode == "soft_and":
                c_final = sum(counts_by_src)
                s_final = sum(scores_by_src) + cooccur_bonus * max(0, k - 1)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            final_count[bi] = c_final
            final_score[bi] = s_final

        if not final_score:
            return pd.DataFrame(columns=["title", "category", "match_count", "score", "source_hits"])

        # order: Prioritize by match_count, then by score
        cand = sorted(final_score.keys(), key=lambda i: (-final_count[i], -final_score[i]))[:topk]

        out = (
            df.loc[cand, ["title", "category"]]
              .assign(
                  score=[round(float(final_score[i]), 3) for i in cand],
                  source_hits=[source_hits[i] for i in cand],
                  match_count=[int(final_count[i]) for i in cand],
              )
              .reset_index(drop=True)
        )
        return out


    # term → keywords
    def term_to_keywords(self, query: str, k: int = 10, sim_th: float = 0.60) -> pd.DataFrame:
        # 1) Get query vector (same as above)
        if query in self.store.tag2id:
            q_vec = self.store.tag_vecs[self.store.tag2id[query]]
            self_id = self.store.tag2id[query]
        else:
            q_vec = self.embedder.embed_text(query)
            self_id = None

        # 2) Use FAISS to retrieve a larger candidate pool (automatically configured)
        candidates_k = max(50, k * 5)
        matched = self.store.similar_tags_by_vec(q_vec, topk=candidates_k, sim_th=sim_th)

        # 3) Exclude self, take top k
        out = []
        for tag, s in matched:
            if self_id is not None and tag == self.store.id2tag[self_id]:
                continue
            out.append((tag, s))
            if len(out) >= k:
                break

        if not out:
            return pd.DataFrame(columns=["keyword", "similarity"])
        kws, sims = zip(*out)
        return pd.DataFrame({"keyword": kws, "similarity": [round(float(x), 4) for x in sims]})
    
    # book → keywords
    def book_to_keywords(self, title: str, k: int = 10) -> List[str]:
        # 1) Retrieve the corresponding tags for the book
        tags = self.store.tags_for_book(title)

        # 2) select top k
        result_tags = tags[:k]
        return result_tags
