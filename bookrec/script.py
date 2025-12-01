# bookrec/script.py
import os, re, json, argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import time, datetime, math, inspect

from bookrec.embeddings import EmbeddingClient
from bookrec.core import VectorStore, SearchService
from data.code.add_tags import batch_generate_labels

from data.code.evaluation import evaluate_bm25, evaluate_cosine
from data.code.evaluation import tokenize, BM25, BM25Params

from scipy.spatial.distance import cosine as cosine_distance

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def merge_query_tokens(tags, tokenizer: str, ngram: int, merged_tf: str):
    """Aligns input behavior with evaluation.py: consolidates multiple terms into a single query token list."""
    all_tokens = []
    for tag in tags:
        all_tokens.extend(tokenize(tag, mode=tokenizer, ngram=ngram))
    if not all_tokens:
        return []
    tf = {}
    for t in all_tokens:
        tf[t] = tf.get(t, 0) + 1
    if merged_tf == "raw":
        return all_tokens
    if merged_tf == "binary":
        return list(tf.keys())
    if merged_tf == "log":
        out = []
        for t, f in tf.items():
            out.extend([t] * max(1, int(round(math.log(1 + f)))))
        return out
    return list(tf.keys())

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u3000\s]+", " ", text)
    return text.strip()

def main():
    p = argparse.ArgumentParser(description="BookRec Simple Tester + BM25 evaluation")

    # (Optional) Regenerate tags
    p.add_argument("--regen-tags", type=str, default="false", help="true/false: Whether to regenerate all tags.")
    p.add_argument("--gpt-model", type=str, default="gpt-4o", help="GPT model used for tag generation.")
    p.add_argument("--books-csv", type=str, default="data/process/processed_books.csv")
    p.add_argument("--regen-range", type=str, default=None, help="start:endExclusive. Use -1 for the end, e.g., 1100:-1.")
    p.add_argument("--save-every", type=int, default=500)

    # Tunable prompt parameters for tag generation
    p.add_argument("--system-prompt", type=str, default="Your main task is to provide corresponding subject tags based on the content of each book.",
                help="System prompt for tag generation (overrides default).")
    p.add_argument("--prompt-inline", type=str, default="", help="Directly provide a user prompt template string, supporting {title} {author} {summary}.")
    p.add_argument("--prompt-file", type=str, default="", help="Load user prompt template from file (UTF-8), supporting {title} {author} {summary}.")
    p.add_argument("--temperature", type=float, default=0.3, help="Temperature for tag generation.")

    p.add_argument("--eval-tags-bm25", dest="eval_tags_bm25", type=str, default="false",
                help="true/false: After generating tags, run BM25 evaluation using evaluation.py.")
    p.add_argument("--eval-out-csv-bm25", dest="eval_out_csv_bm25", type=str,
                default="data/process/bm25_eval_summary.csv")
    p.add_argument("--eval-out-jsonl-bm25", dest="eval_out_jsonl_bm25", type=str,
                default="data/process/bm25_eval_details.jsonl")
    
    p.add_argument("--eval-tags-cosine", dest="eval_tags_cosine", type=str, default="false",
               help="true/false: After generating tags, run cosine similarity evaluation using evaluation.py.")
    p.add_argument("--eval-out-csv-cosine", dest="eval_out_csv_cosine", type=str,
                default="data/process/cosine_eval_summary.csv")
    p.add_argument("--eval-out-jsonl-cosine", dest="eval_out_jsonl_cosine", type=str,
                default="data/process/cosine_eval_details.jsonl")

    # Excel and Vector Store
    p.add_argument("--excel", type=str, default="data/process/十類title_標籤.xlsx")
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    p.add_argument("--rebuild-index", type=str, default="false", help="true/false: Whether to rebuild the vector store index.")

    # Term-to-Books Search: Inputs
    p.add_argument("--terms", type=str, default="", help="Direct query input. Separate multiple queries with semicolon; terms within each query are comma-separated. E.g.: 'psychology,growth; investment,finance'")
    p.add_argument("--terms-file", type=str, default="", help="Text file, one query per line (terms separated by commas).")

    # Books-to-Books Search: Inputs
    p.add_argument("--books", type=str, default="", help="Source book titles, separated by semicolon. E.g.: 'The Courage to Be Disliked; Atomic Habits'")
    p.add_argument("--books-file", type=str, default="", help="Text file, one source book title per line.")

    # Retrieval Parameters (Single Set)
    p.add_argument("--sim-th", type=float, default=0.55)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--tag-topk", type=int, default=20)
    p.add_argument("--mode", type=str, default="soft_and")
    p.add_argument("--use-idf", type=str, default="false")
    p.add_argument("--cooccur-bonus", type=float, default=0.2)

    # (Optional) Tag Evaluation (BM25)
    p.add_argument("--bm25-mode", dest="bm25_mode", type=str, default="both",
                choices=["both","tag-each","tag-merged"])
    p.add_argument("--bm25-tokenizer", dest="tokenizer", type=str, default="auto",
                choices=["auto","jieba","whitespace","char"])
    p.add_argument("--bm25-ngram", dest="ngram", type=int, default=2)
    p.add_argument("--bm25-k1", dest="bm25_k1", type=float, default=1.2)
    p.add_argument("--bm25-b", dest="bm25_b", type=float, default=0.75)
    p.add_argument("--bm25-merged-tf", dest="merged_tf", type=str, default="binary",
                choices=["binary","log","raw"])
    p.add_argument("--bm25-normalize-query", dest="bm25_normalize_query", type=str, default="true",
                choices=["true","false"])
    p.add_argument("--bm25-topk-mean", dest="bm25_topk_mean", type=int, default=5)

    # Books-to-Books BM25 Settings
    p.add_argument("--bookq-mode-bm25", type=str, default="tags", choices=["tags","summary"],
                help="For Books-to-Books search, use source book's tags (merged) or summary for BM25 query.")
    p.add_argument("--bookq-merged-tf-bm25", type=str, default="binary", choices=["binary","log","raw"],
                help="Merging method for TF when bookq-mode=tags.")
    p.add_argument("--bookq-normalize-bm25", type=str, default="true", choices=["true","false"],
                help="Normalize length for Books-to-Books BM25 query.")

    # Books-to-Books Cosine Similarity Settings
    p.add_argument("--bookq-mode-cosine", type=str, default="tags", choices=["tags","summary"],
                help="For Books-to-Books search, use source book's tags (merged) or summary for cosine similarity query.")
    p.add_argument("--embedding-level-books", dest="embedding_level_books", type=str, default="sentence-max",
                choices=["sentence-max","sentence-avg"])
    p.add_argument("--bookq-normalize-cosine", type=str, default="true", choices=["true","false"],
                help="Normalize length for Books-to-Books cosine similarity query.")

    # Books-to-Books Diagnostic Metrics Settings
    p.add_argument("--bookq-title-sim", type=str, default="true", choices=["true","false"],
                help="Additionally calculate book title similarity for Books-to-Books results.")
    p.add_argument("--bookq-type-sim", type=str, default="true", choices=["true","false"],
                help="Additionally calculate category similarity for Books-to-Books results.")

    # (Optional) Score BM25 for 'Term-to-Books Search' (Now aggregated per-term)
    p.add_argument("--score-search-bm25", type=str, default="true",
                help="true/false: Calculate BM25 between each query term and the hit book's summary (aggregated per-term).")
    p.add_argument("--terms-bm25-agg-bm25", dest="terms_bm25_agg",
                type=str, default="avg", choices=["sum","avg","min"],
                help="BM25 aggregation for Term-to-Books: aggregate per-term scores using sum/avg/min.")
    
    # (Optional) Score Cosine Similarity for 'Term-to-Books Search'
    p.add_argument("--score-search-cosine", type=str, default="true",
                help="true/false: Calculate cosine similarity between each query term and the hit book's summary.")
    p.add_argument("--embedding-level-terms", dest="embedding_level_terms", type=str, default="sentence-max",
                choices=["sentence-max","sentence-avg"])

    # output
    p.add_argument("--out", type=str, default="results.jsonl")

    args = p.parse_args()
    as_bool = lambda s: str(s).lower() in {"1", "true", "yes", "y"}

    # ---- start time ----
    start_time = time.time()
    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- build results folder ----
    today_str = datetime.datetime.now().strftime("%Y%m%d")  # e.g. 20250917
    out_dir = f"experiment/{today_str}"
    os.makedirs(out_dir, exist_ok=True)

    # 0) (Optional) Tag generation
    prompt_src = "default"
    if as_bool(args.regen_tags):
        start, end = 0, -1
        if args.regen_range:
            m = re.match(r"^(-?\d+):(-?\d+)$", args.regen_range.strip())
            if not m:
                raise ValueError("--regen-range format error, e.g. 1100:-1")
            start, end = int(m.group(1)), int(m.group(2))

        prompt_tpl = None
        if args.prompt_inline:
            prompt_tpl = args.prompt_inline
            prompt_src = "inline"
        elif args.prompt_file:
            prompt_tpl = Path(args.prompt_file).read_text(encoding="utf-8")
            prompt_src = f"file:{args.prompt_file}"

        # Check signature dynamically to ensure compatibility with batch_generate_labels new parameters.
        call_kwargs = dict(
            books_csv=args.books_csv,
            output_file=args.excel,
            start=start,
            end=end,
            save_every=args.save_every,
            model=args.gpt_model,
        )
        sig = inspect.signature(batch_generate_labels)
        if "prompt_tpl" in sig.parameters:
            call_kwargs["prompt_tpl"] = prompt_tpl
        if "system_prompt" in sig.parameters:
            call_kwargs["system_prompt"] = args.system_prompt
        if "temperature" in sig.parameters:
            call_kwargs["temperature"] = args.temperature

        batch_generate_labels(**call_kwargs)
        print("Successfully generate tags")

    # 1) (Optional) Run BM25 evaluation on tags
    if as_bool(args.eval_tags_bm25):
        evaluate_bm25(
            excel=args.excel,
            books_csv=args.books_csv,
            out_csv=args.eval_out_csv_bm25,
            out_jsonl=args.eval_out_jsonl_bm25,
            mode=args.bm25_mode,
            tokenizer=args.tokenizer,
            ngram=args.ngram,
            k1=args.bm25_k1,
            b=args.bm25_b,
            merged_tf=args.merged_tf,
            normalize_query=as_bool(args.bm25_normalize_query),
            topk_mean=args.bm25_topk_mean,
        )
        print(f"Tag BM25 Evaluation Completed -> {args.eval_out_csv_bm25}")

    # 1b) (Optional) Run cosine similarity evaluation on tags
    if as_bool(args.eval_tags_cosine):
        evaluate_cosine(
            excel=args.excel,
            books_csv=args.books_csv,
            out_csv=args.eval_out_csv_cosine,
            out_jsonl=args.eval_out_jsonl_cosine,
            mode=args.cosine_mode,
            embedding_level=args.embedding_level,
            normalize_query=as_bool(args.cosine_normalize_query),
            topk_mean=args.cosine_topk_mean,
        )
        print(f"Tag Cosine Similarity Evaluation Completed ->  {args.eval_out_csv_cosine}")

    # 2)  Build/Load Vector Store (used for Term-to-Books / Books-to-Books)
    embedder = EmbeddingClient(model=args.embedding_model)
    store = VectorStore(excel_path=args.excel)
    if as_bool(args.rebuild_index):
        store.build(rebuild=True, embedder=embedder)
    else:
        store.load()

    # 3)Read queries (Term-to-Books)
    queries = []
    if args.terms:
        for part in args.terms.split(";"):
            q = part.strip()
            if q:
                queries.append(q)
    if args.terms_file:
        with open(args.terms_file, encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    queries.append(q)
    if not queries:
        queries = ["growth"]

    # 3b) Read source books (Books-to-Books)
    src_books = []
    if args.books:
        src_books = [s.strip() for s in re.split(r"[;\uFF1B]+", args.books) if s.strip()]
    if args.books_file:
        with open(args.books_file, encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    src_books.append(t)

    # 4) Build BM25 Corpus (full summary corpus) & cosine similarity
    books_df = pd.read_csv(args.books_csv).drop_duplicates(subset=["title"]).reset_index(drop=True)
    docs = []
    title_to_docid = {}
    title_to_summary = {}
    for _, r in books_df.iterrows():
        title = str(r.get("title", "")).strip()
        summary = str(r.get("summary", "")).strip()
        tokens = tokenize(summary, mode=args.tokenizer, ngram=args.ngram)
        title_to_docid[title] = len(docs)
        title_to_summary[title] = summary
        docs.append(tokens)
    bm25 = BM25(docs, BM25Params(k1=args.bm25_k1, b=args.bm25_b))

    # 5) Term-to-Books search + BM25 scoring (per-term aggregation)
    svc = SearchService(store, embedder)
    out_rows = []

    terms_csv_rows = []  
    for q in queries:
        df = svc.terms_to_books(
            query_input=q,
            tag_topk=args.tag_topk,
            sim_th=args.sim_th,
            topk=args.topk,
            use_idf=as_bool(args.use_idf),
            mode=args.mode,
            cooccur_bonus=args.cooccur_bonus,
        )

        terms_split = [s.strip() for s in re.split(r"[,，]+", q) if s.strip()]
        #  BM25 scoring per-term aggregation（sum/avg/min）& cosine similarity
        bm25_scores = []
        per_term_list = []
        q_emb = np.mean(embedder.embed_texts(terms_split), axis=0)
        cosine_scores = []
        for _, r in df.iterrows():
            did = title_to_docid.get(r["title"])
            # BM25
            if did is None:
                bm25_scores.append(0.0)
                per_term_list.append([])
                continue
            per_term_scores = []
            for t in terms_split:
                t_tokens = tokenize(t, mode=args.tokenizer, ngram=args.ngram)
                per_term_scores.append(float(bm25.score(t_tokens, did)))
            if args.terms_bm25_agg == "sum":
                agg_s = sum(per_term_scores)
            elif args.terms_bm25_agg == "min":
                agg_s = min(per_term_scores) if per_term_scores else 0.0
            else:  # avg
                agg_s = sum(per_term_scores) / max(1, len(per_term_scores))
            bm25_scores.append(round(agg_s, 6))
            per_term_list.append(per_term_scores)

            # cosine similarity
            sentences = title_to_summary.get(r["title"], "").split("。")
            sentences = [clean_text(d) for d in sentences]
            sentences = [d for d in sentences if d]
            book_emb = embedder.embed_texts(sentences)

            book_emb = [np.ravel(e) for e in book_emb]
            q_emb = np.ravel(q_emb)         
            sims = [1 - cosine_distance(sent_emb, q_emb) for sent_emb in book_emb]
            if args.embedding_level_terms == "sentence-max":
                cosine_scores.append(max(sims))
            else:
                cosine_scores.append(sum(sims) / max(1, len(sims)))


        df = df.assign(
            bm25_score=[round(s,6) for s in bm25_scores],
            cosine_score=[round(s,6) for s in cosine_scores],
        )

        print(f"\n=== Terms Query: {q} (agg={args.terms_bm25_agg}) ===")
        show_cols = ["title", "type", "score", "bm25_score", "cosine_score"]
        print(df[show_cols].head(args.topk))

        # Accumulate and consolidate results table (rank normalization applied).
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            terms_csv_rows.append({
                "query": q,
                "rank": rank,
                "title": row["title"],
                "type": row["type"],
                "score": row["score"],
                "bm25_score": row["bm25_score"],
                "cosine_score": row["cosine_score"],
                "exact_hits": row.get("exact_hits", ""),
                "similar_hits": row.get("similar_hits", "")
            })

        out_rows.append({"type": "terms_to_books", "query": q, "results": df.to_dict(orient="records")})

    # 6) books to books + BM25
    books_csv_rows = [] 
    if src_books:
        # Source book query tokens: depends on 'bookq-mode' setting
        excel_df = pd.read_excel(args.excel).drop_duplicates(subset=["title"]).reset_index(drop=True)
        title_to_tags = {
            str(r["title"]).strip(): [s.strip() for s in str(r["tags"]).split(",") if s and s.strip()]
            for _, r in excel_df.iterrows()
        }

        for titles in src_books:
            df = svc.books_to_books(
                titles_input=titles,
                tag_topk=args.tag_topk,
                sim_th=args.sim_th,
                topk=args.topk,
                mode=args.mode,
                cooccur_bonus=args.cooccur_bonus,
            )

            title_split = [s.strip() for s in re.split(r"[;；]+", titles) if s.strip()]
            for title in title_split:
                # BM25 Query: merged tags OR summary
                if args.bookq_mode_bm25 == "tags":
                    tags = title_to_tags.get(title, [])
                    q_tokens = merge_query_tokens(tags, tokenizer=args.tokenizer,
                                                ngram=args.ngram, merged_tf=args.bookq_merged_tf_bm25)
                else:  # summary
                    src_row = books_df[books_df["title"] == title]
                    src_summary = str(src_row.iloc[0]["summary"]) if not src_row.empty else ""
                    q_tokens = tokenize(src_summary, mode=args.tokenizer, ngram=args.ngram)

                # Cosine Similarity Query: merged tags OR summary
                if args.bookq_mode_cosine == "tags":
                    tags = title_to_tags.get(title, [])
                else:  # summary
                    src_row = books_df[books_df["title"] == title]
                    src_summary = str(src_row.iloc[0]["summary"]) if not src_row.empty else ""
                
                # diagnostic：title
                if as_bool(args.bookq_title_sim):
                    title_clean = embedder.embed_text(clean_text(title))
                    title_clean = np.ravel(title_clean) if title_clean is not None else None

                    emb = pd.DataFrame()
                    emb["title_emb"] = df["title"].apply(lambda t: np.ravel(embedder.embed_text(clean_text(str(t)))))
                    if title_clean is not None:
                        sim_list_title = [1 - cosine_distance(r_emb, title_clean) for r_emb in emb["title_emb"]]
                        if "title_sim" not in df.columns:
                            df["title_sim"] = 0.0
                        df["title_sim"] = df["title_sim"] + sim_list_title

                # diagnostic: type
                if as_bool(args.bookq_type_sim):
                    src_row = books_df[books_df["title"] == title]
                    src_type = str(src_row.iloc[0]["type"]) if not src_row.empty else ""

                    src_type_emb = embedder.embed_text(src_type)
                    src_type_emb = np.ravel(src_type_emb) if src_type_emb is not None else None

                    emb = pd.DataFrame()
                    emb["type_emb"] = df["type"].apply(lambda t: np.ravel(embedder.embed_text(str(t))))
                    if src_type_emb is not None:
                        sim_list_type =[1 - cosine_distance(r_emb, src_type_emb) for r_emb in emb["type_emb"]]
                        if "type_sim" not in df.columns:
                            df["type_sim"] = 0.0
                        df["type_sim"] = df["type_sim"] + sim_list_type

                # score
                scores_bm = []
                scores_cosine = []
                norms_bm = []
                norms_cosine = []
                q_len = len(q_tokens)
                for _, r in df.iterrows():
                    did = title_to_docid.get(r["title"])
                    res_summary = title_to_summary.get(r["title"])
                    # BM25
                    s = float(bm25.score(q_tokens, did)) if did is not None and q_len > 0 else 0.0
                    scores_bm.append(round(s, 6))
                    if as_bool(args.bookq_normalize_bm25):
                        norms_bm.append(round(s / max(1, q_len), 6))

                    # cosine similarity
                    res_sentences = res_summary.split("。")
                    res_sentences = [clean_text(d) for d in res_sentences]
                    res_sentences = [d for d in res_sentences if d]
                    res_book_emb = embedder.embed_texts(res_sentences)

                    if args.bookq_mode_cosine == "tags":
                        q_emb = embedder.embed_texts([",".join(tags)])[0] if tags else None
                    else:
                        q_emb = embedder.embed_texts([src_summary])[0] if src_summary else None

                    if res_book_emb is not None and len(res_book_emb) > 0 and q_emb is not None:
                        res_book_emb = [np.ravel(e) for e in res_book_emb]
                        q_emb = np.ravel(q_emb)
                        sims = [1 - cosine_distance(sent_emb, q_emb) for sent_emb in res_book_emb]
                        if args.embedding_level_books == "sentence-max":
                            score = round(max(sims),6)
                            scores_cosine.append(score)
                        else:
                            score = round(sum(sims)/len(sims),6)
                            scores_cosine.append(score)

                        if as_bool(args.bookq_normalize_cosine):
                            norms_cosine.append(round(score / len(sims),6))

                if "bm25_score" not in df.columns:
                    df["bm25_score"] = 0.0
                df["bm25_score"] = df["bm25_score"] + scores_bm
                if "cosine_score" not in df.columns:
                    df["cosine_score"] = 0.0
                df["cosine_score"] = df["cosine_score"] + scores_cosine

                if as_bool(args.bookq_normalize_bm25):
                    if "bm25_norm" not in df.columns:
                        df["bm25_norm"] = 0.0
                    df["bm25_norm"] = df["bm25_norm"] + norms_bm
                if as_bool(args.bookq_normalize_cosine):
                    if "cosine_norm" not in df.columns:
                        df["cosine_norm"] = 0.0
                    df["cosine_norm"] = df["cosine_norm"] + norms_cosine
            
            df["bm25_score"] = round(df["bm25_score"] / len(title_split),4)
            df["cosine_score"] = round(df["cosine_score"] / len(title_split),4)
            
            print(f"\n=== Books Query: {titles} (bm25: {args.bookq_mode_bm25})(cosine: {args.bookq_mode_cosine}) ===")
            show_cols = ["title", "type", "score", "bm25_score", "cosine_score"]
            if as_bool(args.bookq_normalize_bm25):
                df["bm25_norm"] = round(df["bm25_norm"] / len(title_split),4)
                show_cols.append("bm25_norm")
            if as_bool(args.bookq_normalize_cosine):
                df["cosine_norm"] = round(df["cosine_norm"] / len(title_split),4)
                show_cols.append("cosine_norm")
            if as_bool(args.bookq_title_sim):
                df["title_sim"] = round(df["title_sim"] / len(title_split),4)
                show_cols.append("title_sim")
            if as_bool(args.bookq_type_sim):
                df["type_sim"] = round(df["type_sim"] / len(title_split),4)
                show_cols.append("type_sim")
            print(df[show_cols].head(args.topk))

            # Accumulate and consolidate results table (rank normalization applied).
            for rank, (_, row) in enumerate(df.iterrows(), start=1):
                rec = {
                    "source_title": titles,
                    "bookq_mode_bm25": args.bookq_mode_bm25,
                    "bookq_mode_cosine": args.bookq_mode_cosine,
                    "rank": rank,
                    "title": row["title"],
                    "分類": row["分類"],
                    "score": row["score"],
                    "source_hits": row.get("source_hits", ""),
                    "match_count": row.get("match_count", ""),
                    "bm25_score": row["bm25_score"],
                    "cosine_score": row["cosine_score"],
                }
                if "bm25_norm" in row:
                    rec["bm25_norm"] = row["bm25_norm"]
                if "cosine_norm" in row:
                    rec["cosine_norm"] = row["cosine_norm"]
                if "title_sim" in row:
                    rec["title_sim"] = row["title_sim"]
                if "type_sim" in row:
                    rec["type_sim"] = row["type_sim"]
                books_csv_rows.append(rec)

            out_rows.append({
                "type": "books_to_books",
                "source_title": titles,
                "bookq_mode_bm25": args.bookq_mode_bm25,
                "bookq_mode_cosine": args.bookq_mode_cosine,
                "results": df.to_dict(orient="records"),
            })

    # 7) save result + meta
    out_path = Path(out_dir) / Path(args.out).name
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n✓ 結果已存 {out_path}")

    # 7.1 Additionally output two "all-in-one" summary tables
    terms_csv_path = out_path.with_name(out_path.stem + "_terms.csv")
    if terms_csv_rows:
        pd.DataFrame(terms_csv_rows).to_csv(terms_csv_path, index=False)
        print(f"Term-to-Bookss Consolidation: {terms_csv_path}")

    books_csv_path = out_path.with_name(out_path.stem + "_books.csv")
    if books_csv_rows:
        pd.DataFrame(books_csv_rows).to_csv(books_csv_path, index=False)
        print(f"Books-to-Books Consolidation: {books_csv_path}")

    end_time = time.time()
    end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = round(end_time - start_time, 2)

    meta = {
        "start_time": start_str,
        "end_time": end_str,
        "elapsed_sec": elapsed,
        "args": vars(args),
        "prompt_source": prompt_src,
        "bm25_eval_summary_csv": args.eval_out_csv_bm25 if as_bool(args.eval_tags_bm25) else "",
        "bm25_eval_details_jsonl": args.eval_out_jsonl_bm25 if as_bool(args.eval_tags_bm25) else "",
        "cosine_eval_summary_csv": args.eval_out_csv_bm25 if as_bool(args.eval_tags_bm25) else "",
        "cosine_eval_details_jsonl": args.eval_out_jsonl_bm25 if as_bool(args.eval_tags_bm25) else "",
        "terms_csv": str(terms_csv_path) if terms_csv_rows else "",
        "books_csv": str(books_csv_path) if books_csv_rows else "",
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Meta info saved to {meta_path} (Elapsed time: {elapsed} seconds)")

if __name__ == "__main__":
    main()