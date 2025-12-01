# bookrec/script.py
import os, re, json, argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import time, datetime, math, inspect

from bookrec.embeddings import EmbeddingClient
from bookrec.core import VectorStore, SearchService
from data.code.add_tags import batch_generate_labels

# 從 evaluation 重用 BM25 工具
from data.code.evaluation import evaluate_bm25, evaluate_cosine
from data.code.evaluation import tokenize, BM25, BM25Params

from scipy.spatial.distance import cosine as cosine_distance

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 降低 OMP 衝突風險
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def merge_query_tokens(tags, tokenizer: str, ngram: int, merged_tf: str):
    """與 evaluation.py 的行為一致：把多個詞合併成查詢 token 列表"""
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

    # （選用）重生標籤
    p.add_argument("--regen-tags", type=str, default="false", help="true/false 是否重跑標籤生成")
    p.add_argument("--gpt-model", type=str, default="gpt-4o", help="生成標籤時使用的 GPT 模型")
    p.add_argument("--books-csv", type=str, default="data/process/processed_books.csv")
    p.add_argument("--regen-range", type=str, default=None, help="start:endExclusive，-1 表示最後，如 1100:-1")
    p.add_argument("--save-every", type=int, default=500)

    # 標籤生成的可調 prompt 參數
    p.add_argument("--system-prompt", type=str, default="你的主要工作是根據每本書的內容，提供對應的主題詞。",
                   help="生成標籤時的 system prompt（可覆寫預設）")
    p.add_argument("--prompt-inline", type=str, default="", help="直接提供 user prompt 模板字串，支援 {title} {author} {summary}")
    p.add_argument("--prompt-file", type=str, default="", help="從檔案載入 user prompt 模板（UTF-8），支援 {title} {author} {summary}")
    p.add_argument("--temperature", type=float, default=0.3, help="生成標籤時的 temperature")
    
    p.add_argument("--eval-tags-bm25", dest="eval_tags_bm25", type=str, default="false",
               help="true/false：完成標籤檔後，跑 evaluation.py 的 BM25 評估")
    p.add_argument("--eval-out-csv-bm25", dest="eval_out_csv_bm25", type=str,
                default="data/process/bm25_eval_summary.csv")
    p.add_argument("--eval-out-jsonl-bm25", dest="eval_out_jsonl_bm25", type=str,
                default="data/process/bm25_eval_details.jsonl")
    
    p.add_argument("--eval-tags-cosine", dest="eval_tags_cosine", type=str, default="false",
               help="true/false：完成標籤檔後，跑 evaluation.py 的 cosine similarity 評估")
    p.add_argument("--eval-out-csv-cosine", dest="eval_out_csv_cosine", type=str,
                default="data/process/cosine_eval_summary.csv")
    p.add_argument("--eval-out-jsonl-cosine", dest="eval_out_jsonl_cosine", type=str,
                default="data/process/cosine_eval_details.jsonl")

    # Excel 與向量庫
    p.add_argument("--excel", type=str, default="data/process/十類書名_標籤.xlsx")
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    p.add_argument("--rebuild-index", type=str, default="false", help="true/false 是否重建向量庫")

    # 詞找書：輸入
    p.add_argument("--terms", type=str, default="", help="直接輸入查詢，多個查詢用分號分隔；每個查詢內用逗號分詞。例如：'心理學,成長; 投資,理財'")
    p.add_argument("--terms-file", type=str, default="", help="txt 檔，每行一個查詢 terms（用逗號分詞）")

    # 書搜書：輸入
    p.add_argument("--books", type=str, default="", help="來源書名，分號分隔。例：'被討厭的勇氣; 原子習慣'")
    p.add_argument("--books-file", type=str, default="", help="txt 檔，每行一個來源書名")

    # 檢索參數（單一組）
    p.add_argument("--sim-th", type=float, default=0.55)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--tag-topk", type=int, default=20)
    p.add_argument("--mode", type=str, default="soft_and")
    p.add_argument("--use-idf", type=str, default="false")
    p.add_argument("--cooccur-bonus", type=float, default=0.2)

    # （選用）標籤評估（BM25）
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

    # （選用）標籤評估（cosine similarity）
    p.add_argument("--cosine-mode", dest="cosine_mode", type=str, default="both",
                   choices=["both","tag-each","tag-merged"])
    p.add_argument("--embedding-level", dest="embedding_level", type=str, default="sentence-max",
                    choices=["sentence-max","sentence-avg"])
    p.add_argument("--cosine-normalize-query", dest="cosine_normalize_query", type=str, default="true",
                   choices=["true","false"])
    p.add_argument("--cosine-topk-mean", dest="cosine_topk_mean", type=int, default=5)

    # 書搜書 BM25 設定
    p.add_argument("--bookq-mode-bm25", type=str, default="tags", choices=["tags","summary"],
                   help="書搜書時 BM25 查詢使用來源書的 tags（合併）或 summary")
    p.add_argument("--bookq-merged-tf-bm25", type=str, default="binary", choices=["binary","log","raw"],
                   help="當 bookq-mode=tags 時，合併 TF 的方式")
    p.add_argument("--bookq-normalize-bm25", type=str, default="true", choices=["true","false"],
                   help="是否對書搜書的 BM25 查詢做長度正規化")
    
    # 書搜書 cosine similarity 設定
    p.add_argument("--bookq-mode-cosine", type=str, default="tags", choices=["tags","summary"],
                   help="書搜書時 cosine similarity 查詢使用來源書的 tags（合併）或 summary")
    p.add_argument("--embedding-level-books", dest="embedding_level_books", type=str, default="sentence-max",
                    choices=["sentence-max","sentence-avg"])
    p.add_argument("--bookq-normalize-cosine", type=str, default="true", choices=["true","false"],
                   help="是否對書搜書的 cosine similarity 查詢做長度正規化")
    
    # 書搜書 Diagnostic 指標設定
    p.add_argument("--bookq-title-sim", type=str, default="true", choices=["true","false"],
                   help="是否對書搜書的結果額外計算書名 similarity")
    p.add_argument("--bookq-type-sim", type=str, default="true", choices=["true","false"],
                   help="是否對書搜書的結果額外計算類別 similarity")

    # （選用）對「詞找書結果」加算 BM25（現在改為 per-term 聚合）
    p.add_argument("--score-search-bm25", type=str, default="true",
                   help="true/false：對每個查詢與命中書籍簡介計算 BM25（per-term 聚合）")
    p.add_argument("--terms-bm25-agg-bm25", dest="terms_bm25_agg",
                   type=str, default="avg", choices=["sum","avg","min"],
                   help="詞找書的 BM25 聚合：對每個 term 各自打分後，做 sum/avg/min")
    
    # （選用）對「詞找書結果」加算 cosine similarity
    p.add_argument("--score-search-cosine", type=str, default="true",
                   help="true/false：對每個查詢與命中書籍簡介計算 cosine similarity")
    p.add_argument("--embedding-level-terms", dest="embedding_level_terms", type=str, default="sentence-max",
                    choices=["sentence-max","sentence-avg"])

    # 輸出
    p.add_argument("--out", type=str, default="results.jsonl")

    args = p.parse_args()
    as_bool = lambda s: str(s).lower() in {"1", "true", "yes", "y"}

    # ---- 開始時間 ----
    start_time = time.time()
    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 建立結果資料夾 ----
    today_str = datetime.datetime.now().strftime("%Y%m%d")  # e.g. 20250917
    out_dir = f"experiment/{today_str}"
    os.makedirs(out_dir, exist_ok=True)

    # 0) （選用）生標籤
    prompt_src = "default"
    if as_bool(args.regen_tags):
        start, end = 0, -1
        if args.regen_range:
            m = re.match(r"^(-?\d+):(-?\d+)$", args.regen_range.strip())
            if not m:
                raise ValueError("--regen-range 格式錯誤，例 1100:-1")
            start, end = int(m.group(1)), int(m.group(2))

        prompt_tpl = None
        if args.prompt_inline:
            prompt_tpl = args.prompt_inline
            prompt_src = "inline"
        elif args.prompt_file:
            prompt_tpl = Path(args.prompt_file).read_text(encoding="utf-8")
            prompt_src = f"file:{args.prompt_file}"

        # 兼容 batch_generate_labels 是否支援新參數：動態檢查簽章
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
        print("✓ 標籤生成完成。")

    # 1) （選用）對標籤做 BM25 評估
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
        print(f"✓ 標籤 BM25 評估完成 -> {args.eval_out_csv_bm25}")

    # 1b) （選用）對標籤做 cosine similarity 評估
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
        print(f"✓ 標籤 Cosine 相似度評估完成 -> {args.eval_out_csv_cosine}")

    # 2) 構建/載入向量庫（詞找書/書搜書用）
    embedder = EmbeddingClient(model=args.embedding_model)
    store = VectorStore(excel_path=args.excel)
    if as_bool(args.rebuild_index):
        store.build(rebuild=True, embedder=embedder)
    else:
        store.load()

    # 3) 讀取 queries（詞找書）
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
    # if not queries:
    #     queries = ["心理學,成長"]

    # 3b) 讀取來源書（書搜書）
    src_books = []
    if args.books:
        src_books = [s.strip() for s in re.split(r"[;\uFF1B]+", args.books) if s.strip()]
    if args.books_file:
        with open(args.books_file, encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    src_books.append(t)
    # if not src_books:
    #     src_books = ["一億元的分手費;洛克菲勒寫給兒子的38封信"]

    # 4) 建 BM25 語料（summary 全庫）& cosine similarity
    books_df = pd.read_csv(args.books_csv).drop_duplicates(subset=["書名"]).reset_index(drop=True)
    docs = []
    title_to_docid = {}
    title_to_summary = {}
    for _, r in books_df.iterrows():
        title = str(r.get("書名", "")).strip()
        summary = str(r.get("書籍簡介", "")).strip()
        tokens = tokenize(summary, mode=args.tokenizer, ngram=args.ngram)
        title_to_docid[title] = len(docs)
        title_to_summary[title] = summary
        docs.append(tokens)
    bm25 = BM25(docs, BM25Params(k1=args.bm25_k1, b=args.bm25_b))

    # 5) 詞找書 + BM25（per-term 聚合）
    svc = SearchService(store, embedder)
    out_rows = []

    terms_csv_rows = []  # ← 新增：彙整成一張表
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
        # per-term BM25 聚合（sum/avg/min）& cosine similarity
        bm25_scores = []
        per_term_list = []  # 如要日後輸出細項可用
        q_emb = np.mean(embedder.embed_texts(terms_split), axis=0)
        cosine_scores = []
        for _, r in df.iterrows():
            did = title_to_docid.get(r["書名"])
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
            sentences = title_to_summary.get(r["書名"], "").split("。")
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
        show_cols = ["書名", "分類", "score", "bm25_score", "cosine_score"]
        print(df[show_cols].head(args.topk))

        # 累積彙整表列（rank 化）
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            terms_csv_rows.append({
                "query": q,
                "rank": rank,
                "書名": row["書名"],
                "分類": row["分類"],
                "score": row["score"],
                "bm25_score": row["bm25_score"],
                "cosine_score": row["cosine_score"],
                "exact_hits": row.get("exact_hits", ""),
                "similar_hits": row.get("similar_hits", "")
            })

        out_rows.append({"type": "terms_to_books", "query": q, "results": df.to_dict(orient="records")})

    # 6) 書搜書 + BM25
    books_csv_rows = []  # ← 新增：彙整成一張表
    if src_books:
        # 來源書「查詢」的 token：依 bookq-mode 而定
        excel_df = pd.read_excel(args.excel).drop_duplicates(subset=["書名"]).reset_index(drop=True)
        title_to_tags = {
            str(r["書名"]).strip(): [s.strip() for s in str(r["標籤"]).split(",") if s and s.strip()]
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
                # BM25 查詢：tags 合併 或 summary
                if args.bookq_mode_bm25 == "tags":
                    tags = title_to_tags.get(title, [])
                    q_tokens = merge_query_tokens(tags, tokenizer=args.tokenizer,
                                                ngram=args.ngram, merged_tf=args.bookq_merged_tf_bm25)
                else:  # summary
                    src_row = books_df[books_df["書名"] == title]
                    src_summary = str(src_row.iloc[0]["書籍簡介"]) if not src_row.empty else ""
                    q_tokens = tokenize(src_summary, mode=args.tokenizer, ngram=args.ngram)

                # cosine similarity 查詢：tags 合併 或 summary
                if args.bookq_mode_cosine == "tags":
                    tags = title_to_tags.get(title, [])
                else:  # summary
                    src_row = books_df[books_df["書名"] == title]
                    src_summary = str(src_row.iloc[0]["書籍簡介"]) if not src_row.empty else ""
                
                # diagnostic：title
                if as_bool(args.bookq_title_sim):
                    title_clean = embedder.embed_text(clean_text(title))
                    title_clean = np.ravel(title_clean) if title_clean is not None else None

                    emb = pd.DataFrame()
                    emb["title_emb"] = df["書名"].apply(lambda t: np.ravel(embedder.embed_text(clean_text(str(t)))))
                    if title_clean is not None:
                        sim_list_title = [1 - cosine_distance(r_emb, title_clean) for r_emb in emb["title_emb"]]
                        if "title_sim" not in df.columns:
                            df["title_sim"] = 0.0
                        df["title_sim"] = df["title_sim"] + sim_list_title

                # diagnostic: type
                if as_bool(args.bookq_type_sim):
                    src_row = books_df[books_df["書名"] == title]
                    src_type = str(src_row.iloc[0]["書籍分類第二層"]) if not src_row.empty else ""

                    src_type_emb = embedder.embed_text(src_type)
                    src_type_emb = np.ravel(src_type_emb) if src_type_emb is not None else None

                    emb = pd.DataFrame()
                    emb["type_emb"] = df["分類"].apply(lambda t: np.ravel(embedder.embed_text(str(t))))
                    if src_type_emb is not None:
                        sim_list_type =[1 - cosine_distance(r_emb, src_type_emb) for r_emb in emb["type_emb"]]
                        if "type_sim" not in df.columns:
                            df["type_sim"] = 0.0
                        df["type_sim"] = df["type_sim"] + sim_list_type

                # 打分
                scores_bm = []
                scores_cosine = []
                norms_bm = []
                norms_cosine = []
                q_len = len(q_tokens)
                for _, r in df.iterrows():
                    did = title_to_docid.get(r["書名"])
                    res_summary = title_to_summary.get(r["書名"])
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
            show_cols = ["書名", "分類", "score", "bm25_score", "cosine_score"]
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

            # 累積彙整表列（rank 化）
            for rank, (_, row) in enumerate(df.iterrows(), start=1):
                rec = {
                    "source_title": titles,
                    "bookq_mode_bm25": args.bookq_mode_bm25,
                    "bookq_mode_cosine": args.bookq_mode_cosine,
                    "rank": rank,
                    "書名": row["書名"],
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

    # 7) 存結果 + meta
    out_path = Path(out_dir) / Path(args.out).name
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n✓ 結果已存 {out_path}")

    # 7.1 另外輸出兩張「一次看全部」的表
    terms_csv_path = out_path.with_name(out_path.stem + "_terms.csv")
    if terms_csv_rows:
        pd.DataFrame(terms_csv_rows).to_csv(terms_csv_path, index=False)
        print(f"✓ 詞找書彙整：{terms_csv_path}")

    books_csv_path = out_path.with_name(out_path.stem + "_books.csv")
    if books_csv_rows:
        pd.DataFrame(books_csv_rows).to_csv(books_csv_path, index=False)
        print(f"✓ 書找書彙整：{books_csv_path}")

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
    print(f"✓ Meta info 已存 {meta_path} (耗時 {elapsed} 秒)")

if __name__ == "__main__":
    main()