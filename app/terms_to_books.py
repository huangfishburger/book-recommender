# app/term_to_book.py
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bookrec.core import VectorStore, SearchService
from bookrec.embeddings import EmbeddingClient

def ask(prompt, default=None, cast=str):
    s = input(f"{prompt}" + (f"（Default：{default}）" if default is not None else "") + "：").strip()
    if not s and default is not None:
        return default
    if cast is bool:
        return s.lower() not in ["n", "no", "0", "false", "f", ""]
    return cast(s)

def ask_mode():
    """
    Returns None to use the default (soft_and).
    Supports aliases:
      and -> min,  or -> sum,  soft -> soft_and,  avg -> avg
    """
    s = input("Mode (Enter for default soft_and; options: min/sum/soft_and/avg; aliases: and/or/soft/avg): ").strip().lower()
    if not s:
        return None
    alias = {
        "and": "min",
        "or": "sum",
        "soft": "soft_and",
        "avg": "avg",
    }
    return alias.get(s, s)

def main():
    print("\n=== Term → Book (Input Keywords, Find Books) ===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"Failed to load vector store: {e}\nPlease run: python app/build_index.py to build the index first.")
        return

    svc = SearchService(store, EmbeddingClient())

    q      = ask("Enter query keywords (single term, or multiple terms separated by comma, e.g., Fantasy, Dragon)")
    topk   = ask("How many top books to display?", 10, int)
    tagk   = ask("How many similar tags to retrieve for expansion (tag_topk)", 20, int)
    sim_th = ask("Similarity threshold (0~1)", 0.55, float)
    use_idf = ask("Use IDF weighting? (Y/n)", False, bool)
    mode   = ask_mode()  # None -> defaults to soft_and
    bonus  = ask("Co-occurrence bonus for multiple term hits on the same book (cooccur_bonus, used for soft_and)", 0.2, float)

    df = svc.terms_to_books(
        q,
        tag_topk=tagk,
        sim_th=sim_th,
        topk=topk,
        use_idf=use_idf,
        mode=mode,              # None -> default soft_and
        cooccur_bonus=bonus
    )

    print("\nSearch Results:\n")
    print(df.to_string(index=False) if not df.empty else "(No results found)")

if __name__ == "__main__":
    main()
