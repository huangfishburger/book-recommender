# app/term_to_keywords.py
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
    return cast(s)

def main():
    print("\n=== Term → Keywords (Input Keyword, Find Similar Tags) ===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"Failed to load vector store: {e}\nPlease run: python app/build_index.py to build the index first.")
        return

    svc = SearchService(store, EmbeddingClient())

    q      = ask("Enter the search term (keyword)")
    k      = ask("How many top tags to display?", 10, int)
    sim_th = ask("Similarity threshold (0~1)", 0.60, float)

    # Note: SearchService.term_to_keywords requires a single k version
    df = svc.term_to_keywords(q, k=k, sim_th=sim_th)

    print("\nSearch Results:\n")
    print(df.to_string(index=False) if not df.empty else "(No results found)")

if __name__ == "__main__":
    main()