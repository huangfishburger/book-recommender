# app/book_to_book.py
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

def ask_mode():
    s = input("Mode (Enter for default soft_and; options: min/sum/soft_and/avg; aliases: and/or/soft/avg): ").strip().lower()
    if not s:
        return None
    alias = {"and":"min","or":"sum","soft":"soft_and","avg":"avg","average":"avg"}
    return alias.get(s, s)

def main():
    print("\n=== book → book（Input Title, Find Similar Books）===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"Failed to load vector store: {e}\nPlease run: python app/build_index.py to build the index first.")
        return

    svc = SearchService(store, EmbeddingClient())

    titles = ask("Enter source book titles (can enter multiple, separate with semicolon ; or ；, e.g., Harry Potter; Twilight)")
    topk   = ask("How many top books to display?", 10, int)
    tagk   = ask("How many similar tags to retrieve for expansion (tag_topk)", 20, int)
    sim_th = ask("Tag similarity threshold (0~1)", 0.70, float)
    mode   = ask_mode()  # None -> defaults to soft_and

    bonus = 0.0
    if (mode or "soft_and") == "soft_and":
        bonus = ask("Co-occurrence bonus for multiple title hits (cooccur_bonus, used for soft_and)", 0.2, float)

    df = svc.books_to_books(
        titles,
        tag_topk=tagk,
        sim_th=sim_th,
        topk=topk,
        mode=mode,
        cooccur_bonus=bonus
    )

    print("\nSearch Results:\n")
    print(df.to_string(index=False) if not df.empty else "(No results found)")

if __name__ == "__main__":
    main()
