# app/book_to_keywords.py
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bookrec.core import VectorStore, SearchService

def ask(prompt, default=None, cast=str):
    s = input(f"{prompt}" + (f"（Default：{default}）" if default is not None else "") + "：").strip()
    if not s and default is not None:
        return default
    return cast(s)

def main():
    print("\n=== book → keywords(Input Title, Find Keywords) ===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"Failed to load vector store: {e}\nPlease run: python app/build_index.py to build the index first.")
        return
    svc = SearchService(store)

    title = ask("Enter the source book title (only one title allowed)")
    k   = ask("How many keywords to display?", 10, int)

    result_list = svc.book_to_keywords(
        title,
        k=k,
    )

    print("\nSearch Results:\n")
    print(f"Source Title: {title}\n")
    if len(result_list) != 0:
        for i, tag in enumerate(result_list, start=1):
            print(f"  {i}. {tag}")
    else:
        print("(No results found)")

if __name__ == "__main__":
    main()
