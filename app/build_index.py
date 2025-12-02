# app/build_index.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bookrec.core import VectorStore
from bookrec.embeddings import EmbeddingClient

def ask(prompt, default=None, cast=str):
    s = input(f"{prompt}" + (f"（Default：{default}）" if default is not None else "") + "：").strip()
    if not s and default is not None:
        return default
    if cast is bool:
        return s.lower() in ["y", "yes", "1", "true", "t", ""]
    return cast(s)

def main():
    print("\n=== Build/Update Vector Store (Offline) ===\n")
    excel  = ask("Excel path", "data/process/title_tags.xlsx", str)
    model  = ask("Embedding model", "text-embedding-3-small", str)
    force  = ask("Rebuild all tag vectors? (Y/n)", True, bool)

    store = VectorStore(excel_path=excel)
    emb   = EmbeddingClient(model=model)
    store.build(rebuild=force, embedder=emb)
    print("Vector store built successfully.")

if __name__ == "__main__":
    main()
