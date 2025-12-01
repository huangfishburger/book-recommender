# app/build_index.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bookrec.core import VectorStore
from bookrec.embeddings import EmbeddingClient

def ask(prompt, default=None, cast=str):
    s = input(f"{prompt}" + (f"（預設：{default}）" if default is not None else "") + "：").strip()
    if not s and default is not None:
        return default
    if cast is bool:
        return s.lower() in ["y", "yes", "1", "true", "t", ""]
    return cast(s)

def main():
    print("\n=== 建立/更新向量庫（離線）===\n")
    excel  = ask("Excel 路徑", "data/process/十類書名_標籤.xlsx", str)
    model  = ask("Embedding 模型", "text-embedding-3-small", str)
    force  = ask("要不要重建所有標籤向量？（Y/n）", True, bool)

    store = VectorStore(excel_path=excel)
    emb   = EmbeddingClient(model=model)
    store.build(rebuild=force, embedder=emb)
    print("✅ 向量庫建立完成。")

if __name__ == "__main__":
    main()
