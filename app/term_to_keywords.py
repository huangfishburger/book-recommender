# app/term_to_keywords.py
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bookrec.core import VectorStore, SearchService
from bookrec.embeddings import EmbeddingClient

def ask(prompt, default=None, cast=str):
    s = input(f"{prompt}" + (f"（預設：{default}）" if default is not None else "") + "：").strip()
    if not s and default is not None:
        return default
    return cast(s)

def main():
    print("\n=== term → keywords（輸入關鍵詞找相近標籤）===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"⚠️ 無法載入向量庫：{e}\n請先執行：python app/build_index.py 建立索引。")
        return

    svc = SearchService(store, EmbeddingClient())

    q      = ask("請輸入查詢關鍵詞（term）")
    k      = ask("要顯示前幾個標籤？", 10, int)
    sim_th = ask("相似度門檻（0~1）", 0.60, float)

    # 注意：SearchService.term_to_keywords 需為單一 k 版
    df = svc.term_to_keywords(q, k=k, sim_th=sim_th)

    print("\n查詢結果：\n")
    print(df.to_string(index=False) if not df.empty else "（沒有結果）")

if __name__ == "__main__":
    main()