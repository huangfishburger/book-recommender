# app/book_to_book.py
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

def ask_mode():
    s = input("模式（Enter 預設 soft_and；可選 min/sum/soft_and/avg；別名 and/or/soft/avg）：").strip().lower()
    if not s:
        return None
    alias = {"and":"min","or":"sum","soft":"soft_and","avg":"avg","average":"avg"}
    return alias.get(s, s)

def main():
    print("\n=== book → book（輸入書名找相似書）===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"⚠️ 無法載入向量庫：{e}\n請先執行：python app/build_index.py 建立索引。")
        return

    svc = SearchService(store, EmbeddingClient())

    titles = ask("請輸入來源書名（可輸入多本，請用分號 ; 或 ； 分隔，例如：解憂雜貨店; 你的名字）")
    topk   = ask("要顯示前幾本？", 10, int)
    tagk   = ask("每個標籤擴展相似標籤先取幾個（tag_topk）", 20, int)
    sim_th = ask("標籤相似度門檻（0~1）", 0.70, float)
    mode   = ask_mode()  # None -> 預設 soft_and

    bonus = 0.0
    if (mode or "soft_and") == "soft_and":
        bonus = ask("多本同時命中加分（cooccur_bonus，soft_and 用）", 0.2, float)

    df = svc.books_to_books(
        titles,
        tag_topk=tagk,
        sim_th=sim_th,
        topk=topk,
        mode=mode,
        cooccur_bonus=bonus
    )

    print("\n查詢結果：\n")
    print(df.to_string(index=False) if not df.empty else "（沒有結果）")

if __name__ == "__main__":
    main()
