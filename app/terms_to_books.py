# app/term_to_book.py
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
    if cast is bool:
        return s.lower() not in ["n", "no", "0", "false", "f", ""]
    return cast(s)

def ask_mode():
    """
    回傳 None 表示使用預設（soft_and）。
    支援別名：
      and -> min,  or -> sum,  soft -> soft_and,  avg -> avg
    """
    s = input("模式（Enter 預設 soft_and；可選 min/sum/soft_and/avg；別名 and/or/soft/avg）：").strip().lower()
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
    print("\n=== term → book（輸入關鍵詞找書）===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"⚠️ 無法載入向量庫：{e}\n請先執行：python app/build_index.py 建立索引。")
        return

    svc = SearchService(store, EmbeddingClient())

    q        = ask("請輸入查詢關鍵詞（單一詞，或用逗號分隔多詞，例如：奇幻, 龍）")
    topk     = ask("要顯示前幾本？", 10, int)
    tagk     = ask("擴展相似標籤先取幾個（tag_topk）", 20, int)
    sim_th   = ask("相似度門檻（0~1）", 0.55, float)
    use_idf  = ask("使用 IDF 權重？（Y/n）", False, bool)
    mode     = ask_mode()  # None 代表預設 soft_and
    bonus    = ask("同書多詞命中加分（cooccur_bonus，soft_and 用）", 0.2, float)

    df = svc.terms_to_books(
        q,
        tag_topk=tagk,
        sim_th=sim_th,
        topk=topk,
        use_idf=use_idf,
        mode=mode,              # None -> 預設 soft_and
        cooccur_bonus=bonus
    )

    print("\n查詢結果：\n")
    print(df.to_string(index=False) if not df.empty else "（沒有結果）")

if __name__ == "__main__":
    main()
