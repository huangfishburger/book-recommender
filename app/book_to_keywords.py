# app/book_to_keywords.py
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bookrec.core import VectorStore, SearchService

def ask(prompt, default=None, cast=str):
    s = input(f"{prompt}" + (f"（預設：{default}）" if default is not None else "") + "：").strip()
    if not s and default is not None:
        return default
    return cast(s)

def main():
    print("\n=== book → book（輸入書名找主題詞）===\n")
    try:
        store = VectorStore().load()
    except Exception as e:
        print(f"⚠️ 無法載入向量庫：{e}\n請先執行：python app/build_index.py 建立索引。")
        return
    svc = SearchService(store)

    title = ask("請輸入來源書名（只可輸入一本）")
    k   = ask("要顯示幾個主題詞？", 10, int)

    result_list = svc.book_to_keywords(
        title,
        k=k,
    )

    print("\n查詢結果：\n")
    print(f"書名：{title}\n")
    if len(result_list) != 0:
        for i, tag in enumerate(result_list, start=1):
            print(f"  {i}. {tag}")
    else:
        print("（沒有結果）")

if __name__ == "__main__":
    main()
