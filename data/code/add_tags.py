# data/code/add_tags.py
import os, re, warnings
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

load_dotenv()
client = OpenAI()

# 預設的 system prompt 與 user prompt 模板（可被 script 覆寫）
SYSTEM_PROMPT_DEFAULT = "你的主要工作是根據每本書的內容，提供對應的主題詞。"

PROMPT_TEMPLATE_DEFAULT = """
請使用 **繁體中文**，根據書籍探討的核心內容或情節，找出書中 15 到 20 個詳細且完整的標籤（每個標籤須為 6 個字以內的名詞），
標籤粒度適中，避免過於寬泛（如「歷史」、「文學」）或過於細碎（如「清代乾隆年間文人書信體例」）。

以下為輸入與對應範例：
書名：《被討厭的勇氣》
作者：岸見一郎
內容簡介：透過一位青年與哲人對話的形式，深入淺出地介紹阿德勒心理學的核心思想…

範例標籤：
#阿德勒心理學 #貢獻他人 #自我成長 #人際關係 #課題分離 #對話形式 #信念轉換 #自我接納 #勇氣挑戰 #當下行動 #否定承擔 #責任選擇 #幸福源泉 #自卑與超越 #社會關係 #歸屬感

請依上述格式，根據以下書籍資訊，**以繁體中文**生成 **15 到 20** 個標籤：
書名：{title}
作者：{author}
內容簡介：{summary}
""".strip()


def process_with_ai(
    text: str,
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT_DEFAULT},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ AI處理時出現錯誤：{e}"


def generate_topic_labels(
    title: str,
    author: str,
    summary: str,
    model: str = "gpt-4o",
    prompt_tpl: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> list[str]:
    template = (prompt_tpl or PROMPT_TEMPLATE_DEFAULT)
    prompt = template.format(title=title, author=author, summary=summary)

    raw = process_with_ai(prompt, model=model, system_prompt=system_prompt, temperature=temperature)
    raw = re.sub(r"\s+", " ", raw).strip()
    labels = [s.strip() for s in raw.split("#") if s.strip()]
    return labels


def batch_generate_labels(
    books_csv: str = "data/process/processed_books.csv",
    output_file: str = "data/process/十類書名_標籤.xlsx",
    start: int = 0,
    end: int = -1,
    save_every: int = 500,
    model: str = "gpt-4o",
    prompt_tpl: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.3,
):
    """
    從 processed_books.csv 批次產生標籤，存成 Excel。
    可設定起訖 index、存檔頻率、模型，以及自定 prompt/system_prompt/temperature。
    """
    books = pd.read_csv(books_csv)
    books.drop_duplicates(subset=["書名"], inplace=True)
    books.reset_index(drop=True, inplace=True)

    if end == -1:
        end = len(books)

    results = pd.DataFrame(columns=["書名", "分類", "標籤"])

    for i in range(start, end):
        title = str(books.loc[i, "書名"])
        author = str(books.loc[i, "作者"])
        summary = str(books.loc[i, "書籍簡介"])
        type_ = str(books.loc[i, "書籍分類第二層"])

        # 生成標籤（用可覆寫的模板/參數）
        labels = generate_topic_labels(
            title=title,
            author=author,
            summary=summary,
            model=model,
            prompt_tpl=prompt_tpl,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # 格式化
        labels = re.sub(r"\s+", " ", " ".join(labels)).strip()
        labels_list = [label.strip() for label in labels.split("#") if label.strip()]

        # 加入 DataFrame
        results.loc[i, "書名"] = title
        results.loc[i, "分類"] = type_
        results.loc[i, "標籤"] = ", ".join(labels_list)

        # 週期存檔
        if (i + 1) % save_every == 0 or (i + 1) == end:
            print(f"已處理 {i + 1} 筆書籍，暫存結果...")
            if os.path.exists(output_file):
                old = pd.read_excel(output_file)
                combined = pd.concat([old, results], ignore_index=True)
            else:
                combined = results
            combined.to_excel(output_file, index=False)
            results = pd.DataFrame(columns=["書名", "分類", "標籤"])
            print("暫存完成。")

if __name__ == "__main__":
    batch_generate_labels(start=1100)