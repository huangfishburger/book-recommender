# BookRec Runner & Evaluator

這個專案提供兩條主線功能：

1. **標籤生成**：用 GPT 依書名/作者/簡介產生 15–20 個繁中標籤，輸出到 Excel。
2. **檢索與評估**：

   * **詞找書**（terms → books）：用標籤向量庫檢索，並以 **BM25** 或 **Cosine Similarity** 對每個 term 與命中書的**簡介**逐一計分，最後做聚合（`avg/sum/min`）。
   * **書找書**（books → books）：以來源書（**tags** 或 **summary**）找相似書，並對「來源查詢」與候選書**簡介**計算 BM25、Cosine Similarity（可選長度正規化），也可以對「來源查詢」與候選書**書名**、**分類**計算 Cosine Similarity。
   * **標籤品質評估**：以 BM25、Cosine Similarity 評估「每本書的標籤 vs 自身書籍簡介」的相關度（`evaluation.py`）。

所有執行都可用單檔 **`bookrec/script.py`** 控制；會輸出**結果**、**兩張彙整表（terms/books）**與**執行參數＋耗時（meta.json）**，方便重現與比較。

---

## 0. 需求與安裝

**Python**：3.9+

**套件**：

* 必備：`pandas`, `numpy`, `faiss-cpu`, `openpyxl`, `python-dotenv`, `tqdm`
* 選用：`jieba`（中文切詞更準）、`rich`（想要更漂亮的 CLI 輸出可自加）

```bash
pip install -U pandas numpy faiss-cpu openpyxl python-dotenv tqdm jieba
```

**OpenAI 金鑰（若要生成標籤）**：

* 設定環境變數 `OPENAI_API_KEY`，或在專案根目錄放 `.env`：

  ```env
  OPENAI_API_KEY=sk-************************
  ```

**預設路徑**：

```
data/process/processed_books.csv      # 原始書目（書名/作者/書籍簡介/書籍分類第二層）
data/process/十類書名_標籤.xlsx         # 標籤輸出（書名/分類/標籤）
data/process/                         # 向量、快取與評估輸出
```

> **建議以模組方式執行**（確保 `bookrec` 能被找到）：
>
> ```bash
> python -m bookrec.script ...
> ```

---

## 1. 檔案一覽

* `data/code/add_tags.py`：標籤生成（`batch_generate_labels()`）。
* `bookrec/embeddings.py`：`EmbeddingClient`（OpenAI Embeddings）。
* `bookrec/core.py`：

  * `VectorStore`：讀 Excel 標籤，建/載 FAISS、倒排索引與 IDF。
  * `SearchService`：

    * `terms_to_books(query_input, ...)`（**詞找書**）
    * `books_to_books(titles_input, ...)`（**書找書**）
* `data/code/evaluation.py`：BM25、cosine similarity 工具與標籤評估（`BM25`, `BM25Params`, `tokenize`, `evaluate`）。
* `bookrec/script.py`：一檔整合（可選重生標籤 → 可選標籤評估 → 造/載向量庫 → 詞找書/書找書 + BM25 + cosine similarity → 輸出結果與 meta & 彙整表）。

---

## 2. 輸入格式

### `processed_books.csv`

必備欄位：

* `書名`（str，唯一鍵）
* `作者`（str）
* `書籍簡介`（str）
* `書籍分類第二層`（str）

### `十類書名_標籤.xlsx`

由 `add_tags.py` 或 `batch_generate_labels()` 產生：

* `書名`（str）
* `分類`（str）
* `標籤`（str，逗號分隔，例如 `阿德勒心理學, 課題分離, 自我成長`）

---

## 3. 快速開始

### 3.1 直接跑**詞找書**（不用重跑標籤）

```bash
python -m bookrec.script \
  --excel data/process/十類書名_標籤.xlsx \
  --books-csv data/process/processed_books.csv \
  --terms "奇幻, 科學; 投資, 財報" \
  --topk 10
```

輸出：

* 終端列印每個查詢的 Top-K（向量檢索分數 + **per-term 聚合 BM25** 分數 + cosine similarity 分數）
* 結果檔案在 experiment/當日日期 資料夾內
* **`results.jsonl`**：每行一個 JSON（檢索結果）
* **`results_terms.csv`**：一次看所有「詞找書」查詢的 Top-K（欄位含 `query, rank, 書名, 分類, score, bm25_score, cosine_score, exact_hits, similar_hits`）
* **`results.meta.json`**：開始/結束時間、耗時、所有參數、彙整表路徑

### 3.2 重生**標籤**並做**BM25 標籤評估**

```bash
python -m bookrec.script \
  --regen-tags true \
  --gpt-model gpt-4o \
  --books-csv data/process/processed_books.csv \
  --excel data/process/十類書名_標籤.xlsx \
  --eval-tags-bm25 true \
  --eval-out-csv data/process/bm25_eval_summary.csv \
  --eval-out-jsonl data/process/bm25_eval_details.jsonl
```

輸出：

* 產生/覆寫 `十類書名_標籤.xlsx`
* 標籤 BM25 評估：`bm25_eval_summary.csv`（每本彙整）、`bm25_eval_details.jsonl`（每標籤細節）
* 若同時也輸入查詢/書，會另外產生 `results.jsonl`、`results_terms.csv`/`results_books.csv` 與 `results.meta.json`

### 3.3 重生**標籤**並做**Cosine Simialrity 標籤評估**

```bash
python -m bookrec.script \
  --regen-tags true \
  --gpt-model gpt-4o \
  --books-csv data/process/processed_books.csv \
  --excel data/process/十類書名_標籤.xlsx \
  --eval-tags-cosine true \
  --eval-out-csv data/process/cosine_eval_summary.csv \
  --eval-out-jsonl data/process/cosine_eval_details.jsonl
```

輸出：

* 產生/覆寫 `十類書名_標籤.xlsx`
* 標籤 Cosine Simialrity 評估：`cosine_eval_summary.csv`（每本彙整）、`cosine_eval_details.jsonl`（每標籤細節）
* 若同時也輸入查詢/書，會另外產生 `results.jsonl`、`results_terms.csv`/`results_books.csv` 與 `results.meta.json`

### 3.4 **書找書**（bm25用來源書 **標籤** 作計算）

```bash
python -m bookrec.script \
  --excel data/process/十類書名_標籤.xlsx \
  --books-csv data/process/processed_books.csv \
  --books "被討厭的勇氣; 原子習慣" \
  --bookq-mode-bm25 tags \
  --bookq-merged-tf-bm25 binary \
  --bookq-normalize-bm25 true \
  --bookq-mode-cosine tags \
  --embedding-level-books sentence-max \
  --bookq-normalize-cosine true \
  --bookq_title_sim true \
  --bookq_type_sim true \
  --topk 10
```

輸出（除終端與 `results.jsonl` 外）：

* 結果檔案在 experiment/當日日期 資料夾內
* **`results_books.csv`**：一次看所有「書找書」來源書的 Top-K（欄位含 `source_title, bookq_mode_bm25, bookq_mode_cosine, rank, 書名, 分類, score, source_hits, match_count, bm25_score, cosine_score, [title_sim, type_sim, bm25_norm, cosine_norm]`）

### 3.5 **書找書**（bm25 用來源書 **簡介** 作計算）

```bash
python -m bookrec.script \
  --excel data/process/十類書名_標籤.xlsx \
  --books-csv data/process/processed_books.csv \
  --books "被討厭的勇氣" \
  --bookq-mode-bm25 summary \
  --bookq-mode-cosine summary \
  --embedding-level-books sentence-max \
  --bookq_title_sim true \
  --bookq_type_sim true \
  --topk 10
```

> 若沒有提供 `--books` / `--books-file` 或來源書無法命中，`results_books.csv` 不會生成；可在 `results.meta.json` 的 `books_csv` 檢查實際路徑（或是否為空字串）。

---

## 4. 參數說明（含預設）

### 4.1 標籤生成（可選）

* `--regen-tags` (`false`)：是否重跑標籤生成（呼叫 OpenAI）
* `--gpt-model` (`gpt-4o`)：GPT 模型
* `--books-csv` (`data/process/processed_books.csv`)
* `--regen-range`（如 `1100:-1`）：指定索引區間
* `--save-every` (`500`)：暫存頻率
* `--excel` (`data/process/十類書名_標籤.xlsx`)：標籤輸出目標
* **Prompt 可調**：

  * `--system-prompt`：system 指令（預設：*你的主要工作是根據每本書的內容，提供對應的主題詞。*）
  * `--prompt-inline`：直接在 CLI 傳入 user prompt 模板字串（支援 `{title} {author} {summary}` 替換）
  * `--prompt-file`：從檔案載入模板（UTF-8），同樣支援 `{title} {author} {summary}`
  * `--temperature`：呼叫 Chat Completions 的溫度（預設 0.3）

> **Prompt 來源記錄**：`results.meta.json` 的 `prompt_source` 會標註 `default` / `inline` / `file:PATH`。

### 4.2 向量庫與檢索

* `--embedding-model` (`text-embedding-3-small`)
* `--rebuild-index` (`false`)：是否重建向量庫（否則載入既有）
* `--sim-th` (`0.55`)：相似標籤門檻
* `--tag-topk` (`20`)：每個 term 取相似標籤 Top-K
* `--topk` (`10`)：回傳最終書籍 Top-K
* `--mode` (`soft_and`)：book 分數聚合模式（`sum` / `avg` / `min` / `soft_and`）
* `--use-idf` (`false`)：相似標籤加權是否乘以 tag 的 IDF
* `--cooccur-bonus` (`0.2`)：`soft_and` 模式對多 term 共現加分

### 4.3 詞找書（輸入 + BM25 聚合）

* `--terms`：分號分隔多組查詢；每組以逗號切分詞（支援全形/半形）
* `--terms-file`：每行一組查詢（同上格式）
* **BM25 計分方式**：將查詢切成多個 **term**，**每個 term 分別**與候選書**簡介**計算 BM25 分數，再用下列方式聚合：

  * `--terms-bm25-agg {avg|sum|min}`（預設 `avg`）

> 詞找書 BM25 亦會沿用 `--bm25-*` 參數（tokenizer/ngram/k1/b）。

### 4.4 書找書（輸入 + evaluate）

* `--books` / `--books-file`：來源書名（分號或逐行）
* `--bookq-mode-bm25` (`tags|summary`，預設 `tags`）：
* `--bookq-mode-cosine` (`tags|summary`，預設 `tags`）：

  * `tags`：以來源書的**所有標籤**合併為一次查詢
  * `summary`：以來源書**簡介**當作查詢
* `--bookq-merged-tf-bm25`（`binary|log|raw`，預設 `binary`）：僅在 `tags` 模式下生效

  * `binary`：同一 token 只計一次
  * `log`：出現 f 次 → 重複 ⌈log(1+f)⌉ 次
  * `raw`：保留原始頻次
* `--bookq-normalize-bm25` (`true|false`，預設 `true`)：BM25 分數是否除以查詢長度（避免長查詢佔優）
* `--bookq-normalize-cosine` (`true|false`，預設 `true`)：cosine similarity 分數是否除以查詢長度（避免長查詢佔優）

### 4.5 BM25（共用參數：評估與檢索打分）

* `--bm25-tokenizer` (`auto|jieba|whitespace|char`，預設 `auto`)

  * `auto`：若安裝 `jieba` 則優先使用；否則中文 fallback 到「字元 n-gram」
* `--bm25-ngram` (`2`)：在 `char`/fallback 模式下的 n（中文建議 2）
* `--bm25-k1` (`1.2`), `--bm25-b` (`0.75`)：BM25 參數
* `--bm25-topk-mean` (`5`)：在 **tag-each** 模式下，Top-k 平均（看標籤代表性時很實用）

### 4.6 Cosine Similarity（共用參數：評估與檢索打分）

* `--embedding-level` (`sentence-max|sentence-avg`，預設`sentence-max`)：以所有句子的最大相似度或平均相似度代表整體
* `--cosine-topk-mean` (`5`)：在 **tag-each** 模式下，Top-k 平均（看標籤代表性時很實用）

### 4.7 標籤品質評估（可選）(有 BM 和 Cosine Similarity)

* `--eval-out-csv`：彙整表（每本書一列）
* `--eval-out-jsonl`：細節檔（每本書一列，含每個標籤分數）

---

## 5. 輸出說明

### 5.1 `results.jsonl`

每行一筆 JSON：

* **詞找書**：

  ```json
  {
    "type": "terms_to_books",
    "query": "奇幻, 科學",
    "results": [
      {"書名": "...", "分類": "...", "tags": ["..."], "score": 3.27, "exact_hits": 1, "similar_hits": 5, "bm25_score": 1.482, "cosine_score": 0.393},
      ...
    ]
  }
  ```
* **書找書**：

  ```json
  {
    "type": "books_to_books", 
    "source_title": "一億元的分手費", 
    "bookq_mode_bm25": "tags", 
    "bookq_mode_cosine": "tags", 
    "results": [
      {"書名": "...", "分類": "...", "score": 5.506, "source_hits": 1, "match_count": 8, "title_sim": 0.332112, "type_sim": 0.32942, "bm25_score": 30.789589, "cosine_score": 0.551491, "bm25_norm": 0.789477, "cosine_norm": 0.014141}
      ...
    ]
  }
  ```

### 5.2 `results_terms.csv` & `results_books.csv`

* 結果檔案在 experiment/當日日期 資料夾內
* **`results_terms.csv`**：彙整所有詞找書查詢。欄位：`query, rank, 書名, 分類, score, bm25_score, cosine_score, exact_hits, similar_hits`。
* **`results_books.csv`**：彙整所有書找書查詢。欄位：`source_title, bookq_mode_bm25, bookq_mode_cosine, rank, 書名, 分類, score, source_hits, match_count, bm25_score, cosine_score, [title_sim, type_sim, bm25_norm, cosine_norm]`。

> 若沒有任何書找書輸出列，`results_books.csv` 不會生成；實際路徑可看 `results.meta.json` 的 `books_csv` 欄位。

### 5.3 `results.meta.json`

* `start_time`, `end_time`, `elapsed_sec`
* `args`：所有 CLI 參數（方便重現）
* `prompt_source`：`default` / `inline` / `file:PATH`
* `terms_csv`, `books_csv`：兩張彙整表實際路徑（無資料為空字串）
* 若執行 `--eval-tags-bm25 true`：`bm25_eval_summary_csv`, `bm25_eval_details_jsonl`
* 若執行 `--eval-tags-cosine true`：`cosine_eval_summary_csv`, `cosine_eval_details_jsonl`

### 5.4 標籤 BM25 評估輸出（當 `--eval-tags-bm25 true`）

* `bm25_eval_summary.csv`（每書一列）：

  * 範例欄位：`書名, 標籤數, TE_avg, TE_max, TE_max_tag, TE_coverage, TE_top5_mean, TM_score, TM_len, TM_norm, TM_tf, TM_norm_on`
* `bm25_eval_details.jsonl`（每書一行）：

  * `{ "title": "xxx", "scores": [{"tag": "阿德勒心理學", "bm25": 1.234}, ...], "tag_merged_score": 3.21, ... }`

### 5.5 標籤 Cosine Similarity 評估輸出（當 `--eval-tags-cosine true`）

* `cosine_eval_summary.csv`（每書一列）：

  * 範例欄位：`書名, 標籤數, TE_avg, TE_max, TE_max_tag, TE_top5_mean, TM_score, TM_len, TM_norm, TM_norm_on`
* `bm25_eval_details.jsonl`（每書一行）：

  * `{ "title": "xxx", "scores": [{"tag": "阿德勒心理學", "cosine": 1.234}, ...], "tag_merged_score": 3.21, ... }`

---

## 6. 演算法小抄

* **向量檢索**：
  查詢（terms 或來源書標籤）→ 找相似標籤（FAISS）→ 按書聚合（可選 IDF/模式/共現加分）→ Top-K 書。

* **BM25（文件端一律是候選書「簡介」）**：

  * **詞找書**：將查詢用逗號切成多個 term，**逐 term** 與候選書簡介計分，最後以 `--terms-bm25-agg` 聚合（預設 `avg`）。
  * **書找書**：Query 來源看 `--bookq-mode-bm25`：

    * `tags`：來源書標籤合併（`binary/log/raw`），可 `--bookq-normalize-bm25`。
    * `summary`：來源書完整簡介。
    * 多本書搜尋的分數是取所有查詢書的平均。

* **Cosine Similarity（文件端一律是候選書「簡介」，會拆成句子）**：

  * **詞找書**：將查詢用逗號切成多個 term，**平均 term 的 embedding vector** 與候選書簡介逐句計分，最後以 `--embedding-level-books` 計分（預設 `sentence-max`）。
  * **書找書**：Query 來源看 `--bookq-mode-cosine`：

    * `tags`：來源書標籤合併（`binary/log/raw`），可 `--bookq-normalize-cosine`。
    * `summary`：來源書完整簡介，會拆成句子。
    * 多本書搜尋的分數是取所有查詢書的平均。

---

## 7. 常見問題（FAQ）

* **ModuleNotFoundError: bookrec?** 請在 repo 根目錄執行並用模組模式：`python -m bookrec.script ...`。
* **Excel 跟向量庫不同步？** 重新產生或修改了標籤 Excel 後，若載入時標籤數不同，請加 `--rebuild-index true` 重建向量庫。
* **為什麼沒看到 `results_books.csv`？** 未提供 `--books` / `--books-file`、來源書名不匹配、或無任何命中行時皆不會輸出；路徑可看 `results.meta.json` 的 `books_csv`。
* **BM25 切詞建議？** 安裝 `jieba`；若無，`auto` 會 fallback 到中文字元 n-gram，建議 `--bm25-ngram 2`。
* **只看一個代表性指標？** 建議看 `TE_top{k}_mean`（預設 k=5），比 `TE_max` 更穩定、比均值更有代表性。

---

## 8. 範例工作流

**完整流程（重生標籤 → 標籤評估 → 詞找書 & 書找書）**

```bash
python -m bookrec.script \
  --regen-tags true --gpt-model gpt-4o \
  --books-csv data/process/processed_books.csv \
  --excel data/process/十類書名_標籤.xlsx \
  --rebuild-index true \
  --eval-tags-bm25 true \
  --eval-tags-cosine true \
  --eval-out-csv-bm25 data/process/bm25_eval_summary.csv \
  --eval-out-jsonl-bm25 data/process/bm25_eval_details.jsonl \
  --eval-out-csv-cosine data/process/cosine_eval_summary.csv \
  --eval-out-jsonl-cosine data/process/cosine_eval_details.jsonl \
  --terms "心理學, 成長; 投資, 財報" \
  --books "被討厭的勇氣; 原子習慣" \
  --bookq-mode-bm25 tags --bookq-merged-tf-bm25 binary --bookq-normalize-bm25 true \
  --bookq-mode-cosine tags --embedding-level-books sentence-max --bookq-normalize-cosine true \
  --topk 10
```

> 如需將 **BM25、Cosine Similarity重新排序**（以 `_norm` 或 `_score` 取代向量分數排序）、或把**詞找書 BM25、Cosine Similarity**輸出成 per-term 細項，歡迎提出需求，我們可在 `script.py` 中加入對應開關與輸出欄位。

## 附錄
書籍標籤品質評估指標（`evaluate` 函式產生的各種指標）說明 (BM25、Cosine Similarity)

| 指標            | 說明                                                                 |
|-----------------|----------------------------------------------------------------------|
| **TE_avg**      | 各標籤分數的平均值，反映標籤整體與書籍內容的關聯度。             |
| **TE_max**      | 單一標籤的最高分數，代表最相關的標籤。                           |
| **TE_max_tag**  | 取得最高分數的標籤名稱，方便定位最具代表性的標籤。                     |
| **TE_coverage** | 標籤中分數 > 0 的比例，衡量有效標籤覆蓋率。                           |
| **TE_topK_mean**| 前 K 個最高分數標籤的平均值（預設 K=5），避免單一極端值干擾。          |
| **TM_score**    | 合併所有標籤後的總 BM25 分數。                                       |
| **TM_len**      | 合併查詢的 token 數量，反映標籤總長度。                               |
| **TM_norm**     | 正規化分數 = `TM_score / TM_len` (若啟用 normalize_query)。            |
| **TM_tf**       | 合併標籤時的詞頻策略（如 `binary` 僅計算是否出現，不考慮次數）。        |
| **TM_norm_on**  | 是否啟用正規化 (布林值)，決定 TM_norm 是否基於長度修正。              |

> 輸出格式：  
> - **CSV**：表格化摘要數據  
> - **JSONL**：逐書紀錄，含每標籤分數細節
