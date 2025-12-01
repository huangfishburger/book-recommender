# Book_Recommender

## 1. Clone the repository

```bash
git clone https://github.com/chienyilin/Book_Recommender.git
cd Book_Recommender
```

## 2. Environment setup

### 2.1.1 法一：使用虛擬環境
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
Windows:
.\venv\Scripts\activate

macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2.1.1 法二：使用docker
```bash
# download docker desktop
https://www.docker.com/

# Build Docker image
docker build -t book_recommender .

# run container
docker run -it --rm book_recommender
```

## 3. Running the Project

### 3.1.1 法一：在本機重新建立 / 更新向量庫
```bash
python -m app.build_index
```
* 可更改 Excel 檔路徑（預設：data/process/十類書名_標籤.xlsx）
### 3.1.2 法二：下載已完成嵌入的向量庫資料夾
```bash
python data/code/downloadfiles.py
```

### 3.2 以書搜尋相關的書
```bash
python -m app.books_to_books
```
* 多本輸入用分號 ; 或全形 ； 分隔
* 合併模式：soft_and（預設）、sum、min、avg

### 3.3 以關鍵詞搜尋相關的書 
```bash
python -m app.terms_to_books
```
* 多詞輸入用逗號 , 或全形 ， 分隔
* 合併模式：soft_and（預設）、sum、min、avg
* 主題詞表中包含與否的關鍵詞皆可搜尋（僅程式搜尋邏輯不同）
* exact_hits：有多少個查詢詞本身（term）被包含在該書的 tags 裡
* similar_hits：所有相似標籤（排除與 term 完全相同的）一共被包含了多少次
  
### 3.4 以關鍵詞搜尋相似主題詞
```bash
python -m app.term_to_keywords
```
* 主題詞表中包含與否的關鍵詞皆可搜尋（僅程式搜尋邏輯不同）

### 3.5 以書搜尋相關主題詞
```bash
python -m app.book_to_keywords
```
* 所有主題詞表中包含的書皆可搜尋
* source_hits：統計有多少本不同來源書(搜尋的書）對該書至少命中一個（書籍本身的或擴展的相似）標籤
* match_count：彙總所有來源書的相似標籤命中次數
  
## 附錄：合併模式比較（term/books → book）

| 模式       | 計分公式（簡化）                                     | 命中要求                  | 特性與用途 |
|------------|------------------------------------------------------|---------------------------|------------|
| `soft_and` | `sum(scores) + cooccur_bonus * max(0, k - 1)`        | 不需全中（偏好同時命中）  | **預設**。在召回與精確間取平衡；同時被多詞/多本命中的結果更靠前 |
| `sum`      | `sum(scores)`                                        | 不需全中（寬鬆 OR）       | 召回最大，適合廣泛探索 |
| `min`      | `min(scores)`                                        | **越接近全中越好**        | 接近 AND；缺任一詞/來源貢獻低則整體分數被拉低（可做嚴格過濾） |
| `avg`      | `sum(scores) / #queries`                             | 不需全中                  | 把不同查詢的分數拉到同一尺度，方便跨查詢比較 |

* 註：`k` 為該書被命中的查詢數（多詞或多本來源）；`cooccur_bonus` 只在 `soft_and` 生效（預設 `0.2`）。
* `min` 若要做「嚴格 AND」，可在程式中過濾 `k < 查詢數` 的結果。
