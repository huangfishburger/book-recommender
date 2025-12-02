# Book_Recommender

## 0. Introduction

This project is a hybrid recommendation and retrieval system that combines BM25, cosine similarity, and LLM-generated tags to produce accurate terms-to-books and books-to-books recommendations and retrieval results. It also includes comprehensive tag-quality evaluation tools using BM25 and cosine similarity to help debug, analyze, and improve book recommendations.

**(Note on Authorship): This repository is a derivative copy of the original team project. The code represents the combined efforts of the development team listed below. My primary contributions focused on LLM tag generation, recommendation algorithm design, and cosine similarity–based relevance scoring.**

## 1. Clone the repository

```bash
git clone https://github.com/chienyilin/Book_Recommender.git
cd Book_Recommender
```

## 2. Environment setup

### Method 1: Using a Virtual Environment
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

### Method 2: Using Docker
```bash
# download docker desktop
https://www.docker.com/

# Build Docker image
docker build -t book_recommender .

# run container
docker run -it --rm book_recommender
```

## 3. Running the Project

### 3.1.1 Method 1: Rebuild / Update the Vector Index Locally
```bash
python -m app.build_index
```
* The path to the Excel file can be modified (default: data/process/title_tags.xlsx).
### 3.1.2 Method 2: Download the Pre-built Vector Index Folder
```bash
python data/code/downloadfiles.py
```

### 3.2 Search for Related Books by Book Title
```bash
python -m app.books_to_books
```
* Separate multiple input titles with a semicolon (;) or a full-width semicolon (；).
* Merge Modes: soft_and (default), sum, min, avg.

### 3.3 Search for Related Books by Keywords
```bash
python -m app.terms_to_books
```
* Separate multiple input terms with a comma (,) or a full-width comma (，).
* Merge Modes: soft_and (default), sum, min, avg.
* Both keywords included in the subject term list and those not included can be searched (only the program's search logic differs).
* exact_hits: The number of query terms themselves that are contained within the book's tags.
* similar_hits: The total count of all similar tags (excluding those exactly matching the term) that are contained within the book's tags.
  
### 3.4 Search for Similar Subject Terms by Keyword
```bash
python -m app.term_to_keywords
```
* Both keywords included in the subject term list and those not included can be searched (only the program's search logic differs).

### 3.5 Search for Related Subject Terms by Book Title
```bash
python -m app.book_to_keywords
```
* All books included in the subject term list can be searched.
* source_hits: Counts how many different source books (the queried books) hit at least one tag (either the book's own tag or an expanded similar tag) for a given book.
* match_count: Aggregates the total count of similar tag hits from all source books.
  
## Appendix: Comparison of Merge Modes (terms/books → books)

| Mode       | Scoring Formula (Simplified)  | Hit Requirement  | Characteristics and Use Case |
|------------|-------------------------------|------------------|-----------------------------|
| `soft_and` | `sum(scores) + cooccur_bonus * max(0, k - 1)`        | Not all required (favors co-occurrence)  | **Default.** Balances Recall and Precision. Adds a bonus to the sum of scores if a book is hit by multiple queries ($k > 1$), strongly prioritizing results that satisfy multiple conditions simultaneously. |
| `sum`      | `sum(scores)`                                        | Not all required (loose OR)   |Suitable for broad exploration and maximizing Recall. The final score is high if the book is highly ranked by any single query.|
| `min`      | `min(scores)`                                        | **Closer to all-required is better**        | Used for strict filtering. The final score is determined by the lowest score received from any single query. Requires the result to have reasonable relevance to all input queries. |
| `avg`      | `sum(scores) / #queries`                             | Not all required  | Used for fair comparison across different numbers of queries (e.g., comparing results from 2 terms vs. 5 terms). Normalizes the scores to a common scale. |

* Note: `k` is the number of queries (multiple terms or multiple source books) that hit the book; `cooccur_bonus` only applies to soft_and (default `0.2`).
* For a "Strict AND" in `min` mode, you can filter results where `k < number of queries` in the code.

## Author
This project was developed by Yu-Ting Huang and Chien-Yi Lin under the supervision of Professor Yu-Chang Chen.