import pandas as pd

# 讀取資料
df = pd.read_excel("data/raw/台大經濟系產學合作數據撈取_Itemlist_20250107.xlsx", sheet_name='Item list')

# 只保留書名、作者、書籍簡介、書籍分類第二層
df = df[['書名', '作者', '書籍簡介', '書籍分類第二層']]

# 刪除空值
df = df[df['書名'].notna()]

# 刪除書名重複
df = df.drop_duplicates(subset=['書名'])

# 存入 CSV 檔案
df.to_csv("data/process/processed_books.csv", index=False, encoding='utf-8-sig')

