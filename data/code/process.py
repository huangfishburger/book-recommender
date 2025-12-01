import pandas as pd

# read data
df = pd.read_excel("data/raw/台大經濟系產學合作數據撈取_Itemlist_20250107.xlsx", sheet_name='Item list')

# Retain only the title, author, summary, and the type.
df = df[['title', 'author', 'summary', 'type']]

df = df[df['title'].notna()]
df = df.drop_duplicates(subset=['title'])
df.to_csv("data/process/processed_books.csv", index=False, encoding='utf-8-sig')

