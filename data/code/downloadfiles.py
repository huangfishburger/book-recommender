import gdown
from pathlib import Path

# 專案根目錄（假設這個檔案一定在 data/code 底下）
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "data" / "process" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# .faiss
url1 = f"https://drive.google.com/uc?id=1cnFNuPksXLqBf6hdNIZOf2Ob5urI_wJF"
# .npy
url2 = f"https://drive.google.com/uc?id=1C7ehmyCFbc3d3NtlHnYQ7zaoCwH0qEXj"
# .json
url3 = f"https://drive.google.com/uc?id=17P-BUfDBXR3O1hP_GweD2wUMXuUhqQlI"

gdown.download(url1, str(ARTIFACTS_DIR / "tag_index.faiss"), quiet=False)
gdown.download(url2, str(ARTIFACTS_DIR / "tag_vecs.npy"), quiet=False)
gdown.download(url3, str(ARTIFACTS_DIR / "tag_cache.json"), quiet=False)
# gdown.download(url3, str(ARTIFACTS_DIR / "tag_cache.json"), quiet=False, fuzzy=True)
