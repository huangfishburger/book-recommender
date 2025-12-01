# bookrec/embeddings.py
from __future__ import annotations
import os, time, math
from tqdm.auto import tqdm
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 降低 OMP 衝突風險
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

class EmbeddingClient:
    def __init__(self, model: str = "text-embedding-3-small"):
        load_dotenv()
        self.model = model
        self.client = OpenAI()  # 讀 OPENAI_API_KEY

    def embed_texts(self, texts, batch_size=100, sleep=0.0, progress=True) -> np.ndarray:
        n = len(texts)
        out = np.empty((n, 1536), dtype=np.float32)
        iterator = range(0, n, batch_size)
        if progress:
            iterator = tqdm(iterator, total=math.ceil(n / batch_size),
                            desc="Embedding", unit="batch")
        for i in iterator:
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for j, rec in enumerate(resp.data):
                out[i + j] = rec.embedding
            if sleep:
                time.sleep(sleep)
        # L2 normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out / norms

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.client.embeddings.create(model=self.model, input=[text]).data[0].embedding
        vec = np.asarray(vec, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)
