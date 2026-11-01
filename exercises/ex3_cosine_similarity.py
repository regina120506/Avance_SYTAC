"""Exercise 3 (Topic 4): document-to-document cosine similarity on the collection."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kms_utils import CATEGORIES, OUTPUT_DIR, load_documents, tokenize  # noqa: E402

df = load_documents()
X = TfidfVectorizer(tokenizer=tokenize, token_pattern=None).fit_transform(df["content"])
S = cosine_similarity(X)
np.fill_diagonal(S, 0)

print("Five most similar document pairs:")
pairs = sorted(((S[i, j], i, j) for i in range(len(df)) for j in range(i + 1, len(df))), reverse=True)
for s, i, j in pairs[:5]:
    print(f"  {s:.3f}  {df.id[i]} {df.title[i]}  <->  {df.id[j]} {df.title[j]}")

# Average similarity within and between categories: documents of the same type should be closer.
cats = df["category"].to_numpy()
S_off = np.where(np.eye(len(df), dtype=bool), np.nan, S)  # ignore self-similarity
M = np.array([[np.nanmean(S_off[np.ix_(cats == a, cats == b)]) for b in CATEGORIES] for a in CATEGORIES])
print("\nMean cosine similarity between categories (diagonal = within category):")
print(" " * 25 + "".join(f"{c[:10]:>12}" for c in CATEGORIES))
for a, row in zip(CATEGORIES, M):
    print(f"{a:<25}" + "".join(f"{v:>12.3f}" for v in row))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(S, cmap="Blues")
ticks = np.arange(6, 60, 12)
ax.set_xticks(ticks, [c.split()[0] for c in CATEGORIES], rotation=30)
ax.set_yticks(ticks, [c.split()[0] for c in CATEGORIES])
ax.set_title("Cosine similarity between the 60 documents")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "similarity_heatmap.png", dpi=150)
print("\nHeatmap saved to outputs/similarity_heatmap.png")
