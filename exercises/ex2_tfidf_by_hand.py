"""Exercise 2 (Topic 3): computing TF-IDF by hand and checking it against scikit-learn."""
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "server outage server restart",
    "security policy for passwords",
    "outage report and security review",
]
tokens = [d.split() for d in docs]
vocab = sorted({w for t in tokens for w in t})
N = len(docs)

# scikit-learn's smoothed formula: idf = ln((1 + N) / (1 + df)) + 1, then L2-normalize each row.
idf = {w: math.log((1 + N) / (1 + sum(w in t for t in tokens))) + 1 for w in vocab}
manual = np.array([[t.count(w) * idf[w] for w in vocab] for t in tokens])
manual = manual / np.linalg.norm(manual, axis=1, keepdims=True)

vec = TfidfVectorizer(token_pattern=r"\S+")
sk = vec.fit_transform(docs).toarray()
assert list(vec.get_feature_names_out()) == vocab

print(f"{'term':<10}{'df':>4}{'idf':>8}   tf-idf in doc1 / doc2 / doc3 (manual)")
for j, w in enumerate(vocab):
    df_ = sum(w in t for t in tokens)
    print(f"{w:<10}{df_:>4}{idf[w]:>8.3f}   " + " / ".join(f"{manual[i, j]:.3f}" for i in range(N)))
print("\nManual result equals scikit-learn:", np.allclose(manual, sk))
j = vocab.index("server")
print(f"Note: 'server' appears twice in doc1 and only there, so it gets the highest weight ({manual[0, j]:.3f}).")
