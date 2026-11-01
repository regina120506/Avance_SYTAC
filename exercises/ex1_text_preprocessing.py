"""Exercise 1 (Topics 1-2): text preprocessing - tokenization, stop words and stemming."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kms_utils import clean_text, load_documents, tokenize  # noqa: E402

df = load_documents()
doc = df.loc[df["id"] == "IR01", "content"].iloc[0]

raw_tokens = doc.split()
clean_tokens = clean_text(doc).split()
final_tokens = tokenize(doc)
print("Original text:\n ", doc, "\n")
print(f"1) Whitespace tokens ({len(raw_tokens)}):", raw_tokens[:12], "...")
print(f"2) Cleaned tokens   ({len(clean_tokens)}):", clean_tokens[:12], "...")
print(f"3) Stop words removed + Porter stemming ({len(final_tokens)}):", final_tokens[:12], "...\n")

all_raw = Counter(w for t in df["content"] for w in clean_text(t).split())
all_final = Counter(w for t in df["content"] for w in tokenize(t))
print(f"Vocabulary of the 60-document collection: {len(all_raw)} words -> "
      f"{len(all_final)} stems ({1 - len(all_final) / len(all_raw):.0%} smaller)")
print("Top 10 stems:", all_final.most_common(10))
