"""Exercise 5 (Topic 6): evaluating the search engine with relevance judgments (P@3, MRR)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kms_utils import load_documents  # noqa: E402
from semantic_search import SemanticSearch  # noqa: E402

# Manually judged relevant documents for each test query.
JUDGMENTS = {
    "how do I deploy a service to kubernetes": {"TD03", "TM03", "TD05"},
    "what happened when the payment system went down": {"IR01"},
    "rules for working from home": {"PP02"},
    "course to learn security and avoid phishing": {"TM12", "TM04"},
    "is the project over budget": {"PR10", "PR01", "PR03"},
    "login tokens and identity provider": {"TD06", "TD01"},
    "the website broke after a release": {"IR05", "IR08"},
    "who approves changes to production": {"PP07"},
}


def evaluate(engine, k=3):
    p_at_k, rr = [], []
    for query, relevant in JUDGMENTS.items():
        ranked = list(engine.search(query, top_k=10)["id"])
        p_at_k.append(sum(d in relevant for d in ranked[:k]) / k)
        first = next((i for i, d in enumerate(ranked, 1) if d in relevant), None)
        rr.append(1 / first if first else 0)
    return p_at_k, rr


df = load_documents()
results = {name: evaluate(SemanticSearch(df, baseline=b)) for name, b in
           [("Baseline TF-IDF", True), ("TF-IDF + stemming + ontology expansion", False)]}

print(f"{'Query':<50}{'RR base':>9}{'RR new':>9}")
base_rr, new_rr = results["Baseline TF-IDF"][1], results["TF-IDF + stemming + ontology expansion"][1]
for q, a, b in zip(JUDGMENTS, base_rr, new_rr):
    print(f"{q:<50}{a:>9.2f}{b:>9.2f}")
print()
for name, (p, r) in results.items():
    print(f"{name:<42} P@3 = {sum(p) / len(p):.3f}   MRR = {sum(r) / len(r):.3f}")
