"""
Basic semantic search component (Softtek KMS).

Steps:
  1. Receive a natural-language query.
  2. Convert the query into a TF-IDF vector (same vocabulary as the documents).
     Words are stemmed and the query is expanded with ontology synonyms.
  3. Compute cosine similarity between the query and every document.
  4. Display the most relevant documents ranked by similarity.

Usage:
  python src/semantic_search.py "how do I deploy a service to kubernetes"
  python src/semantic_search.py            # interactive mode
  python src/semantic_search.py --demo     # run the example queries used in the report
  python src/semantic_search.py --demo --baseline   # same, without stemming / expansion
"""
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kms_utils import clean_text, expand_query, load_documents, tokenize

DEMO_QUERIES = [
    "how do I deploy a service to kubernetes",
    "what happened when the payment system went down",
    "rules for working from home",
    "course to learn security and avoid phishing",
    "is the project over budget",
]


class SemanticSearch:
    def __init__(self, df, baseline=False):
        self.df = df.reset_index(drop=True)
        self.baseline = baseline
        if baseline:  # plain TF-IDF: no stemming, no query expansion
            self.vectorizer = TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                                              ngram_range=(1, 2), sublinear_tf=True)
        else:
            self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                              ngram_range=(1, 2), sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(self.df["content"])

    def prepare(self, query):
        return query if self.baseline else expand_query(query)

    def search(self, query, top_k=5, min_score=0.0):
        query_vec = self.vectorizer.transform([self.prepare(query)])
        scores = cosine_similarity(query_vec, self.doc_matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        results = self.df.loc[ranked, ["id", "title", "category"]].copy()
        results["similarity"] = scores[ranked].round(3)
        return results[results["similarity"] > min_score]

    def query_terms(self, query):
        """Terms of the query that exist in the vocabulary (shows what the search 'understood')."""
        vec = self.vectorizer.transform([self.prepare(query)])
        vocab = self.vectorizer.get_feature_names_out()
        return [vocab[i] for i in vec.nonzero()[1]]


def show(engine, query, top_k):
    print(f'\nQuery: "{query}"')
    print(f"Matched terms: {', '.join(engine.query_terms(query)) or '(none)'}")
    results = engine.search(query, top_k=top_k)
    if results.empty:
        print("  No relevant documents found.")
        return
    for rank, row in enumerate(results.itertuples(), 1):
        print(f"  {rank}. [{row.similarity:.3f}] {row.id} - {row.title} ({row.category})")


def main():
    parser = argparse.ArgumentParser(description="Softtek KMS semantic search")
    parser.add_argument("query", nargs="?", help="natural-language query")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--demo", action="store_true", help="run the example queries")
    parser.add_argument("--baseline", action="store_true", help="disable stemming and query expansion")
    args = parser.parse_args()

    engine = SemanticSearch(load_documents(), baseline=args.baseline)
    print(f"Indexed {engine.doc_matrix.shape[0]} documents, "
          f"vocabulary of {engine.doc_matrix.shape[1]} TF-IDF terms.")

    if args.demo:
        for q in DEMO_QUERIES:
            show(engine, q, args.top_k)
    elif args.query:
        show(engine, args.query, args.top_k)
    else:
        print("Type a question (empty line to exit).")
        while (q := input("\nSearch> ").strip()):
            show(engine, q, args.top_k)


if __name__ == "__main__":
    main()
