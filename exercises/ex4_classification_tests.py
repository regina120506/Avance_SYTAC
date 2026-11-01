"""Exercise 4 (Topic 5): classification tests - preprocessing options x models, 5-fold CV."""
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kms_utils import clean_text, load_documents, tokenize  # noqa: E402

df = load_documents()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

vectorizers = {
    "Bag of words (counts)": lambda: CountVectorizer(preprocessor=clean_text, stop_words="english"),
    "TF-IDF unigrams": lambda: TfidfVectorizer(preprocessor=clean_text, stop_words="english"),
    "TF-IDF + stemming": lambda: TfidfVectorizer(tokenizer=tokenize, token_pattern=None),
    "TF-IDF + stemming + bigrams": lambda: TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                                           ngram_range=(1, 2), sublinear_tf=True),
}
models = {
    "Naive Bayes": lambda: MultinomialNB(alpha=0.1),
    "Log. Regression": lambda: LogisticRegression(C=10, max_iter=1000),
    "Linear SVM": lambda: LinearSVC(),
}

print(f"{'Representation':<30}" + "".join(f"{m:>18}" for m in models))
for vname, make_vec in vectorizers.items():
    scores = [cross_val_score(make_pipeline(make_vec(), make_model()), df["content"], df["category"], cv=cv)
              for make_model in models.values()]
    print(f"{vname:<30}" + "".join(f"{s.mean():>12.3f} ±{s.std():.2f}" for s in scores))
print("\nValues = mean accuracy over 5 folds (60 documents, 5 balanced categories).")
