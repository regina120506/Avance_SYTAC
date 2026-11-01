"""
Automated document classification prototype (Softtek KMS).

Steps:
  1. Load the sample documents (data/documents.csv).
  2. Vectorize the text with TF-IDF.
  3. Train two simple models: Multinomial Naive Bayes and Logistic Regression.
  4. Evaluate them and classify new, unlabelled documents into the predefined categories.

Usage:
  python src/classifier.py
  python src/classifier.py --text "The API returned 500 errors for an hour after the deploy"
"""
import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from kms_utils import CATEGORIES, DATA_DIR, OUTPUT_DIR, load_documents, tokenize

SEED = 42


def build_pipeline(model):
    """TF-IDF vectorizer + classifier in a single scikit-learn pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", model),
    ])


MODELS = {
    "Naive Bayes": lambda: MultinomialNB(alpha=0.1),
    "Logistic Regression": lambda: LogisticRegression(C=10, max_iter=1000),
}


def evaluate(df):
    """Hold-out evaluation + 5-fold cross-validation for each model."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["category"], test_size=0.25, stratify=df["category"], random_state=SEED)
    print(f"Training documents: {len(X_train)} | Test documents: {len(X_test)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, make in MODELS.items():
        pipe = build_pipeline(make()).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cv_scores = cross_val_score(build_pipeline(make()), df["content"], df["category"], cv=cv)
        results[name] = (acc, cv_scores.mean(), cv_scores.std())

        print("=" * 70)
        print(f"{name}: hold-out accuracy = {acc:.3f} | "
              f"5-fold CV accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print("=" * 70)
        print(classification_report(y_test, pred, labels=CATEGORIES, zero_division=0))

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, labels=CATEGORIES, ax=ax, cmap="Blues", colorbar=False,
            display_labels=[c.replace(" ", "\n", 1) for c in CATEGORIES])
        ax.set_title(f"Confusion matrix - {name} (test set)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        out = OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {out.relative_to(OUTPUT_DIR.parent)}\n")
    return results


def top_terms(pipe, n=6):
    """Most influential TF-IDF terms per category (Logistic Regression weights)."""
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    print("Most discriminative terms per category:")
    for i, cat in enumerate(clf.classes_):
        best = clf.coef_[i].argsort()[::-1][:n]
        print(f"  {cat:<25} {', '.join(vocab[j] for j in best)}")
    print()


def classify(pipe, texts):
    """Return (predicted category, confidence) for each text."""
    probs = pipe.predict_proba(texts)
    classes = pipe.classes_
    return [(classes[p.argmax()], p.max()) for p in probs]


def main():
    parser = argparse.ArgumentParser(description="Softtek KMS document classifier")
    parser.add_argument("--text", help="classify a single piece of text")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_documents()
    print(f"Loaded {len(df)} documents in {df['category'].nunique()} categories:")
    print(df["category"].value_counts().to_string(), "\n")

    evaluate(df)

    # Final model: Logistic Regression trained on the full collection.
    final = build_pipeline(MODELS["Logistic Regression"]()).fit(df["content"], df["category"])
    joblib.dump(final, OUTPUT_DIR / "classifier.joblib")
    top_terms(final)

    if args.text:
        cat, conf = classify(final, [args.text])[0]
        print(f"Text: {args.text}\n-> {cat} (confidence {conf:.2f})")
        return

    new = pd.read_csv(DATA_DIR / "new_documents.csv")
    print("Classifying new, unlabelled documents (data/new_documents.csv):")
    rows = []
    for (_, doc), (cat, conf) in zip(new.iterrows(), classify(final, new["text"])):
        print(f"  [{doc['id']}] {doc['text'][:70]}...\n        -> {cat} (confidence {conf:.2f})")
        rows.append({"id": doc["id"], "text": doc["text"], "predicted_category": cat,
                     "confidence": round(conf, 3)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "new_documents_classified.csv", index=False)
    print("\nResults saved to outputs/new_documents_classified.csv")


if __name__ == "__main__":
    main()
