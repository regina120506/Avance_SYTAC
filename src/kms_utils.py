"""Shared helpers for the Softtek KMS prototype: data loading and text cleaning."""
from pathlib import Path
import re

import pandas as pd
from nltk.stem import PorterStemmer
from rdflib import Graph
from rdflib.namespace import SKOS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CATEGORIES = [
    "Technical Documentation",
    "Policies & Procedures",
    "Project Reports",
    "Training Manuals",
    "Incident Reports",
]

ONTOLOGY_PATH = PROJECT_DIR / "ontology" / "softtek_ontology.ttl"

_stemmer = PorterStemmer()


def load_documents(path=DATA_DIR / "documents.csv"):
    """Load the labelled document collection (id, title, category, text)."""
    df = pd.read_csv(path)
    # The title carries useful signal, so it is indexed together with the body.
    df["content"] = df["title"] + ". " + df["text"]
    return df


def clean_text(text):
    """Lowercase, drop punctuation/digits and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Clean, remove stop words and reduce each word to its stem (payments -> payment)."""
    return [_stemmer.stem(w) for w in clean_text(text).split() if w not in ENGLISH_STOP_WORDS]


def load_synonyms(path=ONTOLOGY_PATH):
    """Map each skos:altLabel of a concept to its prefLabel plus its broader concept's prefLabel.

    e.g. "went down" -> "outage incident", "rules" -> "policy".
    """
    g = Graph().parse(path, format="turtle")
    synonyms = {}
    for concept, alt in g.subject_objects(SKOS.altLabel):
        terms = [str(g.value(concept, SKOS.prefLabel))]
        broader = g.value(concept, SKOS.broader)
        if broader is not None:
            terms.append(str(g.value(broader, SKOS.prefLabel)))
        synonyms[str(alt).lower()] = " ".join(terms)
    return synonyms


SYNONYMS = load_synonyms()


def expand_query(query):
    """Append ontology synonyms to a query so lexical TF-IDF can match related concepts."""
    q = clean_text(query)
    extra = [terms for phrase, terms in SYNONYMS.items() if re.search(rf"\b{phrase}\b", q)]
    return " ".join([query, *extra])
