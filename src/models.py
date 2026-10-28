"""M0-M5 estimators. Only question strings reach fit/predict.

Uses scikit-learn's Pipeline, FeatureUnion, TfidfVectorizer and LinearSVC.
This is a linear text-classification baseline, not a reproduction of
Joachims (1998)'s original experiments.
"""
import re
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from .common import config, normalize

# Written from intent definitions before inspecting held-out model errors.
# Order is fixed: prevention > diagnosis > symptoms > causes > treatment > basic.
# Multiple matches take the first category; no match falls back to train majority.
RULES = [
    ("Prevention", r"\b(prevent\w*|avoid|reduce (?:the )?risk)\b"),
    ("Diagnosis and tests", r"\b(diagnos\w*|tests?|testing|screen\w*|detect\w*)\b"),
    ("Symptoms", r"\b(symptoms?|signs?|manifestations?)\b"),
    ("Causes", r"\b(caus\w*|why|etiology)\b"),
    ("Treatment", r"\b(treat\w*|therap\w*|cure\w*|manag\w*|medications?)\b"),
    ("Basic information", r"^(what (?:is|are)|what is \(are\)|tell me about|do you have information)\b")
]


class RuleClassifier(ClassifierMixin, BaseEstimator):
    """Inspectable, ordered rules with training-only majority fallback."""
    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.fallback_ = self.classes_[counts.argmax()]
        return self

    def predict(self, X):
        predictions = []
        for text in X:
            match = next((label for label, pattern in RULES if re.search(pattern, normalize(text).lower())), self.fallback_)
            predictions.append(match)
        return np.asarray(predictions)


def build_model(model_id, c=1.0):
    cfg = config()
    if model_id == "M0":
        return DummyClassifier(strategy="most_frequent")
    if model_id == "M1":
        return RuleClassifier()
    word = TfidfVectorizer(lowercase=True, ngram_range=tuple(cfg["word_ngrams"]), min_df=1,
                           sublinear_tf=False, norm="l2", dtype=np.float64)
    if model_id in {"M3", "M5"}:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg["char_ngrams"]),
                               lowercase=True, min_df=1, sublinear_tf=False, norm="l2", dtype=np.float64)
        features = FeatureUnion([("word", word), ("char", char)])
    elif model_id in {"M2", "M4"}:
        features = word
    else:
        raise ValueError("Unknown model ID: " + model_id)
    return Pipeline([("features", features), ("classifier", LinearSVC(
        C=c, class_weight=None, dual="auto", max_iter=10000, tol=1e-4, random_state=cfg["seed"]))])
