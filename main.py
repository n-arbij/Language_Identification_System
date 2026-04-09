from __future__ import annotations

import argparse
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATASET_PATH = BASE_DIR / "language_dataset.csv"
MODEL_PATH = ARTIFACT_DIR / "language_identifier.joblib"
COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"

LANGUAGE_FILES = {
    "Swahili": BASE_DIR / "swahili.txt",
    "Sheng": BASE_DIR / "sheng.txt",
    "Luo": BASE_DIR / "luo.txt",
    "English": BASE_DIR / "english.txt",
}

SHENG_SLANG_MAP = {
    "bana": "bro",
    "beshte": "friend",
    "buda": "friend",
    "doh": "money",
    "fiti": "good",
    "lit": "good",
    "maze": "friend",
    "msee": "person",
    "poa": "good",
    "sawa": "okay",
    "sorted": "sorted",
    "susp": "suspicious",
    "sus": "suspicious",
    "vibe": "mood",
    "drip": "style",
    "sauce": "style",
    "hustle": "work",
}


@dataclass(frozen=True)
class TrainedBundle:
    model_name: str
    pipeline: Pipeline
    comparison: pd.DataFrame
    labels: List[str]


def load_text_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_dataset() -> pd.DataFrame:
    frames = []
    for language, path in LANGUAGE_FILES.items():
        texts = load_text_file(path)
        frames.append(pd.DataFrame({"text": texts, "language": language}))

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("")]
    df = df.dropna(subset=["text"])
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.split().str.len() >= 3]

    min_count = df["language"].value_counts().min()
    balanced_parts = [
        group.sample(n=min_count, random_state=42)
        for _, group in df.groupby("language", sort=False)
    ]
    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[a-z']+", text)
    return " ".join(tokens)


def normalize_sheng_text(text: str) -> str:
    tokens = clean_text(text).split()
    normalized = [SHENG_SLANG_MAP.get(token, token) for token in tokens]
    return " ".join(normalized)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["processed_text"] = cleaned["text"].map(clean_text)
    sheng_mask = cleaned["language"].eq("Sheng")
    cleaned.loc[sheng_mask, "processed_text"] = cleaned.loc[sheng_mask, "text"].map(normalize_sheng_text)
    cleaned = cleaned[cleaned["processed_text"].str.len() > 0].reset_index(drop=True)
    return cleaned


def build_model_pipelines() -> Dict[str, Pipeline]:
    return {
        "Naive Bayes (word TF-IDF)": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=5000,
                    ),
                ),
                ("model", MultinomialNB()),
            ]
        ),
        "Logistic Regression (char n-grams)": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=12000,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=3000, random_state=42),
                ),
            ]
        ),
        "Linear SVM (char n-grams)": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=12000,
                    ),
                ),
                ("model", LinearSVC()),
            ]
        ),
    }


def evaluate_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    pipelines = build_model_pipelines()
    results = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        fitted_pipelines[name] = pipeline

    comparison = pd.DataFrame(results).sort_values(
        by=["f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    return comparison, fitted_pipelines


def save_confusion_matrix(
    y_test: pd.Series,
    predictions: np.ndarray,
    labels: List[str],
) -> None:
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Language Identification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()


def train_system() -> TrainedBundle:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    dataset.to_csv(DATASET_PATH, index=False)

    processed = preprocess_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        processed["processed_text"],
        processed["language"],
        test_size=0.2,
        random_state=42,
        stratify=processed["language"],
    )

    comparison, fitted_pipelines = evaluate_models(x_train, x_test, y_train, y_test)
    comparison.to_csv(COMPARISON_PATH, index=False)

    best_model_name = comparison.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_predictions = best_pipeline.predict(x_test)

    save_confusion_matrix(y_test, best_predictions, sorted(processed["language"].unique()))

    bundle = TrainedBundle(
        model_name=best_model_name,
        pipeline=best_pipeline,
        comparison=comparison,
        labels=sorted(processed["language"].unique()),
    )

    joblib.dump(
        {
            "model_name": bundle.model_name,
            "pipeline": bundle.pipeline,
            "comparison": bundle.comparison,
            "labels": bundle.labels,
        },
        MODEL_PATH,
    )

    print("Balanced dataset saved to:", DATASET_PATH)
    print("Model comparison saved to:", COMPARISON_PATH)
    print("Confusion matrix saved to:", CONFUSION_MATRIX_PATH)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print("\nClassification report:")
    print(classification_report(y_test, best_predictions, zero_division=0))

    return bundle


def load_trained_bundle() -> TrainedBundle:
    if not MODEL_PATH.exists():
        return train_system()

    data = joblib.load(MODEL_PATH)
    return TrainedBundle(
        model_name=data["model_name"],
        pipeline=data["pipeline"],
        comparison=data["comparison"],
        labels=data["labels"],
    )


def predict_language(text: str, bundle: TrainedBundle | None = None) -> str:
    if bundle is None:
        bundle = load_trained_bundle()
    processed = clean_text(text)
    if not processed:
        raise ValueError("Please enter a non-empty text sample.")
    return bundle.pipeline.predict([processed])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the language identification system.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the models even if a saved model exists.")
    args = parser.parse_args()

    if args.retrain and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    train_system()


if __name__ == "__main__":
    main()