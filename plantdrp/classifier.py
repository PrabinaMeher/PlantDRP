"""
plantdrp.classifier
===================
Loads a trained SVM pipeline (scaler + SVM bundled together)
and runs inference on ProtT5 embeddings.

The model file is a pickle dict saved by scripts/train_plantdrp.py:
    {
        "model":        Pipeline(StandardScaler, SVC),
        "best_params":  {...},
        "dataset_name": "DS1" or "DS2"
    }

The Pipeline handles scaling internally — no separate
scaler file is needed.
"""

import pickle
import os
import numpy as np
from pathlib import Path



# CLASSIFIER CLASS


class SVMClassifier:
    """
    Wraps the trained SVM Pipeline for inference.

    Parameters
    ----------
    model_path : str or Path
        Path to the .pkl file saved by train_plantdrp.py

    Example
    -------
        clf = SVMClassifier("plantdrp/models/svm_ds2.pkl")
        prob, label = clf.predict_one(embedding)
        print(prob, label)   # 0.923, "DR"
    """

    def __init__(self, model_path):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Make sure svm_ds1.pkl and svm_ds2.pkl are inside "
                f"plantdrp/models/"
            )

        self.model_path   = model_path
        self.pipeline,    \
        self.best_params, \
        self.dataset_name = self._load(model_path)

        print(f"[Classifier] Loaded: {model_path.name}  "
              f"(dataset: {self.dataset_name})")

    # ── Loading ───────────────────────────────────

    @staticmethod
    def _load(path: Path):
        """
        Load the pkl file and extract components.
        Handles two formats:
          1. dict  → saved by train_plantdrp.py  (expected)
          2. plain Pipeline → saved directly with joblib/pickle
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Format 1 — dict (what train_plantdrp.py saves)
        if isinstance(payload, dict):
            pipeline     = payload["model"]
            best_params  = payload.get("best_params",  {})
            dataset_name = payload.get("dataset_name", "Unknown")

        # Format 2 — raw Pipeline object
        else:
            pipeline     = payload
            best_params  = {}
            dataset_name = "Unknown"

        # Sanity check — make sure it has predict_proba
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError(
                f"{path.name} does not contain a valid sklearn Pipeline. "
                f"Make sure you saved the correct model file."
            )

        return pipeline, best_params, dataset_name

    # ── Single sequence prediction ────────────────

    def predict_one(
        self,
        embedding:  np.ndarray,
        threshold:  float = 0.5
    ):
        """
        Predict for a single embedding vector.

        Parameters
        ----------
        embedding  : np.ndarray, shape (1024,)
            ProtT5 embedding for one sequence.
        threshold  : float
            Probability cutoff. Default 0.5.
            Sequences with P(DR) >= threshold → labelled "DR".

        Returns
        -------
        probability : float   P(DR protein), range 0–1
        label       : str     "DR" or "Non-DR"
        """
        if embedding is None:
            return None, "Error"

        # reshape to (1, 1024) — sklearn expects 2D input
        X    = embedding.reshape(1, -1)
        prob = float(self.pipeline.predict_proba(X)[0][1])
        label = "DR" if prob >= threshold else "Non-DR"

        return round(prob, 4), label

    # ── Batch prediction ──────────────────────────

    def predict_batch(
        self,
        embeddings: list,
        threshold:  float = 0.5
    ):
        """
        Predict for a list of embedding vectors.

        Parameters
        ----------
        embeddings : list of np.ndarray
            Each element is a (1024,) embedding or None (failed embed).
        threshold  : float
            Probability cutoff. Default 0.5.

        Returns
        -------
        list of (probability, label) tuples
        Failed embeddings (None) return (None, "Error").
        """
        results = []
        for emb in embeddings:
            prob, label = self.predict_one(emb, threshold)
            results.append((prob, label))
        return results

    # ── Info ──────────────────────────────────────

    def __repr__(self):
        return (
            f"SVMClassifier("
            f"dataset='{self.dataset_name}', "
            f"params={self.best_params})"
        )
