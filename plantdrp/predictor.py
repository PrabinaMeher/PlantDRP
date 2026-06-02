"""
plantdrp.predictor
==================
High-level prediction API.

Ties together:
    FASTA parsing → Embedder → SVMClassifier → pandas DataFrame

This is the main entry point for both the CLI and Streamlit UI.

Example
-------
    from plantdrp import Predictor

    pred = Predictor(model="ds2")
    df   = pred.predict("proteins.fasta")
    print(df)

    #            id  length_aa  probability prediction model
    # 0  AT1G12220.1       1294       0.9823         DR   DS2
    # 1  AT1G59124.1        868       0.1204     Non-DR   DS2
"""

import os
##import pkg_resources
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from .embedder   import Embedder
from .classifier import SVMClassifier


# ─
# HELPERS
# ─


   
def _model_path(filename: str) -> Path:
    """Resolve absolute path to a bundled model file."""
    return Path(__file__).parent / "models" / filename
    
    
def _parse_fasta(fasta_path: str):
    """
    Parse a FASTA file and return lists of ids, sequences, lengths.

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if no valid sequences are found
    """
    fasta_path = Path(fasta_path)

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}"
        )

    ids, seqs, lengths = [], [], []
    seen = set()

    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        seq_id = rec.id
        seq    = str(rec.seq).upper().strip()

        # skip duplicates
        if seq_id in seen:
            print(f"  [WARN] Duplicate ID '{seq_id}' — skipping")
            continue
        seen.add(seq_id)

        # skip empty
        if len(seq) == 0:
            print(f"  [WARN] '{seq_id}' is empty — skipping")
            continue

        ids.append(seq_id)
        seqs.append(seq)
        lengths.append(len(seq))

    if len(ids) == 0:
        raise ValueError(
            "No valid sequences found in the FASTA file. "
            "Check that the file is in correct FASTA format "
            "and contains protein (amino acid) sequences."
        )

    return ids, seqs, lengths


# ─
# PREDICTOR CLASS
# ─

class Predictor:
    """
    End-to-end disease resistance protein predictor.

    Parameters
    ----------
    model     : str
        Which SVM model to use. "ds1" or "ds2".
        Default is "ds2" (recommended — higher accuracy).
    device    : str
        Device for ProtT5 embedding. "auto", "cpu", "cuda", or "mps".
        Default is "auto" (detects GPU automatically).
    threshold : float
        Probability cutoff for DR label. Default 0.5.
        Sequences with P(DR) >= threshold → labelled "DR".

    Example
    -------
        pred = Predictor(model="ds2", device="auto", threshold=0.5)
        df   = pred.predict("proteins.fasta")
        df.to_csv("results.csv", index=False)
    """

    VALID_MODELS = ("ds1", "ds2")

    def __init__(
        self,
        model:     str   = "ds2",
        device:    str   = "auto",
        threshold: float = 0.5,
    ):
        if model.lower() not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. "
                f"Choose from: {self.VALID_MODELS}"
            )

        self.model_name = model.lower()
        self.threshold  = threshold

        # resolve pkl path
        pkl_file   = f"svm_{self.model_name}.pkl"
        model_file = _model_path(pkl_file)

        print(f"\n[PlantDRP] Model    : {self.model_name.upper()}")
        print(f"[PlantDRP] Threshold: {self.threshold}")
        print(f"[PlantDRP] PKL file : {model_file}\n")

        # load embedder and classifier
        self.embedder   = Embedder(device=device)
        self.classifier = SVMClassifier(model_file)

    #  Main prediction method 

    def predict(
        self,
        fasta_path: str,
        verbose:    bool = True,
    ) -> pd.DataFrame:
        """
        Run end-to-end prediction on a FASTA file.

        Parameters
        ----------
        fasta_path : str
            Path to input protein FASTA file.
        verbose    : bool
            Print per-sequence progress. Default True.

        Returns
        -------
        pd.DataFrame with columns:
            id          — sequence ID from FASTA header
            length_aa   — sequence length in amino acids
            probability — P(DR protein), range 0–1
            prediction  — "DR" or "Non-DR"
            model       — which model was used (DS1 or DS2)
        """
        #  Step 1: Parse FASTA 
        print(f"[PlantDRP] Parsing FASTA: {fasta_path}")
        ids, seqs, lengths = _parse_fasta(fasta_path)
        print(f"[PlantDRP] Found {len(ids)} sequence(s)\n")

        #  Step 2: Validate sequences ─
        clean_seqs = []
        for seq_id, seq in zip(ids, seqs):
            cleaned, is_valid = self.embedder.validate_sequence(
                seq_id, seq
            )
            clean_seqs.append(cleaned if is_valid else "")

        #  Step 3: Generate embeddings 
        print(f"\n[PlantDRP] Generating ProtT5 embeddings ...")
        embeddings = self.embedder.embed_sequences(
            sequences=clean_seqs,
            seq_ids=ids,
            verbose=verbose
        )

        #  Step 4: Run SVM inference 
        print(f"\n[PlantDRP] Running SVM inference ...")
        predictions = self.classifier.predict_batch(
            embeddings=embeddings,
            threshold=self.threshold
        )

        #  Step 5: Build results DataFrame 
        rows = []
        for seq_id, length, emb, (prob, label) in zip(
            ids, lengths, embeddings, predictions
        ):
            rows.append({
                "id":          seq_id,
                "length_aa":   length,
                "probability": prob,
                "prediction":  label,
                "model":       self.model_name.upper(),
            })

        df = pd.DataFrame(rows)

        # summary
        dr_count  = (df["prediction"] == "DR").sum()
        ndr_count = (df["prediction"] == "Non-DR").sum()
        err_count = (df["prediction"] == "Error").sum()

        print(f"\n[PlantDRP]  Results ")
        print(f"[PlantDRP]   Total sequences : {len(df)}")
        print(f"[PlantDRP]   DR proteins     : {dr_count}")
        print(f"[PlantDRP]   Non-DR proteins : {ndr_count}")
        if err_count > 0:
            print(f"[PlantDRP]   Failed          : {err_count}")
        print(f"[PlantDRP] \n")

        return df

    #  Convenience: predict from sequences directly 

    def predict_sequences(
        self,
        sequences: list,
        seq_ids:   list = None,
        verbose:   bool = True,
    ) -> pd.DataFrame:
        """
        Run prediction directly from a list of sequences
        without needing a FASTA file.

        Parameters
        ----------
        sequences : list of str
            Raw amino acid sequences.
        seq_ids   : list of str, optional
            Sequence IDs. If None, uses seq_1, seq_2, ...
        verbose   : bool
            Print per-sequence progress.

        Returns
        -------
        pd.DataFrame  (same columns as predict())

        Example
        -------
            pred = Predictor(model="ds2")
            df   = pred.predict_sequences(
                sequences=["MAKTLVLK...", "MSSDQQQLL..."],
                seq_ids=["protein_A", "protein_B"]
            )
        """
        if seq_ids is None:
            seq_ids = [f"seq_{i+1}" for i in range(len(sequences))]

        if len(sequences) != len(seq_ids):
            raise ValueError(
                "sequences and seq_ids must have the same length."
            )

        lengths    = [len(s) for s in sequences]
        embeddings = self.embedder.embed_sequences(
            sequences=sequences,
            seq_ids=seq_ids,
            verbose=verbose
        )
        predictions = self.classifier.predict_batch(
            embeddings=embeddings,
            threshold=self.threshold
        )

        rows = []
        for seq_id, length, (prob, label) in zip(
            seq_ids, lengths, predictions
        ):
            rows.append({
                "id":          seq_id,
                "length_aa":   length,
                "probability": prob,
                "prediction":  label,
                "model":       self.model_name.upper(),
            })

        return pd.DataFrame(rows)

    def __repr__(self):
        return (
            f"Predictor("
            f"model='{self.model_name}', "
            f"threshold={self.threshold})"
        )
