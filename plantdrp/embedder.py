"""
plantdrp.embedder
=================
ProtT5-XL-U50 embedding generator for inference.

This module mirrors the embedding logic in scripts/generate_embeddings.py.
Both must stay in sync — same MODEL_NAME, WINDOW_SIZE, and STRIDE.

The key difference from the training script:
    - No CSV reading or writing
    - Returns numpy arrays directly in memory
    - Designed to be called sequence by sequence during prediction
"""

import re
import time
import warnings

import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel

warnings.filterwarnings("ignore")


# CONSTANTS — must match generate_embeddings.py


MODEL_NAME  = "Rostlab/prot_t5_xl_uniref50"
WINDOW_SIZE = 1000   # tokens per chunk
STRIDE      = 500    # step between chunks



# EMBEDDER CLASS


class Embedder:
    """
    Loads ProtT5-XL-U50 and generates per-sequence embeddings.

    Parameters
    ----------
    device : str
        "cpu", "cuda", or "mps" (Apple Silicon).
        Default is "auto" — detects GPU automatically.

    Example
    -------
        embedder = Embedder(device="auto")
        embedding = embedder.embed_sequence("MAKTLVLKGTFE...")
        # embedding.shape == (1024,)
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.tokenizer, self.model = self._load_model()

    #  Device 

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                d = torch.device("cuda")
            elif torch.backends.mps.is_available():
                d = torch.device("mps")
            else:
                d = torch.device("cpu")
        else:
            d = torch.device(device)
        print(f"[Embedder] Device: {d}")
        return d

    #  Model loading 

    def _load_model(self):
        """
        Load ProtT5 tokenizer and encoder.
        Uses float16 on GPU to halve memory (~3 GB → ~1.5 GB).
        Keeps float32 on CPU.
        """
        print(f"[Embedder] Loading {MODEL_NAME} ...")
        print("[Embedder] This may take a moment on first run "
              "(~3 GB download if not cached).")

        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_NAME, do_lower_case=False
        )

        if self.device.type == "cuda":
            model = T5EncoderModel.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16
            )
        else:
            model = T5EncoderModel.from_pretrained(MODEL_NAME)

        model = model.to(self.device).eval()
        dtype = next(model.parameters()).dtype
        print(f"[Embedder] Model ready  (dtype: {dtype})")
        return tokenizer, model

    #  Sequence validation ─

    @staticmethod
    def validate_sequence(seq_id: str, seq: str):
        """
        Clean and validate an amino acid sequence.
        - Converts to uppercase
        - Removes stop codons (*)
        - Warns about and removes unknown characters
        - Ambiguous residues (X, B, Z, U, O, J) are kept
          (ProtT5 handles them)

        Returns
        -------
        cleaned_seq : str
        is_valid    : bool  (False if sequence is empty after cleaning)
        """
        VALID    = set("ACDEFGHIKLMNPQRSTVWY")
        AMBIG    = set("XUBZOJ")

        seq = seq.upper().strip()
        cleaned = []
        unknown = []

        for ch in seq:
            if ch in VALID or ch in AMBIG:
                cleaned.append(ch)
            elif ch == "*":
                pass   # stop codon — strip silently
            else:
                unknown.append(ch)

        if unknown:
            bad = "".join(sorted(set(unknown)))
            print(f"  [WARN] {seq_id}: removed unknown character(s): '{bad}'")

        cleaned_seq = "".join(cleaned)

        if len(cleaned_seq) < 50:
            print(f"  [WARN] {seq_id}: sequence length {len(cleaned_seq)} "
                  f"is below minimum (50 aa) — prediction may be unreliable")

        return cleaned_seq, len(cleaned_seq) > 0

    #  Core embedding logic 

    @staticmethod
    def _format_sequence(seq: str) -> str:
        """Insert spaces between residues as required by ProtT5."""
        return " ".join(list(seq))

    def embed_sequence(self, seq: str) -> np.ndarray:
        """
        Generate a 1024-dim embedding for a single amino acid sequence.

        Strategy (mirrors generate_embeddings.py)
        ------------------------------------------
        1. Space-separate residues: "MAKT" → "M A K T"
        2. Split into overlapping windows of WINDOW_SIZE tokens
        3. Embed each window → mean-pool over sequence positions
        4. Average all chunk embeddings → 1024-dim vector

        Parameters
        ----------
        seq : str
            Raw amino acid sequence (no spaces needed).

        Returns
        -------
        np.ndarray, shape (1024,), dtype float32
        """
        tokens = self._format_sequence(seq).split()
        chunk_embeddings = []
        use_amp = (self.device.type == "cuda")

        for start in range(0, len(tokens), STRIDE):
            chunk_tokens = tokens[start : start + WINDOW_SIZE]
            if not chunk_tokens:
                break

            chunk_str = " ".join(chunk_tokens)

            with torch.no_grad():
                inputs = self.tokenizer(
                    chunk_str,
                    return_tensors="pt",
                    truncation=False,
                    padding=False,
                    add_special_tokens=True,
                )
                input_ids      = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                if use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                hidden = outputs.last_hidden_state
                mask   = attention_mask.unsqueeze(-1).float()
                pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
                chunk_embeddings.append(pooled.squeeze(0).cpu())

            # stop if this chunk already covers the end
            if start + WINDOW_SIZE >= len(tokens):
                break

        final_emb = torch.stack(chunk_embeddings).mean(dim=0)
        return final_emb.numpy().astype(np.float32)

    #  Batch embedding ─

    def embed_sequences(
        self,
        sequences: list,
        seq_ids:   list = None,
        verbose:   bool = True
    ) -> list:
        """
        Embed a list of sequences.

        Parameters
        ----------
        sequences : list of str
            Raw amino acid sequences.
        seq_ids   : list of str, optional
            Sequence IDs for logging. If None, uses index numbers.
        verbose   : bool
            Whether to print per-sequence timing.

        Returns
        -------
        list of np.ndarray, each shape (1024,)
        Sequences that fail are returned as None and logged.
        """
        if seq_ids is None:
            seq_ids = [str(i) for i in range(len(sequences))]

        embeddings = []
        total = len(sequences)

        for i, (seq_id, seq) in enumerate(zip(seq_ids, sequences), start=1):
            if verbose:
                print(f"  [{i}/{total}] Embedding {seq_id} "
                      f"(len={len(seq)}) ...", end=" ", flush=True)
            try:
                t0  = time.time()
                emb = self.embed_sequence(seq)
                elapsed = time.time() - t0
                if verbose:
                    print(f"done ({elapsed:.1f}s)")
                embeddings.append(emb)
            except Exception as e:
                print(f"\n  [ERROR] {seq_id}: {e} — skipped, returning None")
                embeddings.append(None)

        return embeddings
