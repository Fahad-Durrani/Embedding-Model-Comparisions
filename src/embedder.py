"""Model wrappers that apply each model's own prefix convention and return
L2-normalized embeddings, with encode-timing captured for the latency report.

Two backends share one interface (`load`, `encode_passages`, `encode_queries`):
  * Embedder      -> sentence-transformers models (all-MiniLM, e5, F2LLM)
  * LiteRTEmbedder -> on-device TFLite/LiteRT models (Gecko) run through a TFLite
                      interpreter with a SentencePiece tokenizer

The prefix handling is the crux of a *fair* comparison:
  * all-MiniLM-L6-v2  -> no prefix
  * multilingual-e5-small -> "query: " on queries, "passage: " on passages
Skipping the e5 prefixes silently degrades its quality, so we never let the
caller forget them: the wrapper takes them from the model config.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EncodeTiming:
    corpus_seconds: float = 0.0
    corpus_items: int = 0
    query_seconds: float = 0.0
    query_items: int = 0

    @property
    def corpus_per_item_ms(self) -> float:
        return 1000.0 * self.corpus_seconds / self.corpus_items if self.corpus_items else 0.0

    @property
    def query_per_item_ms(self) -> float:
        return 1000.0 * self.query_seconds / self.query_items if self.query_items else 0.0

    def as_dict(self) -> dict:
        return {
            "corpus_seconds": round(self.corpus_seconds, 4),
            "corpus_items": self.corpus_items,
            "corpus_per_item_ms": round(self.corpus_per_item_ms, 3),
            "query_seconds": round(self.query_seconds, 4),
            "query_items": self.query_items,
            "query_per_item_ms": round(self.query_per_item_ms, 3),
        }


class _BaseEmbedder:
    """Shared prefix + timing plumbing. Subclasses implement `load()` and
    `_encode(texts, prefix)` (which must return L2-normalized row vectors)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.timing = EncodeTiming()

    @property
    def key(self) -> str:
        return self.cfg["key"]

    @property
    def name(self) -> str:
        return self.cfg["name"]

    def load(self) -> "_BaseEmbedder":
        raise NotImplementedError

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        raise NotImplementedError

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        start = time.perf_counter()
        emb = self._encode(texts, self.cfg["passage_prefix"])
        self.timing.corpus_seconds += time.perf_counter() - start
        self.timing.corpus_items += len(texts)
        return emb

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        start = time.perf_counter()
        emb = self._encode(texts, self.cfg["query_prefix"])
        self.timing.query_seconds += time.perf_counter() - start
        self.timing.query_items += len(texts)
        return emb


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    """Row-wise unit-normalize so a dot product equals cosine similarity.
    Guards against divide-by-zero for any all-zero row."""
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return emb / norms


class Embedder(_BaseEmbedder):
    """Wraps a SentenceTransformer with fixed query/passage prefixes."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._model = None

    def load(self) -> "Embedder":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg["hf_id"])
        return self

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        prefixed = [f"{prefix}{t}" for t in texts]
        # normalize_embeddings=True => unit vectors, so a dot product is cosine.
        return self._model.encode(
            prefixed,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class LiteRTEmbedder(_BaseEmbedder):
    """Runs an on-device TFLite/LiteRT embedding model (Gecko) through a TFLite
    interpreter with a SentencePiece tokenizer.

    The exact input signature of the published Gecko .tflite files is not
    documented, so we introspect the interpreter's input/output tensors at load
    time and adapt: token ids go to the integer input, and an attention mask /
    segment-id input is fed only if the model declares one. The output tensor's
    trailing dimension defines the true embedding size (overwriting cfg["dim"]).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._interp = None
        self._sp = None
        self._seq_len = int(cfg.get("seq_len", 512))
        self._primary_in = None       # input detail for token ids
        self._mask_in = None          # optional attention-mask input detail
        self._segment_in = None       # optional segment/token-type input detail
        self._out_detail = None
        self._allocated_shape = None

    def load(self) -> "LiteRTEmbedder":
        if self._interp is not None:
            return self

        import sentencepiece as spm

        from model_store import ensure_local

        # TFLite interpreter: prefer the light LiteRT package, fall back to the
        # TensorFlow-bundled interpreter (reliable on Windows). Access it as an
        # attribute -- `from tensorflow.lite import Interpreter` fails under the
        # lazy module loader in TF 2.20+.
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter

        model_path = ensure_local(self.cfg["hf_id"], self.cfg["model_file"], "gecko")
        tok_path = ensure_local(self.cfg["hf_id"], self.cfg["tokenizer_file"], "gecko")

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(tok_path)

        self._interp = Interpreter(model_path=model_path)
        self._interp.allocate_tensors()

        in_details = self._interp.get_input_details()
        out_details = self._interp.get_output_details()
        print(f"    [{self.name}] tflite inputs:")
        for d in in_details:
            print(f"      - name={d['name']!r} shape={list(d['shape'])} dtype={np.dtype(d['dtype']).name}")
        print(f"    [{self.name}] tflite outputs:")
        for d in out_details:
            print(f"      - name={d['name']!r} shape={list(d['shape'])} dtype={np.dtype(d['dtype']).name}")

        # ---- map inputs by name + dtype -----------------------------------
        int_inputs = [d for d in in_details if np.issubdtype(np.dtype(d["dtype"]), np.integer)]

        def _find(details, *needles):
            for d in details:
                nm = (d["name"] or "").lower()
                if any(n in nm for n in needles):
                    return d
            return None

        self._primary_in = (
            _find(int_inputs, "input_ids", "ids", "token", "input")
            or (int_inputs[0] if int_inputs else in_details[0])
        )
        others = [d for d in in_details if d["index"] != self._primary_in["index"]]
        self._mask_in = _find(others, "mask", "attention")
        self._segment_in = _find(others, "segment", "type")

        # ---- sequence length from the primary input shape -----------------
        pin_shape = list(self._primary_in["shape"])
        if len(pin_shape) >= 2 and pin_shape[-1] and pin_shape[-1] > 0:
            self._seq_len = int(pin_shape[-1])

        # ---- output dimension: prefer a 2-D [1, D] pooled embedding -------
        self._out_detail = out_details[0]
        for d in out_details:
            if len(d["shape"]) == 2:
                self._out_detail = d
                break
        out_dim = int(list(self._out_detail["shape"])[-1])
        if out_dim > 0 and out_dim != self.cfg.get("dim"):
            print(f"    [{self.name}] output dim {out_dim} (config said {self.cfg.get('dim')}); using {out_dim}")
            self.cfg["dim"] = out_dim

        return self

    # ------------------------------------------------------------------ #
    def _prepare_ids(self, text: str, prefix: str) -> np.ndarray:
        ids = list(self._sp.EncodeAsIds(f"{prefix}{text}"))
        eos = self._sp.eos_id()
        if eos is not None and eos >= 0:
            # Gecko derives from a T5/GTR-style SentencePiece (EOS appended, no
            # BOS). VERIFY: flip this off if quality looks off. See plan risks.
            ids = ids[: self._seq_len - 1] + [eos]
        else:
            ids = ids[: self._seq_len]

        pad = self._sp.pad_id()
        pad = pad if (pad is not None and pad >= 0) else 0
        attn = [1] * len(ids)
        if len(ids) < self._seq_len:
            n_pad = self._seq_len - len(ids)
            ids = ids + [pad] * n_pad
            attn = attn + [0] * n_pad
        return np.asarray(ids, dtype=np.int64), np.asarray(attn, dtype=np.int64)

    def _feed(self, detail, values: np.ndarray) -> None:
        arr = values.reshape(1, -1).astype(np.dtype(detail["dtype"]))
        # Resize if the model declares a dynamic/mismatched shape.
        want = [1, arr.shape[1]]
        if list(detail["shape"]) != want:
            self._interp.resize_tensor_input(detail["index"], want)
            self._allocated_shape = None
        self._interp.set_tensor(detail["index"], arr)

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        assert self._interp is not None, "call load() first"
        vectors = []
        for t in texts:
            ids, attn = self._prepare_ids(t, prefix)

            self._feed(self._primary_in, ids)
            if self._mask_in is not None:
                self._feed(self._mask_in, attn)
            if self._segment_in is not None:
                self._feed(self._segment_in, np.zeros_like(ids))

            cur_shape = (1, len(ids))
            if self._allocated_shape != cur_shape:
                self._interp.allocate_tensors()
                self._allocated_shape = cur_shape
                # tensors were reallocated; set them again after allocation
                self._feed(self._primary_in, ids)
                if self._mask_in is not None:
                    self._feed(self._mask_in, attn)
                if self._segment_in is not None:
                    self._feed(self._segment_in, np.zeros_like(ids))

            self._interp.invoke()
            out = np.asarray(self._interp.get_tensor(self._out_detail["index"]))

            if out.ndim == 3:
                # token-level [1, seq, D] -> masked mean-pool over the sequence
                mask = attn.astype(np.float32)[: out.shape[1]]
                mask = mask.reshape(1, -1, 1)
                summed = (out * mask).sum(axis=1)
                denom = np.clip(mask.sum(axis=1), 1e-9, None)
                vec = (summed / denom)[0]
            else:
                vec = out.reshape(out.shape[0], -1)[0]
            vectors.append(vec)

        return _l2_normalize(np.vstack(vectors))


def make_embedder(cfg: dict) -> _BaseEmbedder:
    """Factory: build the right embedder for a model config's backend."""
    backend = cfg.get("backend", "sentence_transformers")
    try:
        cls = {"sentence_transformers": Embedder, "litert": LiteRTEmbedder}[backend]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r} for model {cfg.get('key')!r}")
    return cls(cfg)
