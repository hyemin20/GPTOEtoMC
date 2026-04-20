import re
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity


def clean_response(text):
    if text is None:
        return None
    text = text.strip()
    if text.lower() in ["idk", "i don't know", "no idea", "??", "", "none"]:
        return None
    text = re.sub(r"[^\w\s.,'’-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 2 else None


t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tok = T5Tokenizer.from_pretrained("t5-base")

def canonicalize(text):
    inp = "paraphrase: " + text
    tokens = t5_tok(inp, return_tensors="pt", truncation=True)
    out = t5_model.generate(**tokens, max_length=48)
    return t5_tok.decode(out[0], skip_special_tokens=True)


embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def cluster_responses(responses, min_cluster_size=10):
    embeddings = embed_model.encode(responses, normalize_embeddings=True)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    return embeddings, labels


def representative_indices(embeddings, labels):
    reps = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        idxs = np.where(labels == cluster_id)[0]
        cluster_emb = embeddings[idxs]
        centroid = cluster_emb.mean(axis=0)
        sims = cosine_similarity([centroid], cluster_emb)[0]
        rep = idxs[np.argmax(sims)]
        reps[cluster_id] = rep
    return reps


def compute_weights(labels):
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    raw_counts = dict(zip(unique, counts))
    log_counts = {cid: np.log(1 + c) for cid, c in raw_counts.items()}
    total = sum(log_counts.values())
    weights = {cid: log_counts[cid] / total for cid in log_counts}
    return weights


def preprocess(raw_responses):
    cleaned = [clean_response(r) for r in raw_responses]
    cleaned = [r for r in cleaned if r is not None]
    canonical = [canonicalize(r) for r in cleaned]
    embeddings, labels = cluster_responses(canonical)
    reps_idx = representative_indices(embeddings, labels)
    reps = {cid: canonical[idx] for cid, idx in reps_idx.items()}
    weights = compute_weights(labels)
    return {
        "canonical_responses": canonical,
        "labels": labels,
        "cluster_representatives": reps,
        "cluster_weights": weights
    }
