# ai/similarity.py
from typing import List, Dict, Tuple, Literal, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi  # opcional
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False


def _tfidf_top(submissions: List[Dict], target_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
    corpus = [s["texto"] for s in submissions]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[target_idx], X).flatten()
    order = sims.argsort()[::-1]
    out = []
    for j in order:
        if j == target_idx:
            continue
        out.append((submissions[j]["id"], float(sims[j])))
        if len(out) >= top_k:
            break
    return out


def _bm25_top(submissions: List[Dict], target_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
    if not _HAS_BM25:
        raise RuntimeError("rank-bm25 não está instalado. Use método 'tfidf' ou instale rank-bm25.")
    # tokenização simples (espaço)
    corpus_tokens = [s["texto"].lower().split() for s in submissions]
    bm25 = BM25Okapi(corpus_tokens)
    query = corpus_tokens[target_idx]
    scores = bm25.get_scores(query)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for j in order:
        if j == target_idx:
            continue
        out.append((submissions[j]["id"], float(scores[j])))
        if len(out) >= top_k:
            break
    return out


def top_similar_submissions(
    submissions: List[Dict],  # [{'id','aluno','texto'}]
    target_id: str,
    top_k: int = 5,
    method: Literal["tfidf", "bm25"] = "tfidf",
) -> List[Tuple[str, float]]:
    """Retorna [(id_semelhante, score), ...]."""
    idx_map = {s["id"]: i for i, s in enumerate(submissions)}
    if target_id not in idx_map:
        return []
    i = idx_map[target_id]
    if method == "bm25":
        return _bm25_top(submissions, i, top_k=top_k)
    return _tfidf_top(submissions, i, top_k=top_k)


# Exemplo: python -m ai.similarity
if __name__ == "__main__":
    subs = [
        {"id": "t1", "aluno": "Ana", "texto": "redes neurais e aprendizado supervisionado"},
        {"id": "t2", "aluno": "Bruno", "texto": "aprendizado supervisionado com regressao logistica"},
        {"id": "t3", "aluno": "Carla", "texto": "introducao a redes neurais convolucionais"},
        {"id": "t4", "aluno": "Davi", "texto": "sistemas distribuídos e redes de computadores"},
    ]
    print("TF-IDF:", top_similar_submissions(subs, target_id="t1", method="tfidf"))
    try:
        print("BM25:", top_similar_submissions(subs, target_id="t1", method="bm25"))
    except Exception as e:
        print("BM25 Indisponível")