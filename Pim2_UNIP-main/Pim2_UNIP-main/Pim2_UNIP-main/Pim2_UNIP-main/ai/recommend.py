# ai/recommend.py
from typing import List, Dict, Sequence, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """
    Recomendação conteúdo-baseado com TF-IDF.
    materials: [{'id','titulo','descricao', ...}]
    """

    def __init__(
        self,
        materials: Sequence[Dict],
        text_fields: Sequence[str] = ("titulo", "descricao"),
        ngram_range: tuple = (1, 2),
        max_features: Optional[int] = None,
        lowercase: bool = True,
        stop_words: Optional[Sequence[str]] = None,
    ) -> None:
        self.text_fields = list(text_fields)
        self.materials: List[Dict] = list(materials)
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
        )
        self._fit()

    def _mat_to_text(self, m: Dict) -> str:
        return " ".join(str(m.get(f, "")) for f in self.text_fields)

    def _fit(self) -> None:
        self.texts = [self._mat_to_text(m) for m in self.materials]
        self.X = self.vectorizer.fit_transform(self.texts)
        self.id_to_idx = {m["id"]: i for i, m in enumerate(self.materials)}

    def rebuild(self, materials: Sequence[Dict]) -> None:
        self.materials = list(materials)
        self._fit()

    def recommend(self, viewed_ids: Sequence[str], top_k: int = 5) -> List[Dict]:
        """Retorna recomendações como [{'id','titulo','score'}]."""
        seen_vecs = [self.X[self.id_to_idx[vid]] for vid in viewed_ids if vid in self.id_to_idx]
        if not seen_vecs:
            return []
        profile = np.asarray(seen_vecs).mean(axis=0)
        sims = cosine_similarity(profile, self.X).flatten()
        seen = set(viewed_ids)
        order = [i for i in sims.argsort()[::-1] if self.materials[i]["id"] not in seen][:top_k]
        results = []
        for i in order:
            m = self.materials[i]
            results.append({"id": m["id"], "titulo": m.get("titulo", ""), "score": float(sims[i])})
        return results


# Exemplo: python -m ai.recommend
if __name__ == "__main__":
    mats = [
        {"id": "a1", "titulo": "Estruturas de Dados", "descricao": "Listas e árvores"},
        {"id": "a2", "titulo": "Exercícios Avançados em C", "descricao": "Ponteiros e funções"},
        {"id": "a3", "titulo": "Guia de Funções em Python", "descricao": "Comparativo"},
    ]
    rec = ContentRecommender(mats)
    print(rec.recommend(viewed_ids=["a2"], top_k=2))