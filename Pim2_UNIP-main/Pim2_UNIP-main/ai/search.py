# ai/search.py
from typing import List, Dict, Sequence, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    """
    Busca semântica sobre uma coleção de documentos usando TF-IDF + similaridade do cosseno.

    docs: lista de dicionários com pelo menos 'id' e campos textuais
          ex.: {'id': 'mat-001', 'titulo': 'Apostila', 'descricao': '...', 'conteudo': '...'}
    """

    def __init__(
        self,
        docs: Sequence[Dict],
        text_fields: Sequence[str] = ("titulo", "descricao", "conteudo"),
        ngram_range: tuple = (1, 2),
        max_features: Optional[int] = None,
        lowercase: bool = True,
        stop_words: Optional[Sequence[str]] = None,  # passe uma lista se quiser stopwords pt-BR
    ) -> None:
        self.text_fields = list(text_fields)
        self.lowercase = lowercase
        self.docs: List[Dict] = list(docs)
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
        )
        self._fit()

    # ---------- internos ----------
    def _doc_to_text(self, d: Dict) -> str:
        parts = [str(d.get(f, "")) for f in self.text_fields]
        return " ".join(parts)

    def _corpus(self) -> List[str]:
        return [self._doc_to_text(d) for d in self.docs]

    def _fit(self) -> None:
        corpus = self._corpus()
        self.matrix = self.vectorizer.fit_transform(corpus)

    # ---------- API ----------
    def add_docs(self, new_docs: Sequence[Dict], rebuild: bool = True) -> None:
        """Adiciona documentos e (opcionalmente) reconstrói o índice."""
        self.docs.extend(list(new_docs))
        if rebuild:
            self._fit()

    def rebuild(self) -> None:
        """Reconstrói o índice a partir de self.docs."""
        self._fit()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retorna top_k resultados como [{'id', 'score', 'titulo'}]."""
        if not self.docs:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        order = sims.argsort()[::-1][:top_k]
        results = []
        for i in order:
            d = self.docs[i]
            results.append(
                {
                    "id": d.get("id", i),
                    "titulo": d.get("titulo", ""),
                    "score": float(sims[i]),
                }
            )
        return results


# Exemplo rápido (execução direta): python -m ai.search
if __name__ == "__main__":
    documentos = [
        {"id": "m1", "titulo": "Prova 1 – Programação em C", "descricao": "funções"},
        {"id": "m2", "titulo": "Apostila: Introdução ao C"},
        {"id": "m3", "titulo": "Exercícios de Funções em C"},
        {"id": "m4", "titulo": "Estruturas de Dados"},
    ]
    ss = SemanticSearch(documentos)
    for r in ss.search("prova funções em C", top_k=4):
        print(r)
