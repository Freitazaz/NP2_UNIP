# ai/risk.py
from typing import List, Dict, Optional
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


class RiskModel:
    """
    Classificador de risco acadêmico.
    Features sugeridas: faltas (int), media (float), entregas_atraso (int).
    Label: reprovou (0/1).
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0, class_weight: Optional[dict] = None) -> None:
        self.model = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight)

    # ---------- treino ----------
    def fit(self, historico: List[Dict]) -> None:
        X = np.array(
            [[h.get("faltas", 0), float(h.get("media", 0.0)), h.get("entregas_atraso", 0)] for h in historico],
            dtype=float,
        )
        y = np.array([int(h.get("reprovou", 0)) for h in historico], dtype=int)
        self.model.fit(X, y)

    # ---------- inferência ----------
    def predict_proba(self, faltas: int, media: float, entregas_atraso: int = 0) -> float:
        X = np.array([[faltas, media, entregas_atraso]], dtype=float)
        return float(self.model.predict_proba(X)[0][1])

    def predict_label(self, faltas: int, media: float, entregas_atraso: int = 0, threshold: float = 0.5) -> int:
        return int(self.predict_proba(faltas, media, entregas_atraso) >= threshold)

    # ---------- persistência ----------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "RiskModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        obj = cls()
        obj.model = model
        return obj


# Exemplo: python -m ai.risk
if __name__ == "__main__":
    dados = [
        {"faltas": 2, "media": 8.5, "entregas_atraso": 0, "reprovou": 0},
        {"faltas": 15, "media": 5.0, "entregas_atraso": 3, "reprovou": 1},
        {"faltas": 10, "media": 6.0, "entregas_atraso": 1, "reprovou": 0},
        {"faltas": 18, "media": 4.8, "entregas_atraso": 4, "reprovou": 1},
    ]
    rm = RiskModel()
    rm.fit(dados)
    p = rm.predict_proba(faltas=12, media=5.8, entregas_atraso=3)
    print(f"Probabilidade de reprovação: {p:.2f}")