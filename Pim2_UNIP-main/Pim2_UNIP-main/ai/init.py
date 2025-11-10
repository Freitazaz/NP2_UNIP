# ai/__init__.py
"""
Pacote de Inteligência Artificial do Sistema Acadêmico (PIM II).
Módulos:
 - search: busca semântica (TF-IDF + similaridade do cosseno)
 - recommend: recomendação conteúdo-baseado
 - similarity: similaridade entre entregas (apoio anti-plágio)
 - risk: modelo de risco acadêmico (Regressão Logística)
"""
__all__ = ["search", "recommend", "similarity", "risk"]