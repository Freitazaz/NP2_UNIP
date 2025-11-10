import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify
from ai.search import SemanticSearch
from ai.similarity import top_similar_submissions
from ai.risk import RiskModel
from ai.recommend import ContentRecommender
import json

# Caminho base (pasta principal do projeto)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")

def load_json(file_name):
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "r", encoding="utf-8-sig") as f:  # <-- o segredo está aqui
        return json.load(f)

materiais = load_json("materiais.json")
print(materiais)

entregas = load_json("entregas.json")
print(entregas)

historico = load_json("historico_notas.json")
print(historico)


app = Flask(__name__)

# Carregar dados iniciais
with open("data/materiais.json", encoding="utf-8") as f:
    materiais = json.load(f)
with open("data/historico_notas.json", encoding="utf-8") as f:
    historico = json.load(f)
with open("data/entregas.json", encoding="utf-8") as f:
    entregas = json.load(f)

# Instanciar modelos
search_engine = SemanticSearch(materiais)
recommender = ContentRecommender(materiais)
risk_model = RiskModel()
risk_model.fit(historico)  # Treina com histórico

# Rotas
@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 5))
    results = search_engine.search(query, top_k=top_k)
    return jsonify(results)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    viewed_ids = data.get("viewed_ids", [])
    top_k = int(data.get("top_k", 5))
    results = recommender.recommend(viewed_ids, top_k=top_k)
    return jsonify(results)

@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.json
    submissions = data.get("submissions", [])
    target_id = data.get("target_id", "")
    method = data.get("method", "tfidf")
    results = top_similar_submissions(submissions, target_id, method=method)
    return jsonify(results)

@app.route("/risk", methods=["GET"])
def risk():
    faltas = int(request.args.get("faltas", 0))
    media = float(request.args.get("media", 0.0))
    entregas_atraso = int(request.args.get("entregas_atraso", 0))
    prob = risk_model.predict_proba(faltas, media, entregas_atraso)
    label = risk_model.predict_label(faltas, media, entregas_atraso)
    return jsonify({"probabilidade": prob, "risco": label})

@app.route("/")
def home():
    return """
    <h2>🚀 API Flask do Projeto PIM2</h2>
    <p>Rotas disponíveis:</p>
    <ul>
        <li><b>GET</b> /search?q=palavra&top_k=5</li>
        <li><b>POST</b> /recommend</li>
        <li><b>POST</b> /similarity</li>
        <li><b>GET</b> /risk?faltas=10&media=6.5&entregas_atraso=2</li>
    </ul>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)