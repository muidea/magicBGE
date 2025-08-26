# embed_server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 本地模型

@app.route("/bge/embed", methods=["POST"])
def embed():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400

    vector = model.encode([text])[0].tolist()
    return jsonify({"vector": vector})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
