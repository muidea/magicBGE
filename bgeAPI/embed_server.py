# embed_server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)  # 本地模型

@app.route("/bge/embed", methods=["POST"])
def embed():
    """
    单条文本向量生成接口
    请求: {"text": "你好世界"}
    响应: {"vector": [0.1, 0.2, ...]}
    """
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400

    vector = model.encode([text])[0].tolist()
    return jsonify({"vector": vector})


@app.route("/bge/embed_bulk", methods=["POST"])
def embed_bulk():
    """
    批量文本向量生成接口
    请求:
    {
        "texts": ["你好世界", "人工智能", "嵌入向量"]
    }
    响应:
    {
        "vectors": [[...], [...], [...]]
    }
    """
    data = request.json
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Invalid texts"}), 400

    vectors = model.encode(texts).tolist()
    return jsonify({"vectors": vectors})


if __name__ == "__main__":
    # threaded=True 允许并发请求，适合批量计算场景
    app.run(host="0.0.0.0", port=8080, threaded=True)
