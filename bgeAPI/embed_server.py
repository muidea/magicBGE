# merged_server_multilang.py
from flask import Flask, request, jsonify
import jieba
from sentence_transformers import SentenceTransformer, util
import re

app = Flask(__name__)

# 加载本地模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

# 加载停用词
def load_stopwords(filepath):
    stopwords = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word.lower())
    except FileNotFoundError:
        print(f"{filepath} 文件未找到")
    return stopwords

stopwords_zh = load_stopwords("stopwords_zh.txt")
stopwords_en = load_stopwords("stopwords_en.txt")
stopwords = stopwords_zh.union(stopwords_en)

def extract_english_words(text):
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

def extract_keywords(text, top_k=10):
    words_zh = [w for w in jieba.cut(text) if w not in stopwords_zh and len(w) > 1]
    words_en = [w for w in extract_english_words(text) if w not in stopwords_en]
    words = list(set(words_zh + words_en))
    if not words:
        return []

    text_emb = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    word_embs = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(text_emb, word_embs)[0].cpu().numpy()
    word_score_pairs = sorted(zip(words, cos_scores), key=lambda x: x[1], reverse=True)
    keywords = [w for w, _ in word_score_pairs[:top_k]]
    return keywords

# ===========================
# 单条文本 embedding 接口
# ===========================
@app.route("/bge/embed", methods=["POST"])
def embed():
    """
    单条文本向量生成接口

    请求格式:
    {
        "text": "字符串文本"
    }

    响应格式:
    {
        "vector": [浮点数向量列表]
    }
    """
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400

    vector = model.encode([text])[0].tolist()
    return jsonify({"vector": vector})

# ===========================
# 批量文本 embedding 接口
# ===========================
@app.route("/bge/embed_bulk", methods=["POST"])
def embed_bulk():
    """
    批量文本向量生成接口

    请求格式:
    {
        "texts": ["文本1", "文本2", ...]
    }

    响应格式:
    {
        "vectors": [
            [向量1], [向量2], ...
        ]
    }
    """
    data = request.json
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Invalid texts"}), 400

    vectors = model.encode(texts).tolist()
    return jsonify({"vectors": vectors})

# ===========================
# 中文+英文关键词提取接口
# ===========================
@app.route("/bge/keywords", methods=["POST"])
def keywords_api():
    """
    中文+英文关键词提取接口

    请求格式:
    {
        "text": "字符串文本",
        "top_k": 关键词数量（整数，可选，默认10）
    }

    响应格式:
    {
        "keywords": ["关键词1", "关键词2", ...]
    }
    """
    try:
        data = request.json
        text = data.get("text", "").strip()
        top_k = int(data.get("top_k", 10))

        if not text:
            return jsonify({"error": "text 参数不能为空"}), 400

        keywords = extract_keywords(text, top_k)
        return jsonify({"keywords": keywords})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# 启动服务
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
