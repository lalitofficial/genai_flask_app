import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from google import genai
from google.genai import types
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from werkzeug.utils import secure_filename

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

# App configuration.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
SYSTEM_PROMPT = "You are Jarvis, a concise and helpful AI assistant."
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "12"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Simple in-memory stores.
CHAT_HISTORY = {}
VECTOR_INDEX = None


def ensure_upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_session_id():
    session_id = session.get("session_id")
    if not session_id:
        session_id = uuid4().hex
        session["session_id"] = session_id
    return session_id


def append_history(history, role, text):
    history.append({"role": role, "text": text})
    if MAX_HISTORY > 0 and len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]


def build_contents(history):
    contents = []
    for entry in history:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        role = "user" if entry.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


def configure_embeddings():
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBED_MODEL_NAME,
        api_key=GEMINI_API_KEY,
    )


def build_index():
    if not GEMINI_API_KEY:
        return None

    ensure_upload_dir()
    documents = SimpleDirectoryReader(
        str(UPLOAD_DIR),
        recursive=True,
        required_exts=sorted(ALLOWED_EXTENSIONS),
    ).load_data()
    if not documents:
        return None

    configure_embeddings()
    return VectorStoreIndex.from_documents(documents)


def get_rag_context(query_text, top_k=4):
    if VECTOR_INDEX is None:
        return "", []

    retriever = VECTOR_INDEX.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query_text)
    context_chunks = []
    sources = []

    for item in nodes:
        node = item.node
        context_chunks.append(node.get_content())
        metadata = node.metadata or {}
        source = metadata.get("file_name") or metadata.get("file_path")
        if source:
            sources.append(Path(source).name)

    return "\n\n".join(context_chunks), sorted(set(sources))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if not GEMINI_API_KEY:
        return jsonify({"message": "Missing GEMINI_API_KEY in your environment."}), 500

    if "file" not in request.files:
        return jsonify({"message": "Missing file in request."}), 400

    upload_file = request.files["file"]
    if not upload_file or upload_file.filename == "":
        return jsonify({"message": "No file selected."}), 400

    if not allowed_file(upload_file.filename):
        return jsonify(
            {
                "message": "Unsupported file type.",
                "allowed": sorted(ALLOWED_EXTENSIONS),
            }
        ), 400

    ensure_upload_dir()
    filename = secure_filename(upload_file.filename)
    file_path = UPLOAD_DIR / filename
    upload_file.save(file_path)

    global VECTOR_INDEX
    VECTOR_INDEX = build_index()

    return jsonify({"message": "File uploaded and indexed.", "file": filename})


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_text = (payload.get("message") or "").strip()
    if not user_text:
        return jsonify({"reply": "Please send a message to continue."}), 400

    if not GEMINI_API_KEY:
        return (
            jsonify({"reply": "Missing GEMINI_API_KEY in your environment."}),
            500,
        )

    session_id = get_session_id()
    history = CHAT_HISTORY.setdefault(session_id, [])
    append_history(history, "user", user_text)

    context_text, sources = get_rag_context(user_text)
    if context_text:
        system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Use the following context to answer. If the answer is not in the "
            "context, say you do not know.\n\n"
            f"Context:\n{context_text}"
        )
    else:
        system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "No documents are indexed yet. Answer generally."
        )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_contents(history),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        reply = (response.text or "").strip() or "No reply yet."
        append_history(history, "assistant", reply)
    except Exception as error:
        if history and history[-1].get("role") == "user":
            history.pop()
        return jsonify({"reply": f"Error contacting model: {error}"}), 500

    return jsonify({"reply": reply, "sources": sources})


if __name__ == "__main__":
    ensure_upload_dir()
    VECTOR_INDEX = build_index()
    app.run(debug=True)
