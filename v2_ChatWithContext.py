import os
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# App configuration.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
SYSTEM_PROMPT = "You are Jarvis, a concise and helpful AI assistant."
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "12"))

# Simple in-memory history store keyed by session ID.
CHAT_HISTORY = {}


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
        contents.append(
            types.Content(role=role, parts=[types.Part(text=text)]),
        )
    return contents


@app.route("/")
def index():
    return render_template("index.html")


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

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_contents(history),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        reply = (response.text or "").strip() or "No reply yet."
        append_history(history, "assistant", reply)
    except Exception as error:
        if history and history[-1].get("role") == "user":
            history.pop()
        return jsonify({"reply": f"Error contacting model: {error}"}), 500

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
