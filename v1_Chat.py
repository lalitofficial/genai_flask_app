import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)

# App configuration.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
SYSTEM_PROMPT = "You are Jarvis, a concise and helpful AI assistant."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_text = (payload.get("message") or "").strip()
    
    if not GEMINI_API_KEY:
        return (
            jsonify({"reply": "Missing GEMINI_API_KEY in your environment."}),
            500,
        )

    try:

        
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        reply = (response.text or "").strip()
        if not reply:
            reply = "No reply yet."



    except Exception as error:
        return jsonify({"reply": f"Error contacting model: {error}"}), 500

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
