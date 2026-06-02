import os
import re
import json
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Load the Sentiment Agent ───────────────────────────────────────────────────
import agent_backend as agent

def _startup():
    agent.initialize()

thread = threading.Thread(target=_startup, daemon=True)
thread.start()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/agent/status")
def agent_status():
    status = agent.get_status()
    # Let the UI know if a key is pre-configured in .env
    status["has_env_key"] = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    return jsonify(status)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data        = request.get_json()
    text        = (data.get("text") or "").strip()
    raw_key     = data.get("api_key") or ""
    # '__env__' sentinel means use the .env key
    api_key     = (os.getenv("ANTHROPIC_API_KEY", "") if raw_key == "__env__" else raw_key).strip()
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    temperature = float(data.get("temperature", 0.0))

    if not text:
        return jsonify({"error": "Please enter regulation text."}), 400

    status = agent.get_status()
    if not status["ready"]:
        if status["loading"]:
            return jsonify({"error": "Agent is still loading models. Please wait a moment."}), 503
        if status["error"]:
            return jsonify({"error": f"Agent failed to load: {status['error']}"}), 500
        return jsonify({"error": "Agent not ready."}), 503

    if not api_key:
        return jsonify({"error": "Claude API key required. Enter it in Settings."}), 400

    try:
        result = agent.analyze_regulation(text, api_key=api_key, temperature=temperature)

        # Check for Claude error sentinels embedded in explanation
        expl = result.get("explanation", "")
        if expl == "__INVALID_KEY__":
            return jsonify({"error": "Invalid Claude API key. Please check your key in Settings."}), 401
        if expl == "__RATE_LIMIT__":
            return jsonify({"error": "Claude rate limit reached. Wait a moment and try again."}), 429
        if expl == "__NO_KEY__":
            return jsonify({"error": "Claude API key required. Enter it in Settings."}), 400
        if expl.startswith("__ERROR__:"):
            return jsonify({"error": expl.replace("__ERROR__:", "Claude error:").strip()}), 500

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    session_id = (data.get("session_id") or "").strip()
    message    = (data.get("message") or "").strip()
    context    = data.get("context") or {}
    raw_key    = data.get("api_key") or ""
    api_key    = (os.getenv("ANTHROPIC_API_KEY", "") if raw_key == "__env__" else raw_key).strip()
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    temperature = float(data.get("temperature", 0.0))

    if not session_id or not message:
        return jsonify({"error": "session_id and message are required."}), 400
    if not api_key:
        return jsonify({"error": "Claude API key required. Enter it in Settings."}), 400

    reply = agent.chat_about_regulation(
        session_id, message, context, api_key=api_key, temperature=temperature
    )

    if reply == "__INVALID_KEY__":
        return jsonify({"error": "Invalid Claude API key. Please check Settings."}), 401
    if reply == "__RATE_LIMIT__":
        return jsonify({"error": "Claude rate limit reached. Wait a moment and try again."}), 429
    if reply == "__NO_KEY__":
        return jsonify({"error": "Claude API key required. Enter it in Settings."}), 400
    if reply.startswith("__ERROR__:"):
        return jsonify({"error": reply.replace("__ERROR__:", "Claude error:").strip()}), 500

    return jsonify({"reply": reply})


@app.route("/api/chat/context", methods=["POST"])
def chat_context():
    data = request.get_json() or {}
    session_id = (data.get("session_id") or "").strip()
    context = data.get("context") or {}

    if not session_id:
        return jsonify({"error": "session_id is required."}), 400

    agent.set_session_context(session_id, context)
    return jsonify({"ok": True})


@app.route("/api/chat/clear", methods=["POST"])
def chat_clear():
    data = request.get_json() or {}
    session_id = (data.get("session_id") or "").strip()
    if session_id:
        agent.clear_session(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Starting Federal Regulation Sentiment Analysis Platform...")
    app.run(debug=False, port=5050, use_reloader=False)
