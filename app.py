from datetime import datetime, timezone
from functools import wraps
import os

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from core.config import get_settings
from rag_engine import assistant
from services.auth_service import AuthService


app = Flask(__name__)
settings = get_settings()
app.secret_key = settings.session_secret
auth_service = AuthService()


def current_user():
    return auth_service.get_user(session.get("user_id"))


def login_required(route):
    @wraps(route)
    def wrapper(*args, **kwargs):
        if current_user():
            return route(*args, **kwargs)
        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication is required. Please log in first."}), 401
        return redirect(url_for("login", next=request.path))

    return wrapper


@app.get("/")
@login_required
def home():
    return render_template("index.html", user=current_user())


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        next_url = request.args.get("next") or url_for("home")
        try:
            user = auth_service.login(email, password)
            session.clear()
            session["user_id"] = user.user_id
            return redirect(next_url)
        except ValueError as error:
            return render_template("auth.html", mode="login", error=str(error), form=request.form), 400

    return render_template("auth.html", mode="login", error=None, form={})


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        try:
            user = auth_service.signup(name, email, password)
            session.clear()
            session["user_id"] = user.user_id
            return redirect(url_for("home"))
        except ValueError as error:
            return render_template("auth.html", mode="signup", error=str(error), form=request.form), 400

    return render_template("auth.html", mode="signup", error=None, form={})


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "hybrid-ai-faq-assistant",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "groq_configured": assistant.groq.enabled,
            "groq_model": assistant.groq.model if assistant.groq.enabled else None,
        }
    )


@app.post("/api/ask")
@login_required
def ask():
    payload = request.get_json(silent=True) or request.form
    question = (payload.get("question") or "").strip()
    history = payload.get("history") if hasattr(payload, "get") else None
    mode = (payload.get("mode") or "hybrid").strip() if hasattr(payload, "get") else "hybrid"

    if not question:
        return (
            jsonify(
                {
                    "error": "Please enter a question before sending your request.",
                }
            ),
            400,
        )

    response = assistant.answer_question(question, history=history, mode=mode)
    return jsonify(response)


@app.get("/api/knowledge")
@login_required
def knowledge_stats():
    return jsonify(assistant.knowledge_base.get_stats())


@app.post("/api/knowledge")
@login_required
def ingest_knowledge():
    manual_text = (request.form.get("manual_text") or "").strip()
    manual_title = (request.form.get("manual_title") or "Manual Knowledge Note").strip()
    source_url = (request.form.get("source_url") or "").strip()
    uploaded_file = request.files.get("knowledge_file")

    try:
        if uploaded_file and uploaded_file.filename:
            result = assistant.ingest_file(uploaded_file.filename, uploaded_file.read())
            return jsonify({"message": "Knowledge file added successfully.", "result": result})

        if manual_text:
            result = assistant.ingest_manual_text(manual_title, manual_text)
            return jsonify({"message": "Manual knowledge added successfully.", "result": result})

        if source_url:
            result = assistant.ingest_url(source_url)
            return jsonify({"message": "Website knowledge added successfully.", "result": result})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({"error": "Submit a file, manual text, or website URL."}), 400


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "The requested resource was not found."}), 404


@app.errorhandler(500)
def internal_error(_error):
    return (
        jsonify(
            {
                "error": "Something went wrong while processing the request.",
            }
        ),
        500,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
