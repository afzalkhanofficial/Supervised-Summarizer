# app.py
import io
import os
import re
import uuid
import pickle
from typing import List, Dict, Any

import numpy as np
import nltk
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    send_from_directory,
    url_for,
)
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# ---------------------- CONFIG & MODEL LOAD ---------------------- #

# Ensure punkt is available (Render-friendly)
nltk.download('punkt', quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")
MODEL_DIR = BASE_DIR

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Supervised model files (expecting a sklearn classifier with predict_proba)
model_path = os.path.join(MODEL_DIR, "model.pkl")
tfidf_path = os.path.join(MODEL_DIR, "tfidf.pkl")

model = None
vectorizer = None

if os.path.exists(model_path) and os.path.exists(tfidf_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            vectorizer = pickle.load(f)
        print("Loaded supervised model and vectorizer.")
    except Exception as e:
        print(f"Failed to load model/vectorizer: {e}")
else:
    print("Model or vectorizer not found. Summaries that use the supervised model will return an error message.")

# ---------------------- HTML TEMPLATES (Yellow Theme, simplified) ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>PolicyBrief AI (Supervised)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Inter','sans-serif'], mono: ['JetBrains Mono','monospace'] },
          colors: {
            amber: {
              50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d',
              400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', 700: '#b45309'
            }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f8fafc; }
    .glass-panel { background: rgba(255,255,255,0.7); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.5); }
  </style>
</head>
<body class="text-slate-800 min-h-screen flex flex-col">

  <nav class="w-full glass-panel border-b border-slate-200/50">
    <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 bg-gradient-to-tr from-amber-500 to-amber-400 rounded-xl flex items-center justify-center text-white">
          <i class="fa-solid fa-file-lines"></i>
        </div>
        <div class="font-extrabold text-xl">Policy<span class="text-amber-600">Brief</span></div>
      </div>
      <div class="text-sm text-slate-600">Supervised Extractive Summarizer</div>
    </div>
  </nav>

  <main class="flex-grow pt-12 pb-20 px-4">
    <div class="max-w-3xl mx-auto">

      <div class="text-center space-y-4 mb-8">
        <h1 class="text-4xl font-extrabold text-slate-900">PolicyBrief AI</h1>
        <p class="text-slate-600">Upload a PDF and our supervised model will extract the most important sentences.</p>
      </div>

      <div class="glass-panel rounded-2xl p-6 bg-white/60 shadow-md">
        <form id="uploadForm" action="/summarize" method="post" enctype="multipart/form-data" class="space-y-4">
          <div class="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer hover:border-amber-400" onclick="document.getElementById('file').click()">
            <input id="file" type="file" name="file" accept=".pdf" style="display:none" />
            <div class="text-amber-500 text-3xl mb-2"><i class="fa-solid fa-cloud-arrow-up"></i></div>
            <div class="font-bold text-slate-700">Click to upload PDF</div>
            <div class="text-sm text-slate-500 mt-2">(PDF only)</div>
          </div>

          <div class="flex gap-3">
            <button type="submit" class="flex-1 py-3 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold">Generate Summary</button>
            <a href="/" class="px-4 py-3 rounded-xl border border-slate-200 text-slate-700">Reset</a>
          </div>
        </form>
      </div>

      <p class="text-xs text-slate-400 mt-4">Note: If the supervised model (model.pkl & tfidf.pkl) isn't present on the server, the app will show a friendly message instead of crashing.</p>
    </div>
  </main>

</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-50 text-slate-800">
  <nav class="w-full bg-white/80 border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-gradient-to-tr from-amber-500 to-amber-400 rounded-lg flex items-center justify-center text-white"><i class="fa-solid fa-file-lines"></i></div>
        <div class="font-extrabold text-lg">Policy<span class="text-amber-600">Brief</span></div>
      </div>
      <a href="/" class="text-xs px-3 py-2 border rounded-full">New</a>
    </div>
  </nav>

  <main class="pt-8 pb-12 px-4">
    <div class="max-w-6xl mx-auto grid lg:grid-cols-12 gap-8">
      <section class="lg:col-span-7 bg-white rounded-2xl p-6 shadow">
        <div class="flex items-start justify-between mb-4">
          <div>
            <div class="text-xs text-amber-700 font-bold uppercase">Supervised (Model)</div>
            <h1 class="text-2xl font-extrabold">Policy Summary (Extractive)</h1>
          </div>
          {% if summary_pdf_url %}
          <a href="{{ summary_pdf_url }}" class="px-3 py-2 bg-slate-900 text-white rounded-md text-xs">Download PDF</a>
          {% endif %}
        </div>

        <div class="mt-4">
          <h2 class="text-sm text-slate-500 uppercase mb-2">Extracted Sentences</h2>
          <div class="space-y-3 text-slate-700">
            {% if summary %}
              {% for s in summary %}
                <p>• {{ s }}</p>
              {% endfor %}
            {% else %}
              <p class="text-slate-500">No summary available.</p>
            {% endif %}
          </div>
        </div>
      </section>

      <aside class="lg:col-span-5 space-y-6">
        <div class="bg-white rounded-2xl p-6 shadow">
          <h3 class="text-sm font-bold text-slate-700 mb-2">Original Document</h3>
          <div class="h-64 border rounded overflow-hidden bg-slate-100">
            {% if orig_type == 'pdf' %}
              <iframe src="{{ orig_url }}" class="w-full h-full" title="Original PDF"></iframe>
            {% elif orig_type == 'text' %}
              <div class="p-3 text-xs font-mono overflow-auto h-full">{{ orig_text }}</div>
            {% else %}
              <div class="p-3 text-xs text-slate-500">Original file type: {{ orig_type }}</div>
            {% endif %}
          </div>
        </div>

        <div class="bg-white rounded-2xl p-6 shadow">
          <h3 class="text-sm font-bold text-slate-700 mb-2">Debug / Info</h3>
          <pre class="text-xs text-slate-600 whitespace-pre-wrap">{{ debug_info }}</pre>
        </div>
      </aside>
    </div>
  </main>
</body>
</html>
"""

# ---------------------- TEXT / PDF UTILITIES ---------------------- #


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', " ", text)
    return text.strip()


def clean_text_line(text: str) -> str:
    """Basic cleaning used by supervised pipeline (remove page headers/footers/numbering)."""
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+(\.\d+)*\s*', '', text)
    return text.strip()


def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception:
        return ""
    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""
        if txt:
            pages.append(txt)
    return "\n".join(pages)


# ---------------------- SUPERVISED EXTRACTIVE SUMMARY CORE ---------------------- #


def generate_extractive_summary(text: str, num_sentences: int = 7) -> List[str]:
    """
    Uses the pre-loaded supervised model + vectorizer to score sentences and pick
    a small set with redundancy removal.
    Returns a list of selected sentences in original order.
    """
    if not model or not vectorizer:
        return ["Model Error: supervised model or vectorizer not loaded on server."]

    # 1. Sentence split using nltk
    try:
        raw_sentences = nltk.sent_tokenize(text)
    except Exception:
        return ["Error: could not split text into sentences."]

    cleaned_sentences = [clean_text_line(s) for s in raw_sentences]

    # 2. Filter out very short / junk
    valid_sentences = []
    for i, s in enumerate(cleaned_sentences):
        s_stripped = s.strip()
        if len(s_stripped) > 40 and re.search('[a-zA-Z]', s_stripped):
            valid_sentences.append(s_stripped)

    if not valid_sentences:
        return ["No valid text found in the document."]

    # 3. Vectorize & predict probabilities
    try:
        features = vectorizer.transform(valid_sentences)  # sparse matrix
    except Exception as e:
        return [f"Vectorizer Error: {str(e)}"]

    try:
        scores = model.predict_proba(features)[:, 1]
    except Exception:
        try:
            scores = model.decision_function(features)
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-12)
        except Exception as e:
            return [f"Model Error: cannot score sentences ({e})"]

    # 4. Rank & pick with redundancy check
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = []
    selected_vectors = []

    for idx in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break

        current_vec = features[idx]
        is_redundant = False
        if selected_vectors:
            try:
                sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
                if np.max(sims) > 0.65:
                    is_redundant = True
            except Exception:
                is_redundant = False

        if not is_redundant:
            selected_indices.append(idx)
            try:
                selected_vectors.append(current_vec.toarray()[0])
            except Exception:
                selected_vectors.append(np.array(current_vec.todense()).ravel())

    final_order = sorted(selected_indices)
    summary_sentences = [valid_sentences[i] for i in final_order]
    return summary_sentences


# ---------------------- PDF OUTPUT ---------------------- #
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit


def save_summary_pdf(title: str, bullets: List[str], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 28

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Executive Extract")
    y -= 18

    c.setFont("Helvetica", 10)
    for b in bullets:
        if y < 80:
            c.showPage()
            y = height - margin
        lines = simpleSplit("• " + b, "Helvetica", 10, width - 2 * margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
        y -= 6

    c.save()


# ---------------------- ROUTES ---------------------- #


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/summaries/<path:filename>")
def summary_file(filename):
    return send_from_directory(app.config["SUMMARY_FOLDER"], filename, as_attachment=True)


@app.route("/summarize", methods=["POST"])
def summarize():
    # Wrap the entire endpoint to catch unexpected errors and return a friendly page
    try:
        f = request.files.get("file")
        if not f or f.filename == "":
            abort(400, "No file uploaded")

        filename = secure_filename(f.filename)
        uid = uuid.uuid4().hex
        stored_name = f"{uid}_{filename}"
        stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        f.save(stored_path)

        lower_name = filename.lower()
        orig_text = ""
        orig_type = "unknown"

        # Only PDF accepted from the UI; keep a safe fallback
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            with open(stored_path, "rb") as fin:
                raw = fin.read()
            orig_text = extract_text_from_pdf_bytes(raw)
        else:
            # attempt to decode text files
            orig_type = "text"
            try:
                with open(stored_path, "rb") as fin:
                    orig_text = fin.read().decode("utf-8", errors="ignore")
            except Exception:
                orig_text = ""

        if len(orig_text.strip()) < 40:
            summary_sentences = ["No or insufficient text extracted from the document to summarize."]
            debug_info = "Extracted text length under threshold."
        else:
            # supervised pipeline: default number of sentences
            num_sentences = 7
            summary_sentences = generate_extractive_summary(orig_text, num_sentences=num_sentences)
            debug_info = (
                f"Original sentences (approx): {len(nltk.sent_tokenize(orig_text))}\n"
                f"Selected (requested): {num_sentences}\n"
                f"Model loaded: {'yes' if model and vectorizer else 'no'}"
            )

        # Save PDF summary (best-effort)
        summary_filename = f"{uid}_summary.pdf"
        summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
        try:
            save_summary_pdf("Policy Brief - Executive Extract", summary_sentences, summary_path)
            summary_pdf_url = url_for("summary_file", filename=summary_filename)
        except Exception as e:
            summary_pdf_url = None
            debug_info += f"\nPDF generation failed: {e}"

        display_orig_text = orig_text[:20000]

        return render_template_string(
            RESULT_HTML,
            title="PolicyBrief Summary",
            summary=summary_sentences,
            orig_type=orig_type,
            orig_url=url_for("uploaded_file", filename=stored_name),
            orig_text=display_orig_text,
            debug_info=debug_info,
            summary_pdf_url=summary_pdf_url,
        )

    except Exception as e:
        # Log the error server-side and show a friendly error to the user
        import traceback
        tb = traceback.format_exc()
        print("Internal error in /summarize:", tb)

        # Minimal error page (no stacktrace shown to user)
        friendly = f"""
        <!doctype html><html><head><meta charset="utf-8"><title>Error</title></head>
        <body style="font-family:Inter, sans-serif; padding:30px;">
        <h2 style="color:#b45309">Internal Server Error</h2>
        <p>Sorry — something went wrong while processing your file.</p>
        <p style="color:#64748b; font-size:0.9rem;">If the issue persists, check server logs for details.</p>
        <a href="/" style="display:inline-block; margin-top:16px; padding:8px 12px; background:#f59e0b; color:white; border-radius:6px; text-decoration:none;">Back</a>
        </body></html>
        """
        return friendly, 500


if __name__ == "__main__":
    # debug=True only for local testing; in production use a WSGI server.
    app.run(host="0.0.0.0", port=5000, debug=True)
