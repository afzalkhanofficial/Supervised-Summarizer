import io
import os
import re
import uuid
import json
import time
import pickle
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import networkx as nx
import nltk
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    send_from_directory,
    jsonify,
    url_for,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from PIL import Image

# PDF Generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

# Gemini AI (Optional)
import google.generativeai as genai

# Download NLTK data for production
nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------------- CONFIG ---------------------- #

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")
MODEL_DIR = BASE_DIR  # Where model.pkl and tfidf.pkl are located

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Configure Gemini (Get API Key from Environment Variable)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_API_KEY = None

# ---------------------- LOAD TRAINED MODEL ---------------------- #

model = None
vectorizer = None

def load_trained_model():
    """Attempts to load the supervised Logistic Regression model."""
    global model, vectorizer
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    tfidf_path = os.path.join(MODEL_DIR, 'tfidf.pkl')
    
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(tfidf_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print("✅ Supervised Model Loaded Successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("⚠️ Model files not found. Using Fallback TextRank Mode.")
        return False

# Load model on startup
has_trained_model = load_trained_model()

# ---------------------- HTML TEMPLATES ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>PolicyBrief AI | Healthcare Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Inter', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] },
          colors: { teal: { 50: '#f0fdfa', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e' } },
          animation: { 'float': 'float 6s ease-in-out infinite', 'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite' },
          keyframes: { float: { '0%, 100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-10px)' } } }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f8fafc; }
    .glass-panel { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.5); }
    .gradient-text { background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    @keyframes progress-stripes { from { background-position: 1rem 0; } to { background-position: 0 0; } }
    .animate-stripes { background-image: linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent); background-size: 1rem 1rem; animation: progress-stripes 1s linear infinite; }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden min-h-screen flex flex-col">

  <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-teal-200/30 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>
  <div class="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-200/30 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>

  <nav class="fixed w-full z-40 glass-panel border-b border-slate-200/50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg text-white">
            <i class="fa-solid fa-file-medical-alt text-xl"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">Policy<span class="text-teal-600">Brief.AI</span></span>
        </div>
      </div>
    </div>
  </nav>

  <main class="flex-grow pt-32 pb-20 px-4">
    <div class="max-w-5xl mx-auto">
      <div class="text-center space-y-6 mb-16">
        <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-teal-50 border border-teal-100 text-teal-700 text-xs font-bold uppercase tracking-wide animate-float">
          <span class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span> Supervised Learning Model v2.0
        </div>
        <h1 class="text-5xl md:text-6xl font-extrabold text-slate-900 leading-tight">
          Summarize Healthcare <br><span class="gradient-text">Policy Documents</span>
        </h1>
        <p class="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Upload PDF, Text, or Images. We use a <strong>Supervised Logistic Regression Model</strong> trained on policy briefs to extract high-value insights automatically.
        </p>
      </div>

      <div class="glass-panel rounded-3xl p-1 shadow-2xl shadow-slate-200/50 max-w-3xl mx-auto">
        <div class="bg-white/50 rounded-[1.3rem] p-6 md:p-10 border border-white/50">
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
            
            <div class="group relative w-full h-64 border-3 border-dashed border-slate-300 rounded-2xl bg-slate-50/50 hover:bg-teal-50/30 hover:border-teal-400 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" onclick="document.getElementById('file-input').click()">
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              <div id="upload-prompt" class="text-center space-y-4">
                <div class="w-16 h-16 bg-white rounded-full shadow-md flex items-center justify-center mx-auto text-teal-500 text-2xl"><i class="fa-solid fa-cloud-arrow-up"></i></div>
                <p class="text-lg font-bold text-slate-700">Click to upload or Drag & Drop</p>
                <p id="filename-display" class="text-sm text-slate-500 font-mono"></p>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-gradient-to-r from-teal-600 to-cyan-700 text-white font-bold text-lg shadow-lg hover:scale-[1.02] transition-all flex items-center justify-center gap-2">
              <i class="fa-solid fa-wand-magic-sparkles"></i> Generate Summary
            </button>

          </form>
        </div>
      </div>
    </div>
  </main>
  
  <div id="progress-overlay" class="fixed inset-0 bg-white/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
     <div class="w-20 h-20 border-4 border-slate-200 border-t-teal-500 rounded-full animate-spin mb-4"></div>
     <h2 class="text-xl font-bold text-slate-800">Analyzing Document...</h2>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const filenameDisplay = document.getElementById('filename-display');
    const form = document.getElementById('uploadForm');
    const overlay = document.getElementById('progress-overlay');

    fileInput.addEventListener('change', function() {
        if(this.files[0]) filenameDisplay.textContent = "Selected: " + this.files[0].name;
    });

    form.addEventListener('submit', function() {
        if(fileInput.files.length > 0) overlay.classList.remove('hidden');
        else { alert("Please select a file."); event.preventDefault(); }
    });
  </script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Summary Result</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
  <style> body { font-family: 'Inter', sans-serif; background: #f8fafc; } </style>
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <span class="font-extrabold text-xl tracking-tight text-slate-900">Policy<span class="text-teal-600">Brief.AI</span></span>
        <a href="{{ url_for('index') }}" class="px-4 py-2 text-xs font-bold rounded-full border border-slate-200 hover:text-teal-600 bg-white shadow-sm">New Summary</a>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      
      <section class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-3xl shadow-xl border border-slate-100 p-8">
          
          <div class="flex items-center justify-between mb-6 border-b border-slate-100 pb-6">
             <div>
                <span class="px-2 py-1 rounded-md bg-teal-50 text-teal-700 text-[0.65rem] font-bold uppercase tracking-wide border border-teal-100">
                  {{ model_used }}
                </span>
                <h1 class="text-2xl font-extrabold text-slate-900 mt-2">Executive Summary</h1>
             </div>
             {% if summary_pdf_url %}
             <a href="{{ summary_pdf_url }}" class="px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-teal-600 transition shadow-lg">
               <i class="fa-solid fa-download mr-2"></i> PDF
             </a>
             {% endif %}
          </div>

          <div class="mb-8 p-5 rounded-2xl bg-slate-50 border border-slate-100 text-sm leading-relaxed text-slate-700 italic">
             <strong>Abstract:</strong> {{ abstract }}
          </div>

          {% for sec in sections %}
          <div class="mb-6">
             <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
               <span class="w-1.5 h-6 rounded-full bg-teal-500 block"></span> {{ sec.title }}
             </h3>
             <ul class="space-y-2">
               {% for bullet in sec.bullets %}
               <li class="flex items-start gap-3 text-sm text-slate-600">
                 <i class="fa-solid fa-check mt-1 text-teal-500 text-xs"></i> <span>{{ bullet }}</span>
               </li>
               {% endfor %}
             </ul>
          </div>
          {% endfor %}

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        <div class="bg-white rounded-3xl shadow-xl border border-slate-100 p-6 flex flex-col h-[500px]">
          <div class="mb-4">
            <h2 class="text-sm font-bold text-slate-800 flex items-center gap-2"><i class="fa-solid fa-robot text-teal-600"></i> Ask Gemini</h2>
          </div>
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-3 mb-4 pr-2">
             <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center text-teal-600 text-xs shrink-0"><i class="fa-solid fa-robot"></i></div>
                <div class="bg-slate-100 rounded-2xl rounded-tl-none p-3 text-xs text-slate-700">Ask me anything about the document!</div>
             </div>
          </div>
          <div class="relative">
             <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3 rounded-full bg-slate-50 border border-slate-200 text-sm focus:outline-none focus:border-teal-500" placeholder="Type a question...">
             <button id="chat-send" onclick="sendMessage()" class="absolute right-1 top-1 p-2 bg-teal-600 text-white rounded-full w-8 h-8 flex items-center justify-center"><i class="fa-solid fa-paper-plane text-xs"></i></button>
          </div>
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>
      </section>

    </div>
  </main>

  <script>
    async function sendMessage() {
        const input = document.getElementById('chat-input');
        const panel = document.getElementById('chat-panel');
        const txt = input.value.trim();
        if(!txt) return;

        // Add User Message
        panel.innerHTML += `<div class="flex gap-3 flex-row-reverse"><div class="bg-slate-800 text-white rounded-2xl rounded-tr-none p-3 text-xs">${txt}</div></div>`;
        input.value = '';
        panel.scrollTop = panel.scrollHeight;

        // Fetch API
        const res = await fetch('{{ url_for("chat") }}', {
            method: 'POST',
            body: JSON.stringify({ message: txt, doc_text: document.getElementById('doc-context').value })
        });
        const data = await res.json();
        
        // Add Bot Message
        panel.innerHTML += `<div class="flex gap-3"><div class="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center text-teal-600 text-xs shrink-0"><i class="fa-solid fa-robot"></i></div><div class="bg-slate-100 text-slate-700 rounded-2xl rounded-tl-none p-3 text-xs">${data.reply}</div></div>`;
        panel.scrollTop = panel.scrollHeight;
    }
  </script>
</body>
</html>
"""

# ---------------------- UTILITIES ---------------------- #

def clean_text(text: str) -> str:
    # 1. Remove "Page X"
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    # 2. Remove Section Headers (e.g., "3.3.3 Re-Orienting Public Hospitals:")
    text = re.sub(r'\d+(\.\d+)*\s+[A-Za-z\s\-]+:', '', text)
    # 3. Remove leading section numbers (e.g., "2.4.1")
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text)
    text = re.sub(r'\s+', " ", text)
    return text.strip()

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        text = " ".join([pg.extract_text() for pg in reader.pages if pg.extract_text()])
        return text
    except:
        return ""

# ---------------------- CATEGORIZATION ---------------------- #
POLICY_KEYWORDS = {
    "key goals": ["aim", "goal", "target", "achieve", "reduce", "increase", "%", "2025"],
    "principles": ["equity", "universal", "access", "quality", "ethics", "rights"],
    "delivery": ["hospital", "primary care", "referral", "ambulance", "emergency", "drug"],
    "prevention": ["prevent", "sanitation", "immunization", "tobacco", "lifestyle"],
    "hr": ["doctor", "nurse", "training", "workforce", "recruit", "incentive"],
    "finance": ["fund", "budget", "cost", "insurance", "private", "spending", "gdp"],
    "digital": ["digital", "technology", "data", "record", "telemedicine", "online"],
    "ayush": ["ayush", "ayurveda", "yoga", "unani", "traditional"],
}

def score_category(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    for cat, kws in POLICY_KEYWORDS.items():
        for kw in kws:
            if kw in s_lower: scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"

# ---------------------- SUMMARIZATION LOGIC ---------------------- #

def summarize_supervised(text: str, num_sents=10):
    """Uses the trained Logistic Regression model."""
    if not has_trained_model:
        return summarize_textrank(text, num_sents) # Fallback

    try:
        raw_sents = nltk.sent_tokenize(text)
        clean_sents = [clean_text(s) for s in raw_sents if len(s) > 30]
        
        if not clean_sents: return []

        # Predict importance
        features = vectorizer.transform(clean_sents)
        scores = model.predict_proba(features)[:, 1] # Probability of being summary-worthy

        # Rank
        ranked = sorted(zip(clean_sents, scores), key=lambda x: x[1], reverse=True)
        top_sents = [s[0] for s in ranked[:num_sents]]
        
        return top_sents
    except Exception as e:
        print(f"Supervised Error: {e}")
        return summarize_textrank(text, num_sents)

def summarize_textrank(text: str, num_sents=10):
    """Fallback unsupervised method if model missing."""
    sents = nltk.sent_tokenize(text)
    if len(sents) < num_sents: return sents
    
    vec = TfidfVectorizer().fit_transform(sents)
    sim_mat = cosine_similarity(vec)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    ranked = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
    return [s for _, s in ranked[:num_sents]]

def build_structure(sentences: List[str]):
    cat_map = defaultdict(list)
    for s in sentences:
        cat = score_category(s)
        cat_map[cat].append(s)
    
    sections = []
    titles = {
        "key goals": "Key Goals & Targets", "principles": "Core Principles",
        "delivery": "Service Delivery", "prevention": "Prevention Strategy",
        "hr": "Human Resources", "finance": "Financing & Private Sector",
        "digital": "Digital Health", "ayush": "AYUSH Integration", "other": "General Points"
    }
    
    for k, title in titles.items():
        if cat_map[k]:
            sections.append({"title": title, "bullets": list(set(cat_map[k]))})
            
    abstract = " ".join(sentences[:3])
    return abstract, sections

# ---------------------- PDF EXPORT ---------------------- #

def create_pdf(filename, abstract, sections):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Policy Brief Summary")
    y -= 30
    
    c.setFont("Helvetica-Oblique", 10)
    lines = simpleSplit(abstract, "Helvetica-Oblique", 10, width - 2*margin)
    for l in lines:
        c.drawString(margin, y, l)
        y -= 12
    y -= 20
    
    for sec in sections:
        if y < 100: c.showPage(); y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, sec['title'])
        y -= 15
        c.setFont("Helvetica", 10)
        for b in sec['bullets']:
            blines = simpleSplit(f"• {b}", "Helvetica", 10, width - 2*margin)
            for l in blines:
                c.drawString(margin, y, l)
                y -= 12
            y -= 4
        y -= 10
    c.save()

# ---------------------- ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/summaries/<path:filename>")
def summary_file(filename):
    return send_from_directory(app.config["SUMMARY_FOLDER"], filename, as_attachment=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    msg = data.get("message", "")
    doc = data.get("doc_text", "")
    
    if not GEMINI_API_KEY: return jsonify({"reply": "Gemini API Key missing."})
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat_sess = model.start_chat(history=[])
        prompt = f"Context: {doc[:10000]}\nQuestion: {msg}\nAnswer shortly."
        resp = chat_sess.send_message(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/summarize", methods=["POST"])
def summarize():
    f = request.files.get("file")
    if not f or f.filename == "": abort(400)
    
    filename = secure_filename(f.filename)
    uid = uuid.uuid4().hex
    path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{filename}")
    f.save(path)
    
    # Process
    text = ""
    if filename.endswith(".pdf"):
        with open(path, "rb") as pdf_file:
            text = extract_text_from_pdf_bytes(pdf_file.read())
    else:
        # Image processing via Gemini if needed
        if GEMINI_API_KEY and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            model = genai.GenerativeModel("gemini-1.5-flash")
            img = Image.open(path)
            text = model.generate_content(["Extract text", img]).text
    
    if len(text) < 50: abort(400, "Could not extract text.")
    
    # Summarize (Supervised -> Fallback)
    summary_sents = summarize_supervised(text, num_sents=15)
    abstract, sections = build_structure(summary_sents)
    
    # Generate PDF
    pdf_name = f"{uid}_summary.pdf"
    create_pdf(os.path.join(SUMMARY_FOLDER, pdf_name), abstract, sections)
    
    return render_template_string(
        RESULT_HTML,
        abstract=abstract,
        sections=sections,
        doc_context=text[:15000],
        summary_pdf_url=url_for("summary_file", filename=pdf_name),
        model_used="Supervised Model" if has_trained_model else "Fallback Mode"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
