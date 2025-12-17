import io
import os
import re
import uuid
import json
import pickle
import time
from collections import defaultdict
from typing import List, Dict

import numpy as np
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
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import google.generativeai as genai

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# Ensure NLTK data is downloaded (Required for Supervised Model splitting)
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")
MODEL_DIR = "."  # Root directory for pkl files

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Configure Gemini (Optional, for Image support)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_API_KEY = None

# ==============================================================================
# 2. LOAD SUPERVISED MODEL
# ==============================================================================
model = None
vectorizer = None
model_path = os.path.join(MODEL_DIR, 'model.pkl')
tfidf_path = os.path.join(MODEL_DIR, 'tfidf.pkl')

if os.path.exists(model_path) and os.path.exists(tfidf_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Supervised Model & Vectorizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Warning: 'model.pkl' or 'tfidf.pkl' not found. App will fail on summarization.")


# ==============================================================================
# 3. SUPERVISED MODEL LOGIC (Preserved Workflow)
# ==============================================================================

def clean_text(text):
    """
    Aggressive cleaning to remove headers, page numbers, and artifacts.
    Matches the logic used during training.
    """
    # 1. Remove "Page X" or "--- PAGE ---"
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'---.*?---', '', text)
    
    # 2. Remove Section Headers with Titles (e.g., "3.3.3 Re-Orienting Public Hospitals:")
    text = re.sub(r'\d+(\.\d+)*\s+[A-Za-z\s\-]+:', '', text)
    
    # 3. Remove standalone leading section numbers (e.g., "2.4.1")
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text)
    
    # 4. Remove extra whitespace
    return text.strip()

def generate_extractive_summary_supervised(text, length_choice="medium"):
    """
    Uses the Supervised Logistic Regression Model to score sentences.
    """
    if not model or not vectorizer:
        return ["Error: Supervised model not active. Please check server logs."]

    # 1. Map UI length choice to number of sentences
    # Short ~ 12, Medium ~ 25, Long ~ 50
    num_sentences = 25
    if length_choice == "short": num_sentences = 12
    elif length_choice == "long": num_sentences = 50

    # 2. Split into sentences (Using NLTK as per training)
    try:
        raw_sentences = nltk.sent_tokenize(text)
    except:
        return ["Error processing text."]

    # 3. Clean sentences
    clean_sentences = [clean_text(s) for s in raw_sentences]
    
    # 4. Filter out junk (empty or very short lines)
    valid_sentences = []
    
    for i, s in enumerate(clean_sentences):
        # Must be at least 40 chars and contain letters
        if len(s) > 40 and re.search('[a-zA-Z]', s):
            valid_sentences.append(s)

    if not valid_sentences:
        return ["No valid text found in document."]

    # 5. Predict Importance Scores (The Supervised Step)
    try:
        features = vectorizer.transform(valid_sentences)
        # Get probability of Class 1 (Important)
        scores = model.predict_proba(features)[:, 1]
    except Exception as e:
        return [f"Prediction Error: {e}"]
    
    # 6. Smart Selection Loop (Redundancy Filter)
    # Sort indices by score (Highest first)
    ranked_indices = np.argsort(scores)[::-1]
    
    selected_indices = []
    selected_vectors = []
    
    for idx in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break
            
        current_vec = features[idx]
        
        # Check if this sentence is too similar to one we already picked
        is_redundant = False
        if selected_vectors:
            # Check cosine similarity against all selected sentences
            # vstack creates a matrix of selected vectors to compare against current one
            sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
            
            # If >65% similar to any existing sentence, skip it
            if np.max(sims) > 0.65:
                is_redundant = True
        
        if not is_redundant:
            selected_indices.append(idx)
            # Convert sparse vector to dense array for storage
            selected_vectors.append(current_vec.toarray()[0])

    # 7. Sort back by original index to maintain document flow
    final_sentences_indices = sorted(selected_indices)
    
    # Retrieve the text using the valid_sentences list
    final_sentences = [valid_sentences[i] for i in final_sentences_indices]
    
    return final_sentences

# ==============================================================================
# 4. UI HELPERS (Categorization & PDF)
# ==============================================================================

# Expanded Dictionary for Higher Accuracy Categorization in UI
POLICY_KEYWORDS = {
    "key goals": ["aim", "goal", "objective", "target", "achieve", "reduce", "increase", "coverage", "mortality", "rate", "%", "2025", "2030", "vision", "outcome"],
    "policy principles": ["principle", "equity", "universal", "right", "access", "accountability", "transparency", "inclusive", "patient-centered", "quality", "ethics"],
    "service delivery": ["hospital", "primary care", "secondary care", "tertiary", "referral", "clinic", "health center", "wellness", "ambulance", "emergency", "drug", "diagnostic"],
    "prevention & promotion": ["prevent", "sanitation", "nutrition", "immunization", "vaccine", "tobacco", "alcohol", "hygiene", "awareness", "lifestyle", "pollution"],
    "human resources": ["doctor", "nurse", "staff", "training", "workforce", "recruit", "medical college", "paramedic", "salary", "incentive", "capacity building"],
    "financing & private sector": ["fund", "budget", "finance", "expenditure", "cost", "insurance", "private", "partnership", "ppp", "out-of-pocket", "reimbursement"],
    "digital health": ["digital", "technology", "data", "record", "ehr", "telemedicine", "mobile", "app", "information system", "cyber", "ai"],
    "ayush integration": ["ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy", "traditional"]
}

def score_sentence_categories(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    words = re.findall(r'\w+', s_lower)
    for cat, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in s_lower: scores[cat] += 2
    if '%' in s_lower or re.search(r'\b20[2-5][0-9]\b', s_lower): scores['key goals'] += 2
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0: return "other"
    return best_cat

def build_structured_summary(summary_sentences: List[str], tone: str):
    # 1. Simple Tone: Return as single clean paragraph
    if tone == "easy":
        text_block = " ".join(summary_sentences)
        text_block = re.sub(r'\s+', ' ', text_block)
        return {
            "abstract": summary_sentences[0] if summary_sentences else "No abstract generated.",
            "sections": [],
            "simple_text": text_block,
            "category_counts": {}
        }

    # 2. Academic/Technical Tone: Use Categorization
    cat_map = defaultdict(list)
    for s in summary_sentences:
        category = score_sentence_categories(s)
        cat_map[category].append(s)
    
    section_titles = {
        "key goals": "Key Goals & Targets", 
        "policy principles": "Policy Principles & Vision",
        "service delivery": "Healthcare Delivery Systems", 
        "prevention & promotion": "Prevention & Wellness",
        "human resources": "Workforce (HR)", 
        "financing & private sector": "Financing & Costs",
        "digital health": "Digital Interventions", 
        "ayush integration": "AYUSH / Traditional Medicine",
        "other": "Other Key Observations"
    }
    
    sections = []
    for k, title in section_titles.items():
        if cat_map[k]:
            unique = list(dict.fromkeys(cat_map[k]))
            sections.append({"title": title, "bullets": unique})
            
    # Abstract: Just take the first 3 highly ranked sentences
    abstract = " ".join(summary_sentences[:3]) if summary_sentences else ""
    
    return {
        "abstract": abstract,
        "sections": sections,
        "simple_text": None,
        "category_counts": {k: len(v) for k, v in cat_map.items()}
    }

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    text = ""
    for pg in reader.pages:
        t = pg.extract_text()
        if t: text += t + " "
    return text

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], simple_text: str, out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Abstract")
    y -= 15
    
    c.setFont("Helvetica", 10)
    if abstract:
        lines = simpleSplit(abstract, "Helvetica", 10, width - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
    y -= 10
    
    if simple_text:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Full Summary")
        y -= 15
        c.setFont("Helvetica", 10)
        lines = simpleSplit(simple_text, "Helvetica", 10, width - 2*margin)
        for line in lines:
            if y < 50: c.showPage(); y = height - margin
            c.drawString(margin, y, line)
            y -= 12
    else:
        for sec in sections:
            if y < 100: c.showPage(); y = height - margin
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, sec["title"])
            y -= 15
            c.setFont("Helvetica", 10)
            for b in sec["bullets"]:
                blines = simpleSplit(f"• {b}", "Helvetica", 10, width - 2*margin)
                for l in blines:
                    c.drawString(margin, y, l)
                    y -= 12
                y -= 4
            y -= 10
    c.save()

def process_images_with_gemini(image_paths: List[str]):
    if not GEMINI_API_KEY: return None, "Gemini API Key missing."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        images = []
        for p in image_paths:
            img = Image.open(p)
            img.thumbnail((256, 256))
            images.append(img)
        
        prompt = """Analyze these images. Extract main text and create a JSON summary: 
        {"extracted_text": "...", "summary_structure": {"abstract": "...", "sections": [{"title": "...", "bullets": ["..."]}]}}"""
        
        response = model.generate_content([prompt] + images)
        text_resp = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text_resp), None
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 5. HTML TEMPLATES (THE FANCY UI)
# ==============================================================================

COMMON_HEAD = """
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            darkMode: "class",
            theme: { extend: { colors: { "background-dark": "#0D0D0F", "surface-dark": "#161b22", "afzal-purple": "#8C4FFF", "afzal-blue": "#4D9CFF" }, fontFamily: { sans: ['Inter', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] } } }
        };
    </script>
    <style>
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0D0D0F; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        .goog-te-banner-frame { display: none !important; }
        body { top: 0 !important; }
        .fade-up { opacity: 0; transform: translateY(20px); transition: opacity 0.6s ease-out, transform 0.6s ease-out; }
    </style>
"""

COMMON_SCRIPTS = """
<script>
    document.addEventListener('DOMContentLoaded', () => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => { if (entry.isIntersecting) { entry.target.style.opacity = '1'; entry.target.style.transform = 'translateY(0)'; observer.unobserve(entry.target); } });
        });
        document.querySelectorAll('.fade-up').forEach(el => observer.observe(el));
        setTimeout(() => { document.querySelectorAll('.fade-up').forEach(el => { el.style.opacity = '1'; el.style.transform = 'translateY(0)'; }); }, 1000);
    });
</script>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Med.AI | Supervised Policy Summarizer</title>
    {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark overflow-x-hidden">
<nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full flex items-center justify-between px-6">
        <div class="flex items-center space-x-2">
            <div class="w-8 h-8 bg-white rounded-full flex items-center justify-center"><i class="fa-solid fa-staff-snake text-black"></i></div>
            <span class="font-bold text-2xl text-white">Med.AI</span>
        </div>
        <span class="text-afzal-purple font-mono text-xs uppercase border border-afzal-purple/30 bg-afzal-purple/10 px-3 py-1 rounded">Supervised Model Active</span>
    </div>
</nav>

<header class="relative pt-20 pb-12 text-center fade-up">
    <h1 class="text-5xl md:text-7xl font-semibold text-white mb-6">SUPERVISED <span class="text-transparent bg-clip-text bg-gradient-to-r from-afzal-purple to-afzal-blue">SUMMARIZATION</span></h1>
    <p class="text-xl text-gray-400 max-w-2xl mx-auto">Upload healthcare policies. Our Logistic Regression model identifies critical clauses and generates structured summaries.</p>
</header>

<section class="py-12 fade-up">
    <div class="max-w-4xl mx-auto px-4">
        <div class="bg-surface-dark border border-gray-800 p-8 rounded-2xl shadow-2xl">
            <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
                <div class="relative w-full h-64 border-2 border-dashed border-gray-700 rounded-xl flex flex-col items-center justify-center hover:border-afzal-purple hover:bg-gray-800 transition cursor-pointer" onclick="document.getElementById('file-input').click()">
                    <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" multiple class="hidden">
                    <i class="fa-solid fa-cloud-arrow-up text-4xl text-afzal-purple mb-4"></i>
                    <p class="text-lg font-bold text-white">Click to upload PDF</p>
                </div>
                <div class="grid grid-cols-2 gap-6">
                    <div class="bg-black/20 p-4 rounded border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase mb-2">Length</label>
                        <select name="length" class="w-full bg-surface-dark border border-gray-700 rounded text-white p-2">
                            <option value="short">Short (12 sentences)</option>
                            <option value="medium" selected>Medium (25 sentences)</option>
                            <option value="long">Long (50 sentences)</option>
                        </select>
                    </div>
                    <div class="bg-black/20 p-4 rounded border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase mb-2">View Style</label>
                        <select name="tone" class="w-full bg-surface-dark border border-gray-700 rounded text-white p-2">
                            <option value="academic" selected>Structured (Categorized)</option>
                            <option value="easy">Simple Text</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="w-full bg-white text-black h-14 font-semibold uppercase hover:bg-gray-200 rounded">Generate Summary</button>
            </form>
        </div>
    </div>
</section>
{COMMON_SCRIPTS}
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Analysis Report | Med.AI</title>
    {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark">
<nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16 flex items-center justify-between px-6">
    <div class="flex items-center gap-4">
        <a href="/" class="flex items-center space-x-2"><div class="w-8 h-8 bg-white rounded-full flex items-center justify-center"><i class="fa-solid fa-staff-snake text-black"></i></div><span class="font-bold text-xl text-white">Med.AI</span></a>
        <span class="text-sm font-mono text-afzal-purple">/ Analysis Report</span>
    </div>
    <div class="flex items-center gap-4">
        <div id="google_translate_element"></div>
        <a href="/" class="text-xs font-bold uppercase text-white border border-gray-700 px-4 py-2 rounded hover:border-afzal-purple">New Analysis</a>
    </div>
</nav>

<main class="py-12 px-4 max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
  <section class="lg:col-span-7 space-y-6">
    <div class="bg-surface-dark border border-gray-800 p-8 rounded-2xl shadow-lg">
       <div class="flex justify-between items-start mb-6 border-b border-gray-700 pb-6">
         <div>
            <div class="flex items-center gap-2 mb-2"><span class="px-2 py-1 rounded text-[10px] font-bold uppercase bg-afzal-purple/10 text-afzal-purple">Supervised Model</span></div>
            <h1 class="text-3xl font-light text-white">Policy Summary</h1>
         </div>
         {% if summary_pdf_url %}
         <a href="{{ summary_pdf_url }}" class="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:bg-afzal-purple hover:text-white"><i class="fa-solid fa-file-arrow-down"></i></a>
         {% endif %}
       </div>
       
       <div class="mb-8">
          <h2 class="text-xs font-bold text-afzal-blue uppercase tracking-widest mb-4"><i class="fa-solid fa-layer-group"></i> Abstract</h2>
          <div class="p-6 rounded-lg bg-black/30 border border-gray-800 text-sm leading-relaxed text-gray-300 font-light">{{ abstract }}</div>
       </div>

       {% if simple_text %}
       <div class="mb-8"><div class="text-sm leading-7 text-gray-400 text-justify">{{ simple_text }}</div></div>
       {% endif %}

       {% if sections %}
       <div class="space-y-8">
          {% for sec in sections %}
          <div class="pl-6 border-l border-gray-800 hover:border-afzal-purple transition-colors group">
             <h3 class="text-lg font-medium text-white mb-3 group-hover:text-afzal-purple font-mono">{{ sec.title }}</h3>
             <ul class="space-y-3">
                {% for bullet in sec.bullets %}
                <li class="flex items-start gap-3 text-sm text-gray-400"><i class="fa-solid fa-angle-right mt-1 text-gray-600"></i><span>{{ bullet }}</span></li>
                {% endfor %}
             </ul>
          </div>
          {% endfor %}
       </div>
       {% endif %}
    </div>
  </section>

  <section class="lg:col-span-5 space-y-6">
    <div class="bg-surface-dark border border-gray-800 rounded-xl shadow-lg h-[400px] flex flex-col">
         <div class="p-3 border-b border-gray-800 bg-[#0d1117]"><h2 class="text-xs font-bold text-gray-400 uppercase">Source Document</h2></div>
         <div class="flex-1 overflow-auto p-4 bg-black/40 font-mono text-xs text-gray-400 whitespace-pre-wrap custom-scrollbar">{{ orig_text[:10000] }}...</div>
    </div>
    <div class="bg-surface-dark border border-gray-800 rounded-xl shadow-lg h-[500px] flex flex-col">
         <div class="p-4 bg-[#0d1117] border-b border-gray-800 flex items-center gap-3">
             <div class="w-8 h-8 rounded-full bg-afzal-purple/20 flex items-center justify-center text-afzal-purple"><i class="fa-solid fa-robot"></i></div>
             <div><h2 class="text-sm font-bold text-white">Med.AI Assistant</h2><p class="text-[10px] text-gray-400">CONTEXT AWARE</p></div>
         </div>
         <div id="chat-panel" class="flex-1 overflow-y-auto p-4 space-y-4 bg-[#0d1117]">
             <div class="flex gap-3"><div class="w-8 h-8 rounded-full bg-afzal-purple flex items-center justify-center text-white text-xs"><i class="fa-solid fa-robot"></i></div><div class="bg-[#1c2128] border border-gray-700 p-3 rounded-2xl rounded-tl-none text-xs text-gray-300">I have analyzed the document using the Supervised Model. Ask me anything about it.</div></div>
         </div>
         <div class="p-3 bg-[#161b22] border-t border-gray-800 relative">
             <input type="text" id="chat-input" class="w-full pl-4 pr-10 py-3 rounded-full bg-[#0d1117] border border-gray-700 text-sm text-white focus:border-afzal-purple outline-none" placeholder="Ask a question...">
             <button id="chat-send" class="absolute right-5 top-5 text-afzal-purple"><i class="fa-solid fa-paper-plane"></i></button>
         </div>
         <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
    </div>
  </section>
</main>
<script>
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const docText = document.getElementById('doc-context').value;

    function addMsg(role, text) {
        const div = document.createElement('div');
        div.className = role === 'user' ? 'flex gap-3 flex-row-reverse' : 'flex gap-3';
        div.innerHTML = `<div class="w-8 h-8 rounded-full flex items-center justify-center text-xs shrink-0 border shadow-md ${role === 'user' ? 'bg-white text-black' : 'bg-afzal-purple text-white'}"><i class="fa-solid ${role === 'user' ? 'fa-user' : 'fa-robot'}"></i></div><div class="max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed border ${role === 'user' ? 'bg-afzal-purple text-white border-afzal-purple rounded-tr-none' : 'bg-[#1c2128] text-gray-300 border-gray-700 rounded-tl-none'}">${text}</div>`;
        panel.appendChild(div);
        panel.scrollTop = panel.scrollHeight;
    }

    async function sendMessage() {
        const txt = input.value.trim();
        if(!txt) return;
        addMsg('user', txt);
        input.value = '';
        try {
            const res = await fetch('{{ url_for("chat") }}', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ message: txt, doc_text: docText }) });
            const data = await res.json();
            addMsg('assistant', data.reply);
        } catch(e) { addMsg('assistant', "Connection interrupted."); }
    }
    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }
</script>
<script>
    function googleTranslateElementInit() { new google.translate.TranslateElement({ pageLanguage: 'en', includedLanguages: 'en,hi,te', layout: google.translate.TranslateElement.InlineLayout.SIMPLE }, 'google_translate_element'); }
</script>
<script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
{COMMON_SCRIPTS}
</body>
</html>
"""

INDEX_HTML = INDEX_HTML.replace("{COMMON_HEAD}", COMMON_HEAD).replace("{COMMON_SCRIPTS}", COMMON_SCRIPTS)
RESULT_HTML = RESULT_HTML.replace("{COMMON_HEAD}", COMMON_HEAD).replace("{COMMON_SCRIPTS}", COMMON_SCRIPTS)

# ==============================================================================
# 6. ROUTING LOGIC
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/summaries/<path:filename>")
def summary_file(filename):
    return send_from_directory(app.config["SUMMARY_FOLDER"], filename, as_attachment=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    doc_text = data.get("doc_text", "")
    
    if not GEMINI_API_KEY:
        return jsonify({"reply": "Gemini Key not configured."})
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        chat = model.start_chat(history=[])
        prompt = f"Context from document: {doc_text[:30000]}\n\nUser Question: {message}\nAnswer concisely."
        resp = chat.send_message(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/summarize", methods=["POST"])
def summarize():
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        abort(400, "No file uploaded")
    
    # Save Files
    saved_paths = []
    saved_urls = []
    uid = uuid.uuid4().hex
    is_multi_image = False
    valid_img_exts = ('.png', '.jpg', '.jpeg', '.webp')

    for f in files:
        fname = secure_filename(f.filename)
        stored_name = f"{uid}_{fname}"
        stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        f.save(stored_path)
        saved_paths.append(stored_path)
        saved_urls.append(url_for("uploaded_file", filename=stored_name))

    # Check Type
    first_name_lower = files[0].filename.lower()
    if len(files) > 1 or first_name_lower.endswith(valid_img_exts):
        is_multi_image = True

    structured_data = {}
    orig_text = ""
    orig_type = "unknown"
    used_model = "supervised" 

    # CASE 1: IMAGES (Fallback to Gemini)
    if is_multi_image:
        orig_type = "image"
        used_model = "gemini"
        gemini_data, err = process_images_with_gemini(saved_paths)
        if err or not gemini_data: abort(500, f"Gemini Failed: {err}")
        orig_text = gemini_data.get("extracted_text", "")
        structured_data = gemini_data.get("summary_structure", {})

    # CASE 2: PDF/TXT -> SUPERVISED MODEL
    else:
        stored_path = saved_paths[0]
        with open(stored_path, "rb") as f_in:
            raw_bytes = f_in.read()
            
        if first_name_lower.endswith(".pdf"):
            orig_type = "pdf"
            orig_text = extract_text_from_pdf_bytes(raw_bytes)
        else:
            orig_type = "text"
            orig_text = raw_bytes.decode("utf-8", errors="ignore")
            
        if len(orig_text) < 50: abort(400, "Not enough text found.")
        
        # Get options
        length = request.form.get("length", "medium")
        tone = request.form.get("tone", "academic")
        
        # --- HERE IS THE SWITCH TO SUPERVISED MODEL ---
        sentences = generate_extractive_summary_supervised(orig_text, length)
        # --- END SWITCH ---
        
        # Feed supervised sentences into UI Structure Builder
        structured_data = build_structured_summary(sentences, tone)

    # Generate PDF
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    save_summary_pdf(
        "Policy Summary",
        structured_data.get("abstract", ""),
        structured_data.get("sections", []),
        structured_data.get("simple_text", None),
        summary_path
    )
    
    return render_template_string(
        RESULT_HTML,
        title="Med.AI Summary",
        orig_type=orig_type,
        orig_url=saved_urls[0],
        orig_images=saved_urls if orig_type == 'image' else [],
        orig_text=orig_text[:20000], 
        doc_context=orig_text[:20000],
        abstract=structured_data.get("abstract", ""),
        sections=structured_data.get("sections", []),
        simple_text=structured_data.get("simple_text", None),
        summary_pdf_url=url_for("summary_file", filename=summary_filename),
        used_model=used_model
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
