# app.py
import io
import os
import re
import uuid
import pickle
from typing import List, Dict
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Optional OCR for images (if user drops images)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# PDF generation for download
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

nltk.download('punkt', quiet=True)

# ---------------------- CONFIG & MODEL LOADING ---------------------- #

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Supervised model files (expected in same directory)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
    try:
        with open(MODEL_PATH, "rb") as mf:
            model = pickle.load(mf)
        with open(TFIDF_PATH, "rb") as vf:
            vectorizer = pickle.load(vf)
    except Exception as e:
        print("Error loading model/vectorizer:", e)

# ---------------------- UTILITIES ---------------------- #

def clean_text(text: str) -> str:
    """Removes headers, page numbers, and common artifacts."""
    text = re.sub(r'Page \d+( of \d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for pg in reader.pages:
            txt = pg.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_pdf_stream(stream) -> str:
    try:
        reader = PdfReader(stream)
        pages = []
        for pg in reader.pages:
            txt = pg.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception:
        return ""

def ocr_image_bytes(raw: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(raw))
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""

def save_summary_pdf(title: str, summary_text: str, out_path: str):
    if not PDF_AVAILABLE:
        # fallback: write plain text file
        with open(out_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
            f.write(title + "\n\n")
            f.write(summary_text)
        return out_path.replace('.pdf', '.txt')
    try:
        c = canvas.Canvas(out_path, pagesize=A4)
        width, height = A4
        margin = 50
        y = height - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, title)
        y -= 30
        c.setFont("Helvetica", 10)
        lines = simpleSplit(summary_text, "Helvetica", 10, width - 2*margin)
        for line in lines:
            if y < margin + 30:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.drawString(margin, y, line)
            y -= 12
        c.save()
        return out_path
    except Exception:
        # fallback to text file
        txt_path = out_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(title + "\n\n")
            f.write(summary_text)
        return txt_path

# ---------------------- SUPERVISED EXTRACTIVE SUMMARY ---------------------- #

def generate_extractive_summary(text: str, num_sentences: int = 7) -> str:
    if model is None or vectorizer is None:
        return "Model or vectorizer not loaded on server."

    # Sentence split using nltk
    raw_sentences = nltk.sent_tokenize(text)
    clean_sentences = [clean_text(s) for s in raw_sentences]

    # Filter out too-short content (likely headers/footers)
    valid_sentences = []
    original_indices = []
    for i, s in enumerate(clean_sentences):
        if s and len(s) > 40:
            valid_sentences.append(s)
            original_indices.append(i)

    if not valid_sentences:
        return "No valid text found to summarize."

    # Vectorize
    try:
        features = vectorizer.transform(valid_sentences)
    except Exception as e:
        return f"Vectorizer transform error: {e}"

    try:
        # If classifier supports predict_proba
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(features)[:, 1]
        else:
            # fallback to decision_function or predict
            if hasattr(model, "decision_function"):
                raw_scores = model.decision_function(features)
                # normalize into 0-1
                smin, smax = raw_scores.min(), raw_scores.max()
                if smax - smin == 0:
                    scores = np.ones_like(raw_scores)
                else:
                    scores = (raw_scores - smin) / (smax - smin)
            else:
                preds = model.predict(features)
                scores = np.array(preds, dtype=float)
    except Exception as e:
        return f"Model scoring error: {e}"

    # Rank sentences by score
    ranked_indices = np.argsort(scores)[::-1]

    selected_indices = []
    selected_vectors = []

    for idx in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break

        current_vec = features[idx]
        is_redundant = False
        if selected_vectors:
            sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
            if np.max(sims) > 0.65:
                is_redundant = True

        if not is_redundant:
            selected_indices.append(idx)
            selected_vectors.append(current_vec.toarray()[0])

    # Sort selected sentences back into document order to preserve flow
    final_order = sorted(selected_indices)
    summary = " ".join([valid_sentences[i] for i in final_order])

    return summary.strip() if summary else "Could not create summary."

# ---------------------- HTML TEMPLATES (YELLOW THEME) ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>PolicyBrief AI — Supervised</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Inter', 'sans-serif'] },
          colors: {
            amber: {
              50: '#fff8e1', 100: '#ffecb3', 200: '#ffe082', 300: '#ffd54f',
              400: '#ffca28', 500: '#f59e0b', 600: '#d97706', 700: '#b45309',
              800: '#92400e', 900: '#78350f'
            }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #fffdf7; }
    .glass-panel {
      background: rgba(255,255,255,0.7);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255,255,255,0.6);
    }
    @keyframes progress-stripes {
      from { background-position: 1rem 0; }
      to { background-position: 0 0; }
    }
    .animate-stripes {
      background-image: linear-gradient(45deg, rgba(255,255,255,.15) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.15) 50%, rgba(255,255,255,.15) 75%, transparent 75%, transparent);
      background-size: 1rem 1rem;
      animation: progress-stripes 1s linear infinite;
    }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden min-h-screen flex flex-col">

  <div class="fixed top-[-12%] left-[-8%] w-[36%] h-[36%] bg-amber-100/50 rounded-full blur-3xl -z-10 animate-pulse"></div>
  <nav class="fixed w-full z-40 glass-panel border-b border-slate-200/40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-amber-400 to-amber-500 rounded-xl flex items-center justify-center shadow-lg text-white">
            <i class="fa-solid fa-file-lines"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">
            PolicyBrief<span class="text-amber-600">.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-6 text-xs font-bold uppercase tracking-wider text-slate-500">
          <span>Supervised Extractive Summarizer</span>
          <a href="#workspace" class="px-5 py-2.5 rounded-full bg-amber-600 text-white hover:bg-amber-700 transition shadow-lg">
            Start Now
          </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="flex-grow pt-28 pb-20 px-4">
    <div class="max-w-5xl mx-auto">
      <div class="text-center space-y-6 mb-16">
        <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-50 border border-amber-100 text-amber-700 text-xs font-bold uppercase tracking-wide">
          Supervised model — tuned TF-IDF + classifier
        </div>
        <h1 class="text-5xl md:text-6xl font-extrabold text-slate-900 leading-tight">
          Summarize Complex Policies
          <br>
          <span style="background:linear-gradient(90deg,#f59e0b,#fbbf24);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Faster · Clearer · Yellow</span>
        </h1>
        <p class="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Upload PDF or plain text and our supervised extractive model will return a concise executive summary.
        </p>
      </div>

      <div id="workspace" class="glass-panel rounded-3xl p-1 shadow-2xl shadow-slate-200/40 max-w-3xl mx-auto">
        <div class="bg-white/60 rounded-[1.3rem] p-6 md:p-10 border border-white/50">
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
            <div class="group relative w-full h-64 border-3 border-dashed border-slate-300 rounded-2xl bg-slate-50/50 hover:bg-amber-50/30 hover:border-amber-300 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105">
                <div class="w-16 h-16 bg-white rounded-full shadow-md flex items-center justify-center mx-auto text-amber-500 text-2xl group-hover:text-amber-600">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div>
                  <p class="text-lg font-bold text-slate-700">Click to upload or Drag & Drop</p>
                  <p class="text-sm text-slate-500 mt-1">PDF or TXT (Images use OCR if available)</p>
                </div>
                <div class="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm text-xs font-bold text-slate-600 uppercase tracking-wide border border-slate-200">
                  <i class="fa-solid fa-file-lines"></i> Policy Files
                </div>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-white/90 backdrop-blur-sm z-10 flex flex-col items-center justify-center p-6 text-center animate-fade-in">
                 <div id="preview-icon" class="mb-4 text-4xl text-amber-600"></div>
                 <div id="preview-image-container" class="mb-4 hidden rounded-lg overflow-hidden shadow-lg border border-slate-200 max-h-32">
                    <img id="preview-image" src="" alt="Preview" class="h-full object-contain">
                 </div>
                 <p id="filename-display" class="font-bold text-slate-800 text-lg break-all max-w-md"></p>
                 <p class="text-xs text-amber-600 font-semibold mt-2 uppercase tracking-wider">Ready to Summarize</p>
                 <button type="button" id="change-file-btn" class="mt-4 text-xs text-slate-400 hover:text-slate-600 underline z-30 relative">Change file</button>
              </div>

            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Summary Length</label>
                <div class="flex bg-slate-100 rounded-lg p-1">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="short" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">Short</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="medium" checked class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">Medium</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="long" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">Long</span>
                  </label>
                </div>
              </div>

              <div class="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Sentences (override)</label>
                <input type="number" name="num_sentences" min="1" max="30" placeholder="Optional: exact number of sentences" class="w-full rounded-md border border-slate-200 p-2 text-sm" />
                <p class="text-xs text-slate-400 mt-2">If provided, overrides Summary Length.</p>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold text-lg shadow-lg hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-2">
              <i class="fa-solid fa-wand-magic-sparkles"></i> Generate Summary
            </button>

          </form>
        </div>
      </div>

    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-white/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-md px-6 text-center space-y-6">
      <div class="relative w-20 h-20 mx-auto">
        <div class="absolute inset-0 rounded-full border-4 border-slate-100"></div>
        <div class="absolute inset-0 rounded-full border-4 border-amber-500 border-t-transparent animate-spin"></div>
        <div class="absolute inset-0 flex items-center justify-center text-amber-600 font-bold text-xl" id="progress-text">0%</div>
      </div>

      <div class="space-y-2">
        <h3 class="text-xl font-bold text-slate-900" id="progress-stage">Starting...</h3>
        <p class="text-sm text-slate-500">Please wait while we analyze your document.</p>
      </div>

      <div class="w-full h-3 bg-slate-200 rounded-full overflow-hidden relative">
        <div id="progress-bar" class="h-full bg-gradient-to-r from-amber-400 to-amber-600 animate-stripes w-0 transition-all duration-300 ease-out"></div>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const filePreview = document.getElementById('file-preview');
    const filenameDisplay = document.getElementById('filename-display');
    const previewIcon = document.getElementById('preview-icon');
    const previewImgContainer = document.getElementById('preview-image-container');
    const previewImg = document.getElementById('preview-image');
    const changeBtn = document.getElementById('change-file-btn');
    const uploadForm = document.getElementById('uploadForm');
    const progressOverlay = document.getElementById('progress-overlay');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressStage = document.getElementById('progress-stage');

    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files[0]) {
        const file = this.files[0];
        const reader = new FileReader();
        uploadPrompt.classList.add('opacity-0');
        setTimeout(() => {
            uploadPrompt.classList.add('hidden');
            filePreview.classList.remove('hidden');
        }, 250);

        filenameDisplay.textContent = file.name;
        previewImgContainer.classList.add('hidden');
        previewIcon.innerHTML = '';

        if (file.type.startsWith('image/')) {
           reader.onload = function(e) {
             previewImg.src = e.target.result;
             previewImgContainer.classList.remove('hidden');
           }
           reader.readAsDataURL(file);
        } else if (file.type === 'application/pdf') {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-pdf text-red-500"></i>';
        } else {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-lines text-slate-500"></i>';
        }
      }
    });

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        filePreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        uploadPrompt.classList.remove('opacity-0');
    });

    uploadForm.addEventListener('submit', function(e) {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert("Please select a file first.");
            return;
        }
        progressOverlay.classList.remove('hidden');
        progressOverlay.classList.add('flex');

        let width = 0;
        const fileType = fileInput.files[0].type;
        const isImage = fileType.startsWith('image/');

        const totalDuration = isImage ? 12000 : 4500;
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 97) {
                clearInterval(interval);
                progressStage.textContent = "Finalizing Summary...";
            } else {
                width += step;
                if(Math.random() > 0.6) width += 0.5;
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';
                if (width < 30) {
                    progressStage.textContent = "Uploading Document...";
                } else if (width < 70) {
                    progressStage.textContent = isImage ? "Running OCR..." : "Running model...";
                } else {
                    progressStage.textContent = "Composing summary...";
                }
            }
        }, intervalTime);
    });
  </script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>Summary Result</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Inter','sans-serif'] }
        }
      }
    }
  </script>
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-tr from-amber-400 to-amber-500 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-file-lines text-sm"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            PolicyBrief<span class="text-amber-600">.AI</span>
          </span>
        </div>
        <a href="{{ url_for('index') }}" class="inline-flex items-center px-4 py-2 text-xs font-bold rounded-full border border-slate-200 hover:border-amber-500 hover:text-amber-600 bg-white transition shadow-sm">
          <i class="fa-solid fa-plus mr-2"></i> New Summary
        </a>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">

      <section class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
          <div class="flex items-start justify-between gap-4 mb-6 border-b border-slate-100 pb-6">
            <div>
              <div class="flex items-center gap-2 mb-2">
                <span class="px-2 py-1 rounded-md bg-amber-50 text-amber-700 text-[0.65rem] font-bold uppercase tracking-wide border border-amber-100">
                   Supervised model
                </span>
              </div>
              <h1 class="text-2xl font-extrabold text-slate-900 leading-tight">Policy Summary</h1>
            </div>

            {% if summary_pdf_url %}
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-amber-600 transition shadow-lg">
              <i class="fa-solid fa-file-arrow-down mr-2"></i> Download Summary
            </a>
            {% endif %}
          </div>

          <div class="mb-8">
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
               <i class="fa-solid fa-align-left text-amber-500"></i> Executive Summary
            </h2>
            <div class="p-5 rounded-2xl bg-slate-50 border border-slate-100 text-sm leading-relaxed text-slate-700">
                {{ abstract }}
            </div>
          </div>

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        <div class="bg-white rounded-3xl shadow-xl border p-6">
          <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Original Document</h2>
          <div class="rounded-xl overflow-hidden border border-slate-200 bg-slate-100 h-[300px] relative group p-3 text-xs">
             {% if orig_type == 'pdf' %}
               <iframe src="{{ orig_url }}" class="w-full h-full" title="Original PDF"></iframe>
             {% elif orig_type == 'text' %}
               <div class="p-2 overflow-y-auto h-full font-mono">{{ orig_text }}</div>
             {% else %}
               <div class="p-2 overflow-y-auto h-full">{{ orig_text }}</div>
             {% endif %}
          </div>
        </div>

      </section>
    </div>
  </main>

</body>
</html>
"""

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

    # If image and OCR available: try OCR
    if lower_name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".webp")):
        orig_type = "image"
        if OCR_AVAILABLE:
            with open(stored_path, "rb") as imgf:
                raw = imgf.read()
                orig_text = ocr_image_bytes(raw)
        else:
            orig_text = "[OCR not available on server]"
    elif lower_name.endswith(".pdf"):
        orig_type = "pdf"
        try:
            with open(stored_path, "rb") as pf:
                orig_text = extract_text_from_pdf_stream(pf)
        except Exception:
            orig_text = ""
    else:
        orig_type = "text"
        try:
            with open(stored_path, "r", encoding="utf-8", errors="ignore") as tf:
                orig_text = tf.read()
        except Exception:
            orig_text = ""

    if not orig_text or len(orig_text.strip()) < 30:
        abort(400, "Not enough text could be extracted from the uploaded file.")

    # Determine number of sentences requested
    num_sent_str = request.form.get("num_sentences", "").strip()
    length_choice = request.form.get("length", "medium")

    if num_sent_str.isdigit():
        num_sent = max(1, min(30, int(num_sent_str)))
    else:
        # map length to number of sentences
        if length_choice == "short":
            num_sent = 3
        elif length_choice == "long":
            num_sent = 12
        else:
            num_sent = 7

    # Run supervised summarizer
    summary_text = generate_extractive_summary(orig_text, num_sentences=num_sent)

    # Create a short abstract for display (first 2 sentences of summary)
    abstract_display = summary_text

    # Save summary PDF (or text fallback)
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    saved_path = save_summary_pdf("Policy Summary", summary_text, summary_path)
    # if fallback created txt, make url point accordingly
    if saved_path.endswith(".txt"):
        summary_pdf_url = url_for("summary_file", filename=os.path.basename(saved_path))
    else:
        summary_pdf_url = url_for("summary_file", filename=summary_filename)

    # trim orig_text for display safety
    display_orig_text = orig_text[:20000]

    return render_template_string(
        RESULT_HTML,
        abstract=abstract_display,
        orig_type=orig_type,
        orig_url=url_for("uploaded_file", filename=stored_name),
        orig_text=display_orig_text,
        summary_pdf_url=summary_pdf_url,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
