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

# ---------------------- HTML TEMPLATES (Yellow Theme) ---------------------- #

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
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            amber: {
              50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d',
              400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', 700: '#b45309', 800: '#92400e', 900: '#78350f'
            }
          },
          animation: {
            'float': 'float 6s ease-in-out infinite',
            'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          },
          keyframes: {
            float: {
              '0%, 100%': { transform: 'translateY(0)' },
              '50%': { transform: 'translateY(-10px)' },
            }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f8fafc; }
    .glass-panel {
      background: rgba(255, 255, 255, 0.7);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.5);
    }
    .animate-stripes {
      background-image: linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent);
      background-size: 1rem 1rem;
      animation: progress-stripes 1s linear infinite;
    }
    @keyframes progress-stripes {
      from { background-position: 1rem 0; }
      to { background-position: 0 0; }
    }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden min-h-screen flex flex-col">

  <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-amber-100/30 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>
  <div class="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-amber-200/20 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>

  <nav class="fixed w-full z-40 glass-panel border-b border-slate-200/50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-amber-500 to-amber-400 rounded-xl flex items-center justify-center shadow-lg shadow-amber-400/20 text-white">
            <i class="fa-solid fa-file-lines text-lg"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">
            Policy<span class="text-amber-600">Brief</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-6 text-xs font-bold uppercase tracking-wider text-slate-500">
          <span>Supervised Extractive Summarizer</span>
          <a href="#workspace" class="px-5 py-2.5 rounded-full bg-slate-900 text-white hover:bg-slate-800 transition shadow-lg shadow-slate-900/20">
            Start Now
          </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="flex-grow pt-32 pb-20 px-4">
    <div class="max-w-5xl mx-auto">
      
      <div class="text-center space-y-6 mb-16">
        <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-50 border border-amber-100 text-amber-700 text-xs font-bold uppercase tracking-wide animate-float">
          <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
          Policy Brief Summaries
        </div>
        <h1 class="text-5xl md:text-6xl font-extrabold text-slate-900 leading-tight">
          Fast, Accurate <br>
          <span class="text-amber-600">Supervised Summaries</span>
        </h1>
        <p class="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Upload a PDF and our supervised model (trained on policy sentences) will extract the most important sentences. Yellow theme unlocked.
        </p>
      </div>

      <div id="workspace" class="glass-panel rounded-3xl p-1 shadow-2xl shadow-slate-200/50 max-w-3xl mx-auto">
        <div class="bg-white/50 rounded-[1.3rem] p-6 md:p-10 border border-white/50">
          
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
            
            <div class="group relative w-full h-64 border-3 border-dashed border-slate-300 rounded-2xl bg-slate-50/50 hover:bg-amber-50/30 hover:border-amber-400 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
              
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              
              <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105">
                <div class="w-16 h-16 bg-white rounded-full shadow-md flex items-center justify-center mx-auto text-amber-500 text-2xl group-hover:text-amber-600">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div>
                  <p class="text-lg font-bold text-slate-700">Click to upload or Drag & Drop</p>
                  <p class="text-sm text-slate-500 mt-1">PDF, TXT, or Image (JPG, PNG)</p>
                </div>
                <div class="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm text-xs font-bold text-slate-600 uppercase tracking-wide border border-slate-200">
                  <i class="fa-solid fa-camera"></i> Mobile Camera Ready
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
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Tone</label>
                <div class="flex bg-slate-100 rounded-lg p-1">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="academic" checked class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">Academic</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="easy" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">Simple</span>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold text-lg shadow-lg shadow-amber-400/30 hover:shadow-xl hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-2">
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
        }, 300);
        
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

        const totalDuration = isImage ? 12000 : 5000;
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 95) {
                clearInterval(interval);
                progressStage.textContent = "Finalizing Summary...";
            } else {
                width += step;
                if(Math.random() > 0.5) width += 0.5;

                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';

                if (width < 30) {
                    progressStage.textContent = "Uploading Document...";
                } else if (width < 70) {
                    progressStage.textContent = isImage ? "Extracting Text..." : "Running Supervised Model...";
                } else {
                    progressStage.textContent = "Structuring Sentences...";
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
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
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
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-tr from-amber-500 to-amber-400 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-file-lines text-sm"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            Policy<span class="text-amber-600">Brief</span>
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
        <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-8">
          
          <div class="flex flex-wrap items-start justify-between gap-4 mb-6 border-b border-slate-100 pb-6">
            <div>
              <div class="flex items-center gap-2 mb-2">
                 <span class="px-2 py-1 rounded-md bg-amber-50 text-amber-700 text-[0.65rem] font-bold uppercase tracking-wide border border-amber-100">
                   Supervised (Model)
                 </span>
              </div>
              <h1 class="text-2xl font-extrabold text-slate-900 leading-tight">Policy Summary (Extractive)</h1>
            </div>
            
            {% if summary_pdf_url %}
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-amber-600 transition shadow-lg">
              <i class="fa-solid fa-file-arrow-down mr-2"></i> Download PDF
            </a>
            {% endif %}
          </div>

          <div class="mb-8">
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <i class="fa-solid fa-align-left"></i> Extracted Sentences (Executive)
            </h2>
            <div class="p-5 rounded-2xl bg-slate-50 border border-slate-100 text-sm leading-relaxed text-slate-700">
                {% if summary %}
                  {% for s in summary %}
                    <p class="mb-3">• {{ s }}</p>
                  {% endfor %}
                {% else %}
                  <p>No summary available.</p>
                {% endif %}
            </div>
          </div>

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        
        <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-6">
          <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Original Document</h2>
          <div class="rounded-xl overflow-hidden border border-slate-200 bg-slate-100 h-[300px] relative group">
             {% if orig_type == 'pdf' %}
               <iframe src="{{ orig_url }}" class="w-full h-full" title="Original PDF"></iframe>
             {% elif orig_type == 'text' %}
               <div class="p-4 overflow-y-auto h-full text-xs font-mono">{{ orig_text }}</div>
             {% elif orig_type == 'image' %}
               <img src="{{ orig_url }}" class="w-full h-full object-contain bg-slate-800">
             {% endif %}
          </div>
        </div>

        <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-6 flex flex-col h-[400px]">
          <div class="mb-4">
            <h2 class="text-sm font-bold text-slate-800 flex items-center gap-2">
               <i class="fa-solid fa-info-circle text-amber-600"></i> Info
            </h2>
            <p class="text-xs text-slate-400">This summary is produced by a supervised classifier + redundancy removal. Download the PDF for shareable output.</p>
          </div>
          <div class="flex-1 overflow-y-auto text-sm text-slate-600">
            <pre class="whitespace-pre-wrap text-xs">{{ debug_info }}</pre>
          </div>
        </div>

      </section>
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
    raw_sentences = nltk.sent_tokenize(text)
    cleaned_sentences = [clean_text_line(s) for s in raw_sentences]

    # 2. Filter out very short / junk
    valid_sentences = []
    original_indices = []
    for i, s in enumerate(cleaned_sentences):
        s_stripped = s.strip()
        # Keep sentences longer than threshold and not just numbers
        if len(s_stripped) > 40 and not re.match(r'^[\d\W]+$', s_stripped):
            valid_sentences.append(s_stripped)
            original_indices.append(i)

    if not valid_sentences:
        return ["No valid text found in the document."]

    # 3. Vectorize & predict probabilities
    try:
        features = vectorizer.transform(valid_sentences)  # sparse matrix
    except Exception as e:
        return [f"Vectorizer Error: {str(e)}"]

    try:
        scores = model.predict_proba(features)[:, 1]
    except Exception as e:
        # Some classifiers might not implement predict_proba
        try:
            scores = model.decision_function(features)
            # scale to 0-1
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-12)
        except Exception:
            return [f"Model Error: cannot score sentences ({e})"]

    # 4. Rank & pick with redundancy check
    ranked_indices = np.argsort(scores)[::-1]  # indices into valid_sentences
    selected_indices = []
    selected_vectors = []

    for idx in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break

        current_vec = features[idx]
        is_redundant = False
        if selected_vectors:
            # compute similarity to already selected (dense array)
            try:
                sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
                if np.max(sims) > 0.65:
                    is_redundant = True
            except Exception:
                # fallback: skip redundancy check if similarity fails
                is_redundant = False

        if not is_redundant:
            selected_indices.append(idx)
            # store dense array for future comparisons
            try:
                selected_vectors.append(current_vec.toarray()[0])
            except Exception:
                # If conversion fails, store a dense copy of transform
                selected_vectors.append(current_vec.A[0] if hasattr(current_vec, "A") else np.array(current_vec.todense())[0])

    # 5. Sort selected indices (so summary flows in original order)
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

    # If image, try simple OCR via pytesseract if available (optional)
    if lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff')):
        # For simplicity we will not run OCR in this app unless pytesseract is installed and configured.
        # Try a best-effort text extraction using pytesseract if available.
        try:
            from PIL import Image
            import pytesseract

            orig_type = "image"
            img = Image.open(stored_path)
            orig_text = pytesseract.image_to_string(img)
        except Exception:
            orig_type = "image"
            orig_text = ""
    else:
        # treat as pdf or text
        with open(stored_path, "rb") as f_in:
            raw = f_in.read()
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            orig_text = extract_text_from_pdf_bytes(raw)
        else:
            orig_type = "text"
            try:
                orig_text = raw.decode("utf-8", errors="ignore")
            except Exception:
                orig_text = ""

    if len(orig_text.strip()) < 40:
        # handle low text gracefully
        summary_sentences = ["No or insufficient text extracted from the document to summarize."]
        debug_info = "Extracted text length under threshold."
    else:
        # Determine number of sentences based on length option
        length_choice = request.form.get("length", "medium")
        if length_choice == "short":
            num_sentences = 4
        elif length_choice == "long":
            num_sentences = 12
        else:
            num_sentences = 7

        summary_sentences = generate_extractive_summary(orig_text, num_sentences=num_sentences)
        debug_info = f"Original sentences count (approx): {len(nltk.sent_tokenize(orig_text))}\nSelected (requested): {num_sentences}\nModel loaded: {'yes' if model and vectorizer else 'no'}"

    # Save PDF summary
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    try:
        save_summary_pdf("Policy Brief - Executive Extract", summary_sentences, summary_path)
        summary_pdf_url = url_for("summary_file", filename=summary_filename)
    except Exception as e:
        summary_pdf_url = None
        debug_info += f"\nPDF generation failed: {e}"

    # For display, cap original text length
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


if __name__ == "__main__":
    # NOTE: In production use gunicorn; debug True is for local testing only.
    app.run(host="0.0.0.0", port=5000, debug=True)
