import io
import os
import re
import uuid
import json
import pickle
import numpy as np
from collections import defaultdict

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

import nltk
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import google.generativeai as genai

# ---------------------- CONFIG ---------------------- #

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Download NLTK data for sentence splitting
nltk.download('punkt')
nltk.download('punkt_tab')

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_API_KEY = None

# ---------------------- LOAD SUPERVISED MODEL ---------------------- #

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

supervised_model = None
supervised_vectorizer = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
        with open(MODEL_PATH, 'rb') as f:
            supervised_model = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f:
            supervised_vectorizer = pickle.load(f)
        print("✅ Supervised Model Loaded Successfully!")
    else:
        print("⚠️  Warning: model.pkl or tfidf.pkl not found. App will fail on text summarization.")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# ---------------------- HTML TEMPLATES (YELLOW THEME) ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>PolicyBrief AI | Supervised</title>
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
            amber: { 50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d', 400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', 700: '#b45309', 800: '#92400e', 900: '#78350f' },
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
    body { background-color: #fcfcfc; }
    .glass-panel {
      background: rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.6);
    }
    .gradient-text {
      background: linear-gradient(135deg, #d97706 0%, #ea580c 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    /* Progress Bar Animation */
    @keyframes progress-stripes {
      from { background-position: 1rem 0; }
      to { background-position: 0 0; }
    }
    .animate-stripes {
      background-image: linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent);
      background-size: 1rem 1rem;
      animation: progress-stripes 1s linear infinite;
    }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden min-h-screen flex flex-col">

  <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-amber-200/20 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>
  <div class="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-orange-100/30 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>

  <nav class="fixed w-full z-40 glass-panel border-b border-amber-200/30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-amber-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg shadow-amber-500/20 text-white">
            <i class="fa-solid fa-file-shield text-xl"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">
            Policy<span class="text-amber-600">Brief.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-6 text-xs font-bold uppercase tracking-wider text-slate-500">
          <span>Supervised Model</span>
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
          Supervised Extractive Summarization
        </div>
        <h1 class="text-5xl md:text-6xl font-extrabold text-slate-900 leading-tight">
          Analyze & Summarize <br>
          <span class="gradient-text">Healthcare Policies</span>
        </h1>
        <p class="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Upload PDF, Text, or <span class="font-semibold text-slate-800">Use Your Camera</span>. 
          Powered by a trained <strong>Logistic Regression</strong> model to identify high-value policy commitments.
        </p>
      </div>

      <div id="workspace" class="glass-panel rounded-3xl p-1 shadow-2xl shadow-amber-500/10 max-w-3xl mx-auto">
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
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Sentence Count</label>
                <div class="flex bg-slate-100 rounded-lg p-1">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="short" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">5 Sents</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="medium" checked class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">10 Sents</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="long" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">20 Sents</span>
                  </label>
                </div>
              </div>

              <div class="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Model</label>
                <div class="flex bg-slate-100 rounded-lg p-1">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="academic" checked class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-amber-700 peer-checked:shadow-sm transition">Supervised</span>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 text-white font-bold text-lg shadow-lg shadow-amber-500/30 hover:shadow-xl hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-2">
              <i class="fa-solid fa-wand-magic-sparkles"></i> Run Supervised Model
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
        <p class="text-sm text-slate-500">Please wait while the Supervised Model analyzes your document.</p>
      </div>

      <div class="w-full h-3 bg-slate-200 rounded-full overflow-hidden relative">
        <div id="progress-bar" class="h-full bg-gradient-to-r from-amber-400 to-orange-600 animate-stripes w-0 transition-all duration-300 ease-out"></div>
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

    // 1. File Upload Preview Logic
    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files[0]) {
        const file = this.files[0];
        const reader = new FileReader();

        // Show preview container, hide prompt
        uploadPrompt.classList.add('opacity-0');
        setTimeout(() => {
            uploadPrompt.classList.add('hidden');
            filePreview.classList.remove('hidden');
        }, 300);
        
        filenameDisplay.textContent = file.name;

        // Reset styling
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

    // Change file button
    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); 
        fileInput.value = ''; 
        filePreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        uploadPrompt.classList.remove('opacity-0');
    });

    // 2. Real Progress Bar Logic
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
        
        const totalDuration = isImage ? 12000 : 3000; 
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
                    progressStage.textContent = "Uploading & Cleaning...";
                } else if (width < 70) {
                    progressStage.textContent = "TF-IDF Vectorization & Prediction...";
                } else {
                    progressStage.textContent = "Structuring Output...";
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
            amber: { 50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d', 400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', 700: '#b45309', 800: '#92400e', 900: '#78350f' },
          },
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
          <div class="w-8 h-8 bg-gradient-to-tr from-amber-500 to-orange-500 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-file-shield text-sm"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            Policy<span class="text-amber-600">Brief.AI</span>
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
                   {{ orig_type }} processed
                 </span>
                 {% if used_model == 'gemini' %}
                 <span class="px-2 py-1 rounded-md bg-violet-50 text-violet-700 text-[0.65rem] font-bold uppercase tracking-wide border border-violet-100">
                    <i class="fa-solid fa-sparkles mr-1"></i> Gemini AI
                 </span>
                 {% else %}
                 <span class="px-2 py-1 rounded-md bg-green-50 text-green-700 text-[0.65rem] font-bold uppercase tracking-wide border border-green-100">
                    <i class="fa-solid fa-brain mr-1"></i> Supervised (LogReg)
                 </span>
                 {% endif %}
              </div>
              <h1 class="text-2xl font-extrabold text-slate-900 leading-tight">Executive Summary</h1>
            </div>
            
            {% if summary_pdf_url %}
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-amber-600 transition shadow-lg">
              <i class="fa-solid fa-file-arrow-down mr-2"></i> Download PDF
            </a>
            {% endif %}
          </div>

          <div class="mb-8">
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <i class="fa-solid fa-align-left"></i> Abstract / High Level
            </h2>
            <div class="p-5 rounded-2xl bg-amber-50 border border-amber-100 text-sm leading-relaxed text-slate-700">
               {{ abstract }}
            </div>
          </div>

          {% if sections %}
          <div class="space-y-6">
            {% for sec in sections %}
            <div>
               <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
                 <span class="w-1.5 h-6 rounded-full bg-amber-500 block"></span>
                 {{ sec.title }}
               </h3>
               <ul class="space-y-2">
                 {% for bullet in sec.bullets %}
                 <li class="flex items-start gap-3 text-sm text-slate-600">
                    <i class="fa-solid fa-check mt-1 text-amber-500 text-xs"></i>
                    <span>{{ bullet }}</span>
                 </li>
                 {% endfor %}
               </ul>
            </div>
            {% endfor %}
          </div>
          {% endif %}

          {% if implementation_points %}
          <div class="mt-8 pt-6 border-t border-slate-100">
            <h3 class="text-sm font-bold text-slate-800 uppercase tracking-wide mb-4 flex items-center gap-2">
               <i class="fa-solid fa-road text-amber-600"></i> Way Forward / Implementation
            </h3>
            <div class="grid gap-3">
               {% for p in implementation_points %}
               <div class="flex items-start gap-3 p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm text-slate-700">
                  <i class="fa-solid fa-arrow-right text-amber-500 mt-1 text-xs"></i>
                  {{ p }}
               </div>
               {% endfor %}
            </div>
          </div>
          {% endif %}

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
               <i class="fa-solid fa-robot text-amber-600"></i> Ask Gemini
            </h2>
            <p class="text-xs text-slate-400">Ask questions based on the document content.</p>
          </div>
          
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-3 mb-4 pr-2 custom-scrollbar">
             <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center text-amber-600 text-xs shrink-0"><i class="fa-solid fa-robot"></i></div>
                <div class="bg-slate-100 rounded-2xl rounded-tl-none p-3 text-xs text-slate-700 leading-relaxed">
                   Hello! I've analyzed this document. Ask me about specific goals, financing, or strategies.
                </div>
             </div>
          </div>

          <div class="relative">
             <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3 rounded-full bg-slate-50 border border-slate-200 text-sm focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition" placeholder="Type a question...">
             <button id="chat-send" class="absolute right-1 top-1 p-2 bg-amber-500 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-amber-600 transition">
                <i class="fa-solid fa-paper-plane text-xs"></i>
             </button>
          </div>
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>

      </section>
    </div>
  </main>

  <script>
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const docText = document.getElementById('doc-context').value;

    function addMsg(role, text) {
        const div = document.createElement('div');
        div.className = role === 'user' ? 'flex gap-3 flex-row-reverse' : 'flex gap-3';
        
        const avatar = document.createElement('div');
        avatar.className = `w-8 h-8 rounded-full flex items-center justify-center text-xs shrink-0 ${role === 'user' ? 'bg-slate-800 text-white' : 'bg-amber-100 text-amber-600'}`;
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = `max-w-[80%] rounded-2xl p-3 text-xs leading-relaxed ${role === 'user' ? 'bg-slate-800 text-white rounded-tr-none' : 'bg-slate-100 text-slate-700 rounded-tl-none'}`;
        bubble.textContent = text;

        div.appendChild(avatar);
        div.appendChild(bubble);
        panel.appendChild(div);
        panel.scrollTop = panel.scrollHeight;
    }

    async function sendMessage() {
        const txt = input.value.trim();
        if(!txt) return;
        addMsg('user', txt);
        input.value = '';
        
        try {
            const res = await fetch('{{ url_for("chat") }}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: txt, doc_text: docText })
            });
            const data = await res.json();
            addMsg('assistant', data.reply);
        } catch(e) {
            addMsg('assistant', "Sorry, I encountered an error.");
        }
    }

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }
  </script>

</body>
</html>
"""

# ---------------------- TEXT PROCESSING UTILS (ROBUST) ---------------------- #

def clean_text(text: str) -> str:
    """Removes headers, page numbers, and artifacts."""
    # 1. Remove "Page X"
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    # 2. Remove Section Headers (e.g., "3.3.3 Re-Orienting:")
    text = re.sub(r'\d+(\.\d+)*\s+[A-Za-z\s\-]+:', '', text)
    # 3. Remove leading section numbers
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text)
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', " ", text)
    return text.strip()

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        text = ""
        for pg in reader.pages:
            t = pg.extract_text()
            if t: text += t + " "
        return text
    except Exception as e:
        print(f"PDF Extract Error: {e}")
        return ""

def score_sentence_categories(sentence: str) -> str:
    """Scores a sentence against categories based on keywords."""
    s_lower = sentence.lower()
    
    # Categorization Dictionary
    KEYWORDS = {
        "key goals": ["aim", "goal", "objective", "target", "achieve", "reduce", "%", "2025", "2030"],
        "policy principles": ["principle", "equity", "universal", "right", "access", "accountability"],
        "service delivery": ["hospital", "primary care", "secondary care", "tertiary", "referral", "ambulance", "emergency"],
        "prevention & promotion": ["prevent", "sanitation", "nutrition", "immunization", "tobacco", "lifestyle"],
        "human resources": ["doctor", "nurse", "staff", "training", "workforce", "recruit", "paramedic"],
        "financing & private sector": ["fund", "budget", "finance", "expenditure", "insurance", "private", "ppp", "out-of-pocket"],
        "digital health": ["digital", "technology", "data", "record", "telemedicine", "online"],
        "ayush integration": ["ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy"],
        "implementation": ["implement", "strategy", "roadmap", "monitor", "evaluate", "framework"]
    }

    scores = {cat: 0 for cat in KEYWORDS}
    
    for cat, keywords in KEYWORDS.items():
        for kw in keywords:
            if kw in s_lower:
                scores[cat] += 1
                
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return "other"
    return best_cat

def build_structured_summary(summary_sentences: list):
    """
    Takes the RAW list of important sentences from the Supervised Model
    and organizes them into UI sections.
    """
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
        "implementation": "Implementation Strategy", 
        "other": "Other Key Observations"
    }
    
    sections = []
    
    # Create sections if content exists
    for k, title in section_titles.items():
        if cat_map[k]:
            unique = list(dict.fromkeys(cat_map[k])) # Dedup
            sections.append({"title": title, "bullets": unique})
            
    # Create Abstract (First 3 significant sentences)
    abstract_cands = cat_map.get('key goals', []) + cat_map.get('policy principles', []) + summary_sentences
    abstract = " ".join(list(dict.fromkeys(abstract_cands))[:3])
    
    # Implementation specific points
    impl_points = cat_map.get("implementation", [])
    
    return {
        "abstract": abstract,
        "sections": sections,
        "implementation_points": impl_points
    }

# ---------------------- SUPERVISED SUMMARIZATION CORE ---------------------- #

def generate_supervised_summary(text: str, num_sentences: int = 10):
    """
    Uses the trained Logistic Regression model to identify important sentences.
    """
    if not supervised_model or not supervised_vectorizer:
        return ["Error: Model not loaded. Check server logs."]

    # 1. Split and Clean
    raw_sentences = nltk.sent_tokenize(text)
    clean_sentences = [clean_text(s) for s in raw_sentences]
    
    # Filter valid sentences (length check)
    valid_sentences = []
    for s in clean_sentences:
        if len(s) > 30: # Ignore very short lines
            valid_sentences.append(s)
            
    if not valid_sentences:
        return ["No valid text found to summarize."]

    try:
        # 2. Vectorize
        features = supervised_vectorizer.transform(valid_sentences)
        
        # 3. Predict Importance (Probability of Class 1)
        scores = supervised_model.predict_proba(features)[:, 1]
        
        # 4. Rank
        ranked_indices = np.argsort(scores)[::-1] # Descending order
        
        # 5. Select Top N with Redundancy Check
        selected_indices = []
        selected_vectors = []
        
        for idx in ranked_indices:
            if len(selected_indices) >= num_sentences:
                break
                
            current_vec = features[idx]
            
            # Simple cosine check against selected
            is_redundant = False
            if selected_vectors:
                sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
                if np.max(sims) > 0.70: # 70% similarity threshold
                    is_redundant = True
            
            if not is_redundant:
                selected_indices.append(idx)
                selected_vectors.append(current_vec.toarray()[0])

        # 6. Sort back by original order for narrative flow
        final_indices = sorted(selected_indices)
        final_summary = [valid_sentences[i] for i in final_indices]
        
        return final_summary

    except Exception as e:
        print(f"Prediction Error: {e}")
        return ["Error during model prediction."]

# ---------------------- GEMINI IMAGE PROCESSING ---------------------- #

def process_image_with_gemini(image_path: str):
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        img = Image.open(image_path)
        
        prompt = """
        Analyze this image of a policy document. 
        Perform two tasks:
        1. Extract the main text content.
        2. Create a structured summary.
        
        Output strictly valid JSON:
        {
            "extracted_text": "...",
            "summary_structure": {
                "abstract": "...",
                "sections": [
                    { "title": "Key Goals", "bullets": ["..."] },
                    { "title": "Financing", "bullets": ["..."] }
                ],
                "implementation_points": ["..."]
            }
        }
        """
        response = model.generate_content([prompt, img])
        text_resp = response.text.strip()
        
        if text_resp.startswith("```json"):
            text_resp = text_resp.replace("```json", "").replace("```", "")
        
        data = json.loads(text_resp)
        return data, None
        
    except Exception as e:
        return None, str(e)

# ---------------------- PDF GENERATION ---------------------- #

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Executive Summary")
    y -= 15
    
    c.setFont("Helvetica", 10)
    if abstract:
        lines = simpleSplit(abstract, "Helvetica", 10, width - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
    y -= 10
    
    for sec in sections:
        if y < 100:
            c.showPage(); y = height - margin
        
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

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    doc_text = data.get("doc_text", "")
    
    if not GEMINI_API_KEY:
        return jsonify({"reply": "Gemini Key not configured."})
        
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        prompt = f"Context from document: {doc_text[:30000]}\n\nUser Question: {message}\nAnswer concisely."
        resp = chat.send_message(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

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
    
    structured_data = {}
    orig_text = ""
    orig_type = "unknown"
    used_model = "supervised" 
    
    # CASE 1: IMAGE -> GEMINI
    if lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        orig_type = "image"
        used_model = "gemini"
        gemini_data, err = process_image_with_gemini(stored_path)
        if err or not gemini_data:
            abort(500, f"Gemini Image Processing Failed: {err}")
        orig_text = gemini_data.get("extracted_text", "")
        structured_data = gemini_data.get("summary_structure", {})
        
        # Defaults
        if "abstract" not in structured_data: structured_data["abstract"] = "Summary not generated."
        if "sections" not in structured_data: structured_data["sections"] = []
        if "implementation_points" not in structured_data: structured_data["implementation_points"] = []

    # CASE 2: PDF/TXT -> SUPERVISED MODEL
    else:
        used_model = "supervised"
        with open(stored_path, "rb") as f_in:
            raw_bytes = f_in.read()
            
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            orig_text = extract_text_from_pdf_bytes(raw_bytes)
        else:
            orig_type = "text"
            orig_text = raw_bytes.decode("utf-8", errors="ignore")
            
        if len(orig_text) < 50:
            abort(400, "Not enough text found in document.")
            
        length_choice = request.form.get("length", "medium")
        
        # Determine number of sentences based on UI choice
        num_sents = 10 # medium
        if length_choice == "short": num_sents = 5
        if length_choice == "long": num_sents = 20
        
        # --- RUN SUPERVISED MODEL ---
        raw_summary_list = generate_supervised_summary(orig_text, num_sentences=num_sents)
        
        # --- STRUCTURE THE OUTPUT ---
        structured_data = build_structured_summary(raw_summary_list)

    # Generate PDF
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    save_summary_pdf(
        "Policy Summary",
        structured_data.get("abstract", ""),
        structured_data.get("sections", []),
        summary_path
    )
    
    return render_template_string(
        RESULT_HTML,
        title="PolicyBrief Summary",
        orig_type=orig_type,
        orig_url=url_for("uploaded_file", filename=stored_name),
        orig_text=orig_text[:20000], 
        doc_context=orig_text[:20000],
        abstract=structured_data.get("abstract", ""),
        sections=structured_data.get("sections", []),
        implementation_points=structured_data.get("implementation_points", []),
        summary_pdf_url=url_for("summary_file", filename=summary_filename),
        used_model=used_model
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
