import io
import os
import re
import uuid
import json
import time
import pickle
import nltk
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

import numpy as np
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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import google.generativeai as genai

# ==============================================================================
# CONFIG & SETUP
# ==============================================================================

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Configure Gemini (Optional fallback)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_API_KEY = None

# ==============================================================================
# LOAD SUPERVISED MODEL
# ==============================================================================
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf.pkl')

supervised_model = None
supervised_vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            supervised_model = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f:
            supervised_vectorizer = pickle.load(f)
        print("✅ Supervised Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Warning: 'model.pkl' or 'tfidf.pkl' not found. App will run but summarization may fail.")


# ==============================================================================
# HTML TEMPLATES (UNTOUCHED)
# ==============================================================================

COMMON_HEAD = """
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <script>
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    colors: {
                        "background-light": "#FFFFFF",
                        "background-dark": "#0D0D0F",
                        "surface-dark": "#161b22",
                        "afzal-purple": "#8C4FFF",
                        "afzal-blue": "#4D9CFF",
                        "afzal-red": "#FF5757",
                        "text-light": "#1F2937",
                        "text-dark": "#F3F4F6",
                    },
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    backgroundImage: {
                        'grid-pattern-dark': "linear-gradient(to right, rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.05) 1px, transparent 1px)",
                        'radial-glow': "radial-gradient(circle at center, rgba(140, 79, 255, 0.15) 0%, transparent 70%)",
                    },
                    animation: {
                        'pulse-slow': 'pulse-opacity 4s ease-in-out infinite',
                        'float': 'float 6s ease-in-out infinite',
                        'spin-slow': 'spin 3s linear infinite',
                    },
                    keyframes: {
                        'pulse-opacity': {
                            '0%, 100%': { opacity: 0.2, transform: 'scale(1)' },
                            '50%': { opacity: 0.5, transform: 'scale(1.1)' },
                        },
                        'float': {
                            '0%, 100%': { transform: 'translateY(0)' },
                            '50%': { transform: 'translateY(-10px)' },
                        }
                    }
                },
            },
        };
    </script>

    <style>
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0D0D0F; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }

        body { background-color: #0D0D0F; color: #F3F4F6; }

        .perspective-1000 { perspective: 1000px; }
        .transform-style-3d { transform-style: preserve-3d; }
        
        /* Glass Card Style */
        .glossary-card {
            border-radius: 0.75rem;
            padding: 2rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.1s ease-out, box-shadow 0.3s ease;
            transform-style: preserve-3d;
            background-color: #161b22;
            border: 1px solid #374151;
        }

        .glossary-card:hover {
            box-shadow: 0 0 30px -5px rgba(140, 79, 255, 0.15);
            z-index: 10;
            border-color: rgba(140, 79, 255, 0.5);
        }

        /* Initial state for fade-up animation */
        .fade-up {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.6s ease-out, transform 0.6s ease-out;
        }
        
        .upload-zone {
            background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='12' ry='12' stroke='%23374151FF' stroke-width='2' stroke-dasharray='12%2c 12' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
            transition: all 0.3s ease;
        }
        .upload-zone:hover {
            background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='12' ry='12' stroke='%238C4FFF' stroke-width='2' stroke-dasharray='12%2c 12' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
            background-color: rgba(22, 27, 34, 0.8);
        }

        /* ==============================
            FORCE HIDE GOOGLE UI (Robust)
            ============================== */
        .goog-logo-link,
        .goog-te-gadget,
        .goog-te-banner-frame,
        .goog-te-balloon-frame,
        .goog-te-combo {
            display: none !important;
        }
        body > .skiptranslate {
            display: none !important;
        }
        body {
            top: 0 !important;
        }
        
        /* Custom Select Styles */
        .custom-select-wrapper { position: relative; user-select: none; }
    </style>
"""

COMMON_SCRIPTS = """
<script>
document.addEventListener('DOMContentLoaded', () => {
    // 3D Tilt Effect - Only applied where [data-tilt] exists
    const glossaryCards = document.querySelectorAll('[data-tilt]');
    glossaryCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top; 
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -5;
            const rotateY = ((x - centerX) / centerX) * 5;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
        });
    });

    // Fade Up Animation Observer
    const observerOptions = { threshold: 0.1, rootMargin: "0px 0px -20px 0px" };
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const fadeElements = document.querySelectorAll('.fade-up');
    fadeElements.forEach(el => { observer.observe(el); });
    
    // Fallback
    setTimeout(() => {
        fadeElements.forEach(el => {
            if(getComputedStyle(el).opacity === '0') {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    }, 2000);
});
</script>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Workspace | Med.AI</title>
    {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark overflow-x-hidden selection:bg-afzal-purple selection:text-white" translate="no">

<nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full">
        <div class="grid grid-cols-[auto_1fr_auto] lg:grid-cols-3 items-center h-full">
            <div class="flex items-center h-full pl-6 lg:pl-8">
                <a href="https://afzalkhanofficial.github.io/Med-AI/index.html" class="flex items-center space-x-2.5 mr-8 shrink-0">
                    <div class="w-8 h-8 bg-white rounded-full relative overflow-hidden flex items-center justify-center">
                        <i class="fa-solid fa-staff-snake text-black text-lg"></i>
                    </div>
                    <span class="font-bold text-2xl tracking-tight text-white font-sans">Med.AI</span>
                </a>
            </div>
            <div class="hidden lg:flex justify-center">
                 <span class="text-afzal-purple font-mono text-xs uppercase tracking-widest border border-afzal-purple/30 bg-afzal-purple/10 px-3 py-1 rounded">Workspace : Supervised Text Summarization</span>
            </div>
            <div class="flex items-center justify-end h-full pr-6 lg:pr-8">
                 <a href="https://afzalkhanofficial.github.io/Med-AI/about.html" class="text-sm font-medium text-gray-400 hover:text-white transition-colors">About</a>
            </div>
        </div>
    </div>
</nav>

<header class="relative pt-20 pb-12 overflow-hidden bg-background-dark">
    <div class="absolute inset-0 bg-grid-pattern-dark bg-[size:50px_50px] opacity-40"></div>
    <div class="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-radial-glow blur-3xl pointer-events-none"></div>

    <div class="max-w-4xl mx-auto px-4 text-center relative z-10 fade-up">
        <h1 class="text-5xl md:text-7xl font-semibold text-white mb-6 leading-tight tracking-tight">
            SUPERVISED <br>
            <span class="text-transparent bg-clip-text bg-gradient-to-r from-afzal-purple via-white to-afzal-blue">TF-IDF and Logistic Regression</span>
        </h1>
        <p class="text-xl text-gray-400 leading-relaxed max-w-2xl mx-auto font-light">
            Upload healthcare policy briefs. Our trained ML engine identifies key sentences, extracts entities, and generates structured summaries.
        </p>
    </div>
</header>

<section class="py-12 relative z-10 bg-background-dark">
    <div class="max-w-4xl mx-auto px-4 fade-up delay-100">
        
        <div class="glossary-card bg-surface-dark border border-gray-800 p-8 rounded-2xl shadow-2xl" data-tilt>
            <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
                
                <div class="upload-zone relative w-full h-64 rounded-xl flex flex-col items-center justify-center cursor-pointer group" id="drop-zone">
                    <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" multiple class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
                    
                    <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105 transform-style-3d">
                        <div class="w-16 h-16 bg-surface-dark border border-gray-700 rounded-full flex items-center justify-center mx-auto text-afzal-purple text-2xl group-hover:border-afzal-purple group-hover:bg-afzal-purple/10 transition-colors shadow-lg">
                            <i class="fa-solid fa-cloud-arrow-up transform translate-z-[10px]"></i>
                        </div>
                        <div>
                            <p class="text-lg font-bold text-white">Click to upload or Drag & Drop</p>
                            <p class="text-sm text-gray-500 font-mono mt-1">Supported: PDF, TXT, Images</p>
                        </div>
                    </div>

                    <div id="file-preview" class="hidden absolute inset-0 bg-surface-dark z-10 flex flex-col items-center justify-center p-6 text-center rounded-xl">
                        <div id="preview-icon" class="mb-4 text-4xl text-afzal-purple drop-shadow-[0_0_10px_rgba(140,79,255,0.5)]"></div>
                        <div id="preview-image-container" class="mb-4 hidden rounded-lg overflow-hidden border border-gray-700 max-h-32 shadow-lg">
                            <img id="preview-image" src="" alt="Preview" class="h-full object-contain">
                        </div>
                        <p id="filename-display" class="font-mono text-white text-sm break-all max-w-md bg-black/30 px-4 py-2 rounded border border-gray-800"></p>
                        <button type="button" id="change-file-btn" class="mt-4 text-xs text-afzal-blue hover:text-white font-bold tracking-wide uppercase transition-colors z-30 relative">Change file</button>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-black/20 rounded-lg p-4 border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Summary Length</label>
                        <div class="flex gap-2">
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="length" value="short" class="peer hidden">
                                <span class="block text-center py-2 text-xs font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-purple peer-checked:text-white peer-checked:border-afzal-purple transition-all">SHORT</span>
                            </label>
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="length" value="medium" checked class="peer hidden">
                                <span class="block text-center py-2 text-xs font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-purple peer-checked:text-white peer-checked:border-afzal-purple transition-all">MEDIUM</span>
                            </label>
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="length" value="long" class="peer hidden">
                                <span class="block text-center py-2 text-xs font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-purple peer-checked:text-white peer-checked:border-afzal-purple transition-all">LONG</span>
                            </label>
                        </div>
                    </div>

                    <div class="bg-black/20 rounded-lg p-4 border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Model Tone</label>
                        <div class="flex gap-2">
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="tone" value="academic" checked class="peer hidden">
                                <span class="block text-center py-2 text-xs font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-blue peer-checked:text-white peer-checked:border-afzal-blue transition-all">TECHNICAL</span>
                            </label>
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="tone" value="easy" class="peer hidden">
                                <span class="block text-center py-2 text-xs font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-blue peer-checked:text-white peer-checked:border-afzal-blue transition-all">SIMPLE</span>
                            </label>
                        </div>
                    </div>
                </div>

                <button type="submit" class="w-full relative z-[1] flex items-center justify-center bg-white text-black h-14 font-semibold text-sm uppercase tracking-widest hover:bg-gray-200 transition-colors rounded">
                    Generate Analysis
                </button>
            </form>
        </div>

    </div>
</section>

<div id="progress-overlay" class="fixed inset-0 bg-background-dark/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-md px-6 text-center space-y-8 fade-up">
        
        <div class="relative w-24 h-24 mx-auto">
            <div class="absolute inset-0 rounded-full border-4 border-gray-800"></div>
            <div class="absolute inset-0 rounded-full border-4 border-afzal-purple border-t-transparent animate-spin"></div>
            <div class="absolute inset-0 flex items-center justify-center text-white font-mono text-xl font-bold" id="progress-text">0%</div>
        </div>

        <div class="space-y-3">
            <h3 class="text-2xl font-light text-white" id="progress-stage">Initializing...</h3>
            <p class="text-sm text-gray-500 font-mono uppercase tracking-widest">Processing secure document</p>
        </div>

        <div class="w-full h-1 bg-gray-800 rounded-full overflow-hidden relative">
            <div id="progress-bar" class="h-full bg-gradient-to-r from-afzal-purple to-afzal-blue w-0 transition-all duration-300 ease-out shadow-[0_0_15px_rgba(140,79,255,0.5)]"></div>
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
      if (this.files && this.files.length > 0) {
        const file = this.files[0];
        const count = this.files.length;
        const reader = new FileReader();

        uploadPrompt.classList.add('hidden');
        filePreview.classList.remove('hidden');
        
        if (count > 1) {
            filenameDisplay.textContent = `${count} Files Selected`;
            previewImgContainer.classList.add('hidden');
            previewIcon.innerHTML = '<i class="fa-solid fa-layer-group"></i>';
        } else {
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
               previewIcon.innerHTML = '<i class="fa-regular fa-file-pdf"></i>';
            } else {
               previewIcon.innerHTML = '<i class="fa-regular fa-file-lines"></i>';
            }
        }
      }
    });

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        filePreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
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
                progressStage.textContent = "Finalizing Output...";
            } else {
                width += step;
                if(Math.random() > 0.5) width += 0.5;
                
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';

                if (width < 30) {
                    progressStage.textContent = "Uploading & Encrypting...";
                } else if (width < 60) {
                    progressStage.textContent = "Extracting Entities...";
                } else if (width < 85) {
                    progressStage.textContent = "Generating Summary...";
                } else {
                    progressStage.textContent = "Structuring Data...";
                }
            }
        }, intervalTime);
    });
</script>
{COMMON_SCRIPTS}
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>{{ title }} | Med.AI</title>
    {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark overflow-x-hidden selection:bg-afzal-purple selection:text-white" translate="no">

<nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full max-w-7xl mx-auto px-6">
        <div class="flex items-center justify-between h-full">
            <div class="flex items-center gap-4">
                <a href="/" class="flex items-center space-x-2.5">
                    <div class="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                        <i class="fa-solid fa-staff-snake text-black text-lg"></i>
                    </div>
                    <span class="font-bold text-xl text-white">Med.AI</span>
                </a>
                <span class="text-gray-600">/</span>
                <span class="text-sm font-mono text-afzal-purple">Analysis Report</span>
            </div>
            
            <div class="flex items-center gap-4">
                <div id="google_translate_element" class="hidden absolute"></div>

                <a href="{{ url_for('index') }}" class="group relative z-[1] inline-flex items-center cursor-pointer transition-colors text-xs font-bold uppercase tracking-widest text-white hover:text-afzal-purple border border-gray-700 hover:border-afzal-purple px-4 py-2 rounded">
                    <i class="fa-solid fa-plus mr-2"></i> New
                </a>
            </div>
        </div>
    </div>
</nav>

<main class="py-12 px-4 relative">
    <div class="fixed top-20 left-10 w-64 h-64 bg-afzal-purple/10 rounded-full blur-[100px] pointer-events-none"></div>

    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8 relative z-10">
        
      <section class="lg:col-span-7 space-y-6">
        
        <div id="translate-text" class="glossary-card bg-surface-dark border border-gray-800 p-8 rounded-2xl shadow-lg" translate="yes">
           <div class="flex flex-wrap items-start justify-between gap-4 mb-6 border-b border-gray-700 pb-6">
             <div>
                <div class="flex items-center gap-2 mb-3">
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-gray-800 text-gray-400 border border-gray-600">
                        {{ orig_type }} Source
                    </span>
                    {% if used_model == 'gemini' %}
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-afzal-purple/10 text-afzal-purple border border-afzal-purple/20">
                        <i class="fa-solid fa-wand-magic-sparkles mr-1"></i> GEMINI 2.5
                    </span>
                    {% else %}
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-blue-500/10 text-blue-400 border border-blue-500/20">
                        TF-IDF + Logistic Regression
                    </span>
                    {% endif %}
                </div>
                <h1 class="text-3xl font-light text-white leading-tight">Policy Summary</h1>
             </div>
             
             <div class="flex items-center gap-4">
                 <div class="relative group custom-select-wrapper">
                    <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
                        <i class="fa-solid fa-language"></i>
                    </div>
                    <select id="languageSelect" class="appearance-none bg-[#161b22] border border-gray-700 text-gray-300 text-xs font-bold uppercase tracking-widest rounded px-4 pl-9 py-2 pr-8 focus:outline-none focus:border-afzal-purple cursor-pointer hover:text-white transition-colors w-32">
                        <option value="en" selected>English</option>
                        <option value="hi">Hindi</option>
                        <option value="te">Telugu</option>
                    </select>
                    <div class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                        <i class="fa-solid fa-chevron-down text-[10px]"></i>
                    </div>
                </div>

                 {% if summary_pdf_url %}
                 <a href="{{ summary_pdf_url }}" class="flex items-center justify-center w-10 h-10 rounded-full bg-white text-black hover:bg-afzal-purple hover:text-white transition-all shadow-[0_0_15px_rgba(255,255,255,0.2)]">
                    <i class="fa-solid fa-file-arrow-down"></i>
                 </a>
                 {% endif %}
             </div>
           </div>

           <div class="mb-8">
              <h2 class="text-xs font-bold text-afzal-blue uppercase tracking-widest mb-4 flex items-center gap-2">
                 <i class="fa-solid fa-layer-group"></i> Abstract
              </h2>
              <div class="p-6 rounded-lg bg-black/30 border border-gray-800 text-sm leading-relaxed text-gray-300 font-light">
                 {{ abstract }}
              </div>
           </div>

           {% if simple_text %}
           <div class="mb-8">
              <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">Detailed Text</h2>
              <div class="text-sm leading-7 text-gray-400 text-justify">
                 {{ simple_text }}
              </div>
           </div>
           {% endif %}

           {% if sections %}
           <div class="space-y-8">
              {% for sec in sections %}
              <div class="relative pl-6 border-l border-gray-800 hover:border-afzal-purple transition-colors duration-300 group">
                 <h3 class="text-lg font-medium text-white mb-3 group-hover:text-afzal-purple transition-colors font-mono">
                    {{ sec.title }}
                 </h3>
                 <ul class="space-y-3">
                    {% for bullet in sec.bullets %}
                    <li class="flex items-start gap-3 text-sm text-gray-400">
                       <i class="fa-solid fa-angle-right mt-1 text-gray-600 group-hover:text-afzal-purple transition-colors text-xs"></i>
                       <span>{{ bullet }}</span>
                    </li>
                    {% endfor %}
                 </ul>
              </div>
              {% endfor %}
           </div>
           {% endif %}
        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
          
        <div class="bg-surface-dark border border-gray-800 rounded-xl shadow-lg overflow-hidden flex flex-col h-[400px]">
             <div class="bg-[#0d1117] px-4 py-3 border-b border-gray-800 flex justify-between items-center">
                 <h2 class="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                     <i class="fa-regular fa-eye"></i> Source View
                 </h2>
                 <div class="flex gap-1.5">
                    <span class="w-2.5 h-2.5 rounded-full bg-red-500/50"></span>
                    <span class="w-2.5 h-2.5 rounded-full bg-yellow-500/50"></span>
                    <span class="w-2.5 h-2.5 rounded-full bg-green-500/50"></span>
                 </div>
             </div>
             
             <div class="flex-1 bg-black/40 overflow-auto p-4 custom-scrollbar">
                 {% if orig_type == 'pdf' %}
                   <iframe src="{{ orig_url }}" class="w-full h-full rounded border border-gray-800" title="Original PDF"></iframe>
                 {% elif orig_type == 'text' %}
                   <div class="p-4 text-xs font-mono text-gray-400 whitespace-pre-wrap">{{ orig_text }}</div>
                 {% elif orig_type == 'image' %}
                   <div class="space-y-4">
                        {% if orig_images is defined and orig_images|length > 0 %}
                            {% for img_src in orig_images %}
                            <div class="border border-gray-700 rounded-lg overflow-hidden bg-black">
                                <img src="{{ img_src }}" class="w-full h-auto object-contain">
                            </div>
                            {% endfor %}
                        {% else %}
                            <img src="{{ orig_url }}" class="w-full h-auto object-contain border border-gray-700 rounded-lg">
                        {% endif %}
                   </div>
                 {% endif %}
             </div>
        </div>

        <div class="bg-surface-dark border border-gray-800 rounded-xl shadow-lg overflow-hidden flex flex-col h-[500px]">
            <div class="p-4 bg-gradient-to-r from-[#1c2128] to-[#161b22] border-b border-gray-800 flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="relative">
                        <div class="w-10 h-10 rounded-full bg-afzal-purple/20 flex items-center justify-center text-afzal-purple border border-afzal-purple/30">
                            <i class="fa-solid fa-robot"></i>
                        </div>
                        <div class="absolute bottom-0 right-0 w-2.5 h-2.5 bg-green-500 rounded-full border-2 border-[#1c2128]"></div>
                    </div>
                    <div>
                        <h2 class="text-sm font-bold text-white">Med.AI Assistant</h2>
                        <p class="text-[10px] text-gray-400 font-mono">ONLINE • CONTEXT AWARE</p>
                    </div>
                </div>
            </div>
            
            <div id="chat-panel" class="flex-1 overflow-y-auto p-4 space-y-4 bg-[#0d1117]">
                 <div class="flex gap-3">
                    <div class="w-8 h-8 rounded-full bg-afzal-purple border border-afzal-purple flex items-center justify-center text-white text-xs shrink-0 shadow-lg">
                        <i class="fa-solid fa-robot"></i>
                    </div>
                    <div class="bg-[#1c2128] border border-gray-700 rounded-2xl rounded-tl-none p-3 text-xs text-gray-300 leading-relaxed max-w-[85%]">
                       Analysis complete. I have context on the document above. Ask me about specific figures, dates, or compliance requirements.
                    </div>
                 </div>
            </div>

            <div class="p-3 bg-[#161b22] border-t border-gray-800">
                <div class="relative flex items-center">
                    <input type="text" id="chat-input" class="w-full pl-4 pr-10 py-3 rounded-full bg-[#0d1117] border border-gray-700 text-sm text-white focus:outline-none focus:border-afzal-purple focus:ring-1 focus:ring-afzal-purple transition placeholder-gray-600 font-mono" placeholder="Ask a question...">
                    <button id="chat-send" class="absolute right-2 p-2 w-8 h-8 rounded-full bg-afzal-purple text-white flex items-center justify-center hover:bg-afzal-blue transition shadow-lg">
                        <i class="fa-solid fa-paper-plane text-xs"></i>
                    </button>
                </div>
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
        avatar.className = `w-8 h-8 rounded-full flex items-center justify-center text-xs shrink-0 border shadow-md ${role === 'user' ? 'bg-white text-black border-white' : 'bg-afzal-purple text-white border-afzal-purple'}`;
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = `max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed border ${role === 'user' ? 'bg-afzal-purple text-white border-afzal-purple rounded-tr-none' : 'bg-[#1c2128] text-gray-300 border-gray-700 rounded-tl-none'}`;
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
            addMsg('assistant', "Connection interrupted. Please try again.");
        }
    }

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }
</script>

<script>
    let translateReady = false;

    function googleTranslateElementInit() {
        new google.translate.TranslateElement({
            pageLanguage: 'en',
            includedLanguages: 'en,hi,te',
            autoDisplay: false
        }, 'google_translate_element');

        // Delay to ensure combo is created
        setTimeout(() => translateReady = true, 500);
    }

    document.getElementById("languageSelect").addEventListener("change", function () {
        if (!translateReady) return;

        const combo = document.querySelector(".goog-te-combo");
        if (!combo) return;

        combo.value = this.value;
        combo.dispatchEvent(new Event("change"));
    });
</script>

<script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>

{COMMON_SCRIPTS}
</body>
</html>
"""

# Inject Common Parts
INDEX_HTML = INDEX_HTML.replace("{COMMON_HEAD}", COMMON_HEAD).replace("{COMMON_SCRIPTS}", COMMON_SCRIPTS)
RESULT_HTML = RESULT_HTML.replace("{COMMON_HEAD}", COMMON_HEAD).replace("{COMMON_SCRIPTS}", COMMON_SCRIPTS)


# ==============================================================================
# 1. TEXT CLEANING & SUPERVISED LOGIC
# ==============================================================================

def clean_text(text):
    """
    Aggressive cleaning to remove headers, page numbers, and artifacts.
    Matches the cleaning logic used to train the Supervised Model.
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

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    """Helper to get text from PDF bytes via PyPDF2"""
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for pg in reader.pages:
            txt = pg.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        print(f"PDF Extract Error: {e}")
        return ""

def generate_supervised_summary(text, num_sentences=7):
    """
    Core Supervised Logic:
    1. Tokenize & Clean
    2. Vectorize using pre-trained TF-IDF
    3. Predict Importance using pre-trained Model
    4. Filter Redundancy using Cosine Similarity
    """
    if not supervised_model or not supervised_vectorizer:
        return ["Error: Model not loaded. Check server logs."]

    # 1. Split into sentences
    try:
        raw_sentences = nltk.sent_tokenize(text)
    except:
        return ["Error processing text."]

    # 2. Clean sentences
    clean_sentences = [clean_text(s) for s in raw_sentences]
    
    # 3. Filter out junk (empty or very short lines)
    valid_sentences = []
    original_indices = [] # Keep track of original order
    
    for i, s in enumerate(clean_sentences):
        # Must be at least 40 chars and contain letters
        if len(s) > 40 and re.search('[a-zA-Z]', s):
            valid_sentences.append(s)
            original_indices.append(i)

    if not valid_sentences:
        return ["No valid text found in document."]

    # 4. Predict Importance Scores
    try:
        features = supervised_vectorizer.transform(valid_sentences)
        # Get probability of Class 1 (Important)
        scores = supervised_model.predict_proba(features)[:, 1]
    except Exception as e:
        return [f"Prediction Error: {e}"]
    
    # 5. Smart Selection Loop (Redundancy Filter)
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
            sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
            
            # If >65% similar to any existing sentence, skip it
            if np.max(sims) > 0.65:
                is_redundant = True
        
        if not is_redundant:
            selected_indices.append(idx)
            # Convert sparse vector to dense array for storage
            selected_vectors.append(current_vec.toarray()[0])

    # 6. Sort back by original index to maintain document flow
    final_sentences_indices = sorted(selected_indices)
    
    # Retrieve the text using the valid_sentences list
    summary_list = [valid_sentences[i] for i in final_sentences_indices]
    
    return summary_list

# ==============================================================================
# 2. UI CATEGORIZATION (Visual Only)
# ==============================================================================
# We keep this strictly to format the output of the Supervised Model 
# into the pretty "Sections" the UI expects.

POLICY_KEYWORDS = {
    "key goals": ["aim", "goal", "objective", "target", "achieve", "reduce", "increase", "rate", "%", "2025", "2030"],
    "policy principles": ["principle", "equity", "universal", "right", "access", "accountability", "transparency", "inclusive"],
    "service delivery": ["hospital", "primary care", "clinic", "health center", "infrastructure", "bed", "drug", "diagnostic"],
    "prevention": ["prevent", "sanitation", "nutrition", "immunization", "tobacco", "hygiene", "awareness"],
    "financing": ["fund", "budget", "finance", "expenditure", "cost", "insurance", "spending", "gdp"],
    "digital health": ["digital", "technology", "data", "ehr", "telemedicine", "app", "online"],
}

def score_sentence_categories(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    for cat, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in s_lower: scores[cat] += 1
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "other"

def build_structured_from_supervised(summary_sentences: List[str], tone: str):
    """
    Takes the sentences selected by the Supervised Model and formats them
    for the UI (Bullets/Sections).
    """
    # 1. Simple Tone: Return as single clean paragraph
    if tone == "easy":
        text_block = " ".join(summary_sentences)
        return {
            "abstract": summary_sentences[0] if summary_sentences else "No abstract.",
            "sections": [],
            "simple_text": text_block,
        }

    # 2. Technical Tone: Categorize the selected sentences
    cat_map = defaultdict(list)
    for s in summary_sentences:
        category = score_sentence_categories(s)
        cat_map[category].append(s)
    
    section_titles = {
        "key goals": "Key Goals & Targets", 
        "policy principles": "Policy Principles",
        "service delivery": "Healthcare Delivery", 
        "prevention": "Prevention & Wellness",
        "financing": "Financing & Costs",
        "digital health": "Digital Interventions", 
        "other": "Other Key Points"
    }
    
    sections = []
    for k, title in section_titles.items():
        if cat_map[k]:
            sections.append({"title": title, "bullets": cat_map[k]})
            
    # Abstract is just the first few sentences
    abstract = " ".join(summary_sentences[:3])
    
    return {
        "abstract": abstract,
        "sections": sections,
        "simple_text": None,
    }

# ==============================================================================
# 3. GEMINI IMAGE PROCESSING
# ==============================================================================

def process_images_with_gemini(image_paths: List[str]):
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        images = []
        for p in image_paths:
            img = Image.open(p)
            img.thumbnail((256, 256))
            images.append(img)
        
        prompt = """
        Analyze these images (pages of a policy document). 
        Perform two tasks:
        1. Extract the main text content combined.
        2. Create a structured summary.
        
        Output strictly valid JSON:
        {
            "extracted_text": "...",
            "summary_structure": {
                "abstract": "...",
                "sections": [
                    { "title": "Key Goals", "bullets": ["..."] },
                    { "title": "Financing", "bullets": ["..."] }
                ]
            }
        }
        """
        content = [prompt] + images
        response = model.generate_content(content)
        text_resp = response.text.strip()
        
        if text_resp.startswith("```json"):
            text_resp = text_resp.replace("```json", "").replace("```", "")
        
        data = json.loads(text_resp)
        return data, None
        
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 4. PDF GENERATION
# ==============================================================================

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
            if y < 50:
                 c.showPage(); y = height - margin
            c.drawString(margin, y, line)
            y -= 12
    else:
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

# ==============================================================================
# 5. ROUTES
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
    # Handle Multiple Files
    files = request.files.getlist("file")
    
    if not files or files[0].filename == "":
        abort(400, "No file uploaded")
    
    is_multi_image = False
    valid_img_exts = ('.png', '.jpg', '.jpeg', '.webp')
    
    saved_paths = []
    saved_urls = []
    
    # Save all files
    uid = uuid.uuid4().hex
    for f in files:
        fname = secure_filename(f.filename)
        stored_name = f"{uid}_{fname}"
        stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        f.save(stored_path)
        saved_paths.append(stored_path)
        saved_urls.append(url_for("uploaded_file", filename=stored_name))

    first_name_lower = files[0].filename.lower()
    
    if len(files) > 1:
        for f in files:
            if not f.filename.lower().endswith(valid_img_exts):
                 abort(400, "Multiple file upload only supported for Images. Upload PDF individually.")
        is_multi_image = True
    elif first_name_lower.endswith(valid_img_exts):
        is_multi_image = True
    
    structured_data = {}
    orig_text = ""
    orig_type = "unknown"
    used_model = "supervised" 
    
    # CASE 1: IMAGE(S) -> GEMINI (Fallback to GenAI for images as ML model is text-only)
    if is_multi_image:
        orig_type = "image"
        used_model = "gemini"
        gemini_data, err = process_images_with_gemini(saved_paths)
        
        if err or not gemini_data:
            abort(500, f"Gemini Image Processing Failed: {err}")
        
        orig_text = gemini_data.get("extracted_text", "")
        structured_data = gemini_data.get("summary_structure", {})
        
        # Defaults
        if "abstract" not in structured_data: structured_data["abstract"] = "Summary not generated."
        if "sections" not in structured_data: structured_data["sections"] = []

    # CASE 2: PDF/TXT (Single File) -> SUPERVISED ML
    else:
        stored_path = saved_paths[0]
        used_model = "supervised"
        with open(stored_path, "rb") as f_in:
            raw_bytes = f_in.read()
            
        if first_name_lower.endswith(".pdf"):
            orig_type = "pdf"
            orig_text = extract_text_from_pdf_bytes(raw_bytes)
        else:
            orig_type = "text"
            orig_text = raw_bytes.decode("utf-8", errors="ignore")
            
        if len(orig_text) < 50:
            abort(400, "Not enough text found in document.")
            
        length_choice = request.form.get("length", "medium")
        tone = request.form.get("tone", "academic")
        
        # Map choice to number of sentences for the supervised model
        num_sentences = 7 # Default medium
        if length_choice == "short": num_sentences = 4
        if length_choice == "long": num_sentences = 15
        
        # --- CALL SUPERVISED MODEL ---
        summary_sentences = generate_supervised_summary(orig_text, num_sentences)
        
        # --- STRUCTURE RESULT FOR UI ---
        structured_data = build_structured_from_supervised(summary_sentences, tone)

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
