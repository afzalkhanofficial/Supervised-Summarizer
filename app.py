import os
import pickle
import PyPDF2
import re
import numpy as np
import nltk
from flask import Flask, request, abort
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is downloaded (Required for Render)
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# ==============================================================================
# 1. LOAD MODEL
# ==============================================================================
# We look for model files in the current root directory
MODEL_DIR = "." 
model_path = os.path.join(MODEL_DIR, 'model.pkl')
tfidf_path = os.path.join(MODEL_DIR, 'tfidf.pkl')

model = None
vectorizer = None

if os.path.exists(model_path) and os.path.exists(tfidf_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Smart model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Warning: Model files not found in current directory.")

# ==============================================================================
# 2. CLEANING & TEXT EXTRACTION
# ==============================================================================
def clean_text(text):
    """
    Aggressive cleaning to remove headers, page numbers, and artifacts.
    """
    # 1. Remove "Page X" or "--- PAGE ---"
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'---.*?---', '', text)
    
    # 2. Remove Section Headers with Titles (e.g., "3.3.3 Re-Orienting Public Hospitals:")
    # Regex: Number sequence + spaces + text + colon
    text = re.sub(r'\d+(\.\d+)*\s+[A-Za-z\s\-]+:', '', text)
    
    # 3. Remove standalone leading section numbers (e.g., "2.4.1")
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text)
    
    # 4. Remove extra whitespace
    return text.strip()

def extract_text_from_pdf(file_stream):
    """Extracts text from PDF file stream."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + " "
    except Exception as e:
        print(f"PDF Error: {e}")
        return ""
    return text

# ==============================================================================
# 3. GENERATE SUMMARY (With Redundancy Check)
# ==============================================================================
def generate_extractive_summary(text, num_sentences=7):
    if not model or not vectorizer:
        return "Error: Model not active. Please check server logs."

    # 1. Split into sentences
    try:
        raw_sentences = nltk.sent_tokenize(text)
    except:
        return "Error processing text."

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
        return "No valid text found in document."

    # 4. Predict Importance Scores
    try:
        features = vectorizer.transform(valid_sentences)
        # Get probability of Class 1 (Important)
        scores = model.predict_proba(features)[:, 1]
    except Exception as e:
        return f"Prediction Error: {e}"
    
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
            # vstack creates a matrix of selected vectors to compare against current one
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
    summary_parts = [valid_sentences[i] for i in final_sentences_indices]
    summary = " ".join(summary_parts)
    
    return summary

# ==============================================================================
# 4. WEB INTERFACE ROUTES
# ==============================================================================
@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PolicyBrief AI</title>
    <style>
        :root { --primary: #0f766e; --bg: #f0fdfa; }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: white; padding: 2.5rem; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); width: 100%; max-width: 450px; text-align: center; }
        h1 { color: var(--primary); margin-bottom: 0.5rem; }
        p { color: #64748b; margin-bottom: 1.5rem; }
        .upload-area { border: 2px dashed #cbd5e1; padding: 2rem; border-radius: 12px; cursor: pointer; transition: 0.3s; margin: 1.5rem 0; background: #f8fafc; }
        .upload-area:hover { border-color: var(--primary); background: #ccfbf1; }
        button { background: var(--primary); color: white; border: none; padding: 1rem; border-radius: 8px; font-weight: 600; width: 100%; cursor: pointer; font-size: 1rem; transition: 0.2s; }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
    </style>
</head>
<body>
    <div class="card">
        <h1>PolicyBrief AI 🧠</h1>
        <p>Smart Extractive Summarization for Healthcare Policies</p>
        <form action="/summarize" method="post" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('file').click()">
                📂 <strong>Click to Upload PDF</strong><br>
                <span style="font-size: 0.9rem; color: #94a3b8;">(Max 5MB recommended)</span>
                <input type="file" name="file" id="file" accept=".pdf" style="display:none" onchange="this.parentElement.style.borderColor='#0f766e'; this.parentElement.style.background='#e6fffa';">
            </div>
            <button type="submit">Generate Summary</button>
        </form>
    </div>
</body>
</html>
    '''

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files: abort(400)
    file = request.files['file']
    if file.filename == "": abort(400)
    
    # Process
    text = extract_text_from_pdf(file.stream)
    summary = generate_extractive_summary(text)
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head> 
        <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Summary Result</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f1f5f9; padding: 2rem; display: flex; justify-content: center; }}
            .container {{ max-width: 800px; background: white; padding: 3rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            h2 {{ color: #0f766e; border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem; margin-top: 0; }}
            .summary-text {{ line-height: 1.8; color: #334155; font-size: 1.1rem; text-align: justify; }}
            .btn {{ display: inline-block; margin-top: 2rem; padding: 0.8rem 1.5rem; background: #0f766e; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; }}
            .btn:hover {{ opacity: 0.9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📄 Executive Summary</h2>
            <div class="summary-text">
                {summary}
            </div>
            <a href="/" class="btn">← Summarize Another File</a>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
