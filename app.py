import os
import pickle
import PyPDF2
import re
import numpy as np
import nltk
from flask import Flask, request, abort
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# Load Model
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
    except Exception as e:
        print(f"Error: {e}")

# --- IMPROVED CLEANING ---
def clean_text(text):
    """Removes headers, page numbers, and artifacts."""
    # Remove "Page X"
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    # Remove section numbers like "2.3.1" at start of line
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text)
    return text.strip()

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + " "
    except: return ""
    return text

def generate_extractive_summary(text, num_sentences=7):
    if not model or not vectorizer: return "Model Error"

    # 1. Split & Clean
    raw_sentences = nltk.sent_tokenize(text)
    clean_sentences = [clean_text(s) for s in raw_sentences]
    
    # Filter out short junk (headers/footers)
    valid_sentences = []
    original_indices = []
    for i, s in enumerate(clean_sentences):
        if len(s) > 40: # Ignore very short lines
            valid_sentences.append(s)
            original_indices.append(i)

    if not valid_sentences: return "No valid text found."

    # 2. Predict Importance
    features = vectorizer.transform(valid_sentences)
    scores = model.predict_proba(features)[:, 1]

    # 3. Smart Selection (Redundancy Removal)
    ranked_indices = np.argsort(scores)[::-1] # Indices of best scores
    
    selected_indices = []
    selected_vectors = []
    
    for idx in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break
            
        # Check similarity to already selected sentences
        current_vec = features[idx]
        is_redundant = False
        if selected_vectors:
            # Calculate similarity with all currently selected
            # Note: sparse matrix operation
            sims = cosine_similarity(current_vec, np.vstack(selected_vectors))
            if np.max(sims) > 0.65: # If >65% similar to an existing sentence, skip
                is_redundant = True
        
        if not is_redundant:
            selected_indices.append(idx)
            selected_vectors.append(current_vec.toarray()[0])

    # 4. Sort back by original order for flow
    # We map back to the 'valid_sentences' list
    final_sentences = sorted(selected_indices)
    summary = " ".join([valid_sentences[i] for i in final_sentences])
    
    return summary

# --- ROUTES (Same as before) ---
@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PolicyBrief AI 2.0</title>
    <style>
        :root { --primary: #0f766e; --bg: #f0fdfa; }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: white; padding: 2.5rem; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); width: 100%; max-width: 450px; text-align: center; }
        h1 { color: var(--primary); margin-bottom: 0.5rem; }
        .upload-area { border: 2px dashed #cbd5e1; padding: 2rem; border-radius: 12px; cursor: pointer; transition: 0.3s; margin: 1.5rem 0; }
        .upload-area:hover { border-color: var(--primary); background: #ccfbf1; }
        button { background: var(--primary); color: white; border: none; padding: 1rem; border-radius: 8px; font-weight: 600; width: 100%; cursor: pointer; font-size: 1rem; }
    </style>
</head>
<body>
    <div class="card">
        <h1>PolicyBrief AI 2.0 🧠</h1>
        <p>Smart Extractive Summarization</p>
        <form action="/summarize" method="post" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('file').click()">
                📂 Upload PDF Document
                <input type="file" name="file" id="file" accept=".pdf" style="display:none" onchange="this.parentElement.style.borderColor='#0f766e'">
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
    
    text = extract_text_from_pdf(file.stream)
    summary = generate_extractive_summary(text)
    
    return f'''
    <div style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 40px; border-radius: 12px; background: #ffffff; box-shadow: 0 4px 20px rgba(0,0,0,0.08); line-height: 1.8; color: #334155;">
        <h2 style="color: #0f766e; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px;">📄 Executive Summary</h2>
        <p>{summary}</p>
        <a href="/" style="display: inline-block; margin-top: 20px; color: #0f766e; font-weight: bold; text-decoration: none;">&larr; Analyze Another File</a>
    </div>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
