import os
import pickle
import PyPDF2
import numpy as np
import nltk
from flask import Flask, request, abort

# Download NLTK data (Required for Render)
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# ==============================================================================
# 1. LOAD MODEL (Updated for Root Directory)
# ==============================================================================
# Since you uploaded files to root, we look in the current directory (".")
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
        print("✅ Lightweight model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Warning: Model files not found. Expected 'model.pkl' and 'tfidf.pkl' in current folder.")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def extract_text_from_pdf(file_stream):
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

def generate_extractive_summary(text, num_sentences=7):
    if not model or not vectorizer:
        return "Error: Model not active. Please check server logs."

    try:
        sentences = nltk.sent_tokenize(text)
    except:
        return "Error processing text."

    if len(sentences) < 1:
        return "No text found in document."
    
    try:
        features = vectorizer.transform(sentences)
        # Predict importance (Class 1)
        scores = model.predict_proba(features)[:, 1]
    except Exception as e:
        return f"Prediction Error: {e}"
    
    # Sort by score (Highest Importance first)
    ranked_sentences = sorted(
        zip(range(len(sentences)), scores, sentences), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Keep top N and sort back to original order for readability
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[0])
    
    summary = " ".join([s[2] for s in top_sentences])
    return summary

# ==============================================================================
# 3. WEB INTERFACE
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
        :root { --primary: #2563EB; --bg: #F3F4F6; }
        body { font-family: sans-serif; background: var(--bg); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; max-width: 400px; text-align: center; }
        h1 { color: #1F2937; margin-bottom: 0.5rem; }
        p { color: #6B7280; margin-bottom: 1.5rem; }
        .upload-btn { border: 2px dashed #D1D5DB; padding: 2rem; border-radius: 8px; cursor: pointer; transition: 0.2s; }
        .upload-btn:hover { border-color: var(--primary); background: #EFF6FF; }
        button { background: var(--primary); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-weight: bold; margin-top: 1rem; cursor: pointer; width: 100%; }
        button:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="card">
        <h1>PolicyBrief AI 🏥</h1>
        <p>Upload a policy document to summarize.</p>
        <form action="/summarize" method="post" enctype="multipart/form-data">
            <div class="upload-btn" onclick="document.getElementById('file').click()">
                📂 Click to Upload PDF
                <input type="file" name="file" id="file" accept=".pdf" style="display:none" required onchange="this.parentElement.style.borderColor='#2563EB'">
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
    <!DOCTYPE html>
    <html lang="en">
    <head> 
        <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: sans-serif; background: #F3F4F6; padding: 2rem; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h2 {{ color: #2563EB; border-bottom: 2px solid #E5E7EB; padding-bottom: 1rem; }}
            p {{ line-height: 1.6; color: #374151; }}
            a {{ display: inline-block; margin-top: 1rem; color: #2563EB; text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📄 Summary Result</h2>
            <p>{summary}</p>
            <a href="/">← Summarize Another</a>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
