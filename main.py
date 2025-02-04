import os
import openai
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "".join([page.extract_text() for page in reader.pages])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "")
    file = request.files.get("file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            text = extract_text_from_file(file_path)
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
