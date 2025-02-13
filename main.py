import os
import openai
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
#from flask_cors import CORS
from pdf2docx import Converter
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

# Folders for file processing
UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# OpenAI API Key (Set in Render/Environment Variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Allowed file types for summarization
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}

def allowed_file(filename):
    """Check if file is allowed based on extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    """Extract text from uploaded file (PDF, DOCX, or TXT)."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "".join([page.extract_text() or "" for page in reader.pages])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return None

@app.route("/")
def index():
    """Render the main webpage."""
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    """AI Summarization Route"""
    text = request.form.get("text", "")
    file = request.files.get("file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extract text from file
        text = extract_text_from_file(file_path)
        if not text:
            return jsonify({"error": "Unable to extract text from the file."}), 400

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI that summarizes text."},
                {"role": "user", "content": f"Summarize this:\n\n{text}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/convert_pdf_to_word", methods=["POST"])
def convert_pdf_to_word():
    try:
        if "pdf-file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        pdf_file = request.files["pdf-file"]
        if pdf_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        docx_path = os.path.join(CONVERTED_FOLDER, filename.replace(".pdf", ".docx"))

        pdf_file.save(pdf_path, buffer_size=8192)

        # **Debugging Step: Check if File Exists**
        if not os.path.exists(pdf_path):
            print(f"🚨 ERROR: File was not saved at {pdf_path}")
            return jsonify({"error": "File saving failed"}), 500

        print(f"Converting: {pdf_path} -> {docx_path}")

        # Convert PDF to Word
        converter = Converter(pdf_path)
        converter.convert(docx_path, start=0, end=None)
        converter.close()

        return send_file(docx_path, as_attachment=True, download_name="converted.docx")

    except Exception as e:
        print(f"🔥 ERROR: {str(e)}")  # Print actual error
        return jsonify({"error": str(e)}), 500

    return send_file(docx_path, as_attachment=True, download_name="converted.docx")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
