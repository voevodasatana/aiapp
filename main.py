import os
import openai
from flask import Flask, request, render_template, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from pdf2docx import Converter
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

# Folders for file processing
UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB file size limit

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

# ✅ Function to Process Large PDFs in Chunks
def convert_large_pdf(pdf_path, docx_path, chunk_size=10):
    """Convert large PDFs to DOCX in smaller chunks to avoid memory issues."""
    
    # ✅ Use PyPDF2 to count the number of pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)  # Correct way to count pages
    print(f"🔹 Total Pages: {total_pages}")

    converter = Converter(pdf_path)

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        print(f"🔄 Converting pages {start} to {end}")

        # Convert only a small chunk of pages
        converter.convert(docx_path, start=start, end=end, continuous=True)

    converter.close()
    print("✅ Full document converted successfully!")

@app.route("/convert_pdf_to_word", methods=["POST"])
def convert_pdf_to_word():
    """Convert uploaded PDF to Word and return download link."""
    try:
        if "pdf-file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        pdf_file = request.files["pdf-file"]
        if pdf_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        docx_filename = filename.replace(".pdf", ".docx")
        docx_path = os.path.join(CONVERTED_FOLDER, docx_filename)

        pdf_file.save(pdf_path)

        # Convert PDF to Word
        converter = Converter(pdf_path)
        converter.convert(docx_path, start=0, end=None)
        converter.close()

        # Ensure the file was created
        if not os.path.exists(docx_path):
            return jsonify({"error": "File conversion failed"}), 500

        # Generate a correct download URL
        download_url = url_for("download_file", filename=docx_filename, _external=True)
        return jsonify({"download_url": download_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/<filename>")
def download_file(filename):
    """Serve the converted Word file for download."""
    file_path = os.path.join(CONVERTED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


