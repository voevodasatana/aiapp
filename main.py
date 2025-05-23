import os
import openai
import requests
from flask import Flask, request, render_template, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from pdf2docx import Converter
from PyPDF2 import PdfReader, errors as PyPDF2Errors # For specific PyPDF2 errors
from docx import Document
from bs4 import BeautifulSoup
import uuid # For unique filenames
import logging

# --- Application Setup ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB file size limit
MAX_TEXT_INPUT_LENGTH = 100000  # Max chars for direct text/webpage summarization input to OpenAI

# OpenAI API Key (Set in Render/Environment Variables)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    app.logger.warning("OPENAI_API_KEY environment variable not set. Summarization will fail.")

# Allowed file types for summarization
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def allowed_file(filename):
    """Check if file is allowed based on extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generates a unique filename using UUID and secure_filename."""
    name, ext = os.path.splitext(original_filename)
    # Ensure the original name part is also secure before combining
    secure_name_part = secure_filename(name) if name else "file" # Handle empty name part
    return f"{secure_name_part}_{uuid.uuid4().hex}{ext}"


def extract_text_from_file(file_path):
    """
    Extract text from the uploaded file.
    - PDFs: First 10 pages.
    - DOCX: First 10 paragraphs (Note: this is different from PDF's page-based limit).
    - TXT: Full content, attempts UTF-8 then latin-1 encoding.
    Returns extracted text or None on failure.
    """
    extracted_text = ""
    filename = os.path.basename(file_path) 

    try:
        if file_path.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            except UnicodeDecodeError:
                app.logger.warning(f"UTF-8 decoding failed for {filename}. Trying latin-1.")
                try:
                    with open(file_path, "r", encoding="latin-1") as f: 
                        extracted_text = f.read()
                except Exception as e_latin1:
                    app.logger.error(f"Failed to read TXT file {filename} with fallback encoding: {e_latin1}")
                    return None 
            except Exception as e_utf8:
                app.logger.error(f"Failed to read TXT file {filename}: {e_utf8}")
                return None

        elif file_path.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for i in range(min(10, len(reader.pages))):
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
            except PyPDF2Errors.PdfReadError as e_pdf_read: 
                app.logger.error(f"Error reading PDF file {filename} (possibly corrupted or password-protected): {e_pdf_read}")
                return None
            except Exception as e_pdf:
                app.logger.error(f"Error processing PDF file {filename}: {e_pdf}")
                return None

        elif file_path.endswith(".docx"):
            try:
                doc = Document(file_path)
                # Note: Extracts first 10 paragraphs, differing from PDF's page-based handling.
                extracted_text = "\n".join([para.text for para in doc.paragraphs[:10]])
            except Exception as e_docx: 
                app.logger.error(f"Error processing DOCX file {filename} (possibly corrupted): {e_docx}")
                return None
        
        extracted_text = ' '.join(extracted_text.split()) # Normalize whitespace
        # Optional: Truncate very long extracted texts further if needed before summarization input limit
        # if len(extracted_text) > SOME_ABSOLUTE_MAX_INTERNAL_LENGTH:
        #     extracted_text = extracted_text[:SOME_ABSOLUTE_MAX_INTERNAL_LENGTH]

    except Exception as e:
        app.logger.error(f"General error extracting text from {file_path}: {e}")
        return None 

    return extracted_text

# --- Routes ---
@app.route("/")
def index():
    """Render the main webpage."""
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    """AI Summarization Route"""
    text_to_summarize = request.form.get("text", "")
    file = request.files.get("file")
    source_description = "direct text input"

    if file:
        if not file.filename: # Check if a file was selected but no filename (e.g. empty input)
             return jsonify({"error": "No file selected or file has no name."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type '{file.filename.rsplit('.', 1)[1]}' not allowed for summarization."}), 400
        
        original_filename = file.filename
        unique_filename = generate_unique_filename(original_filename)
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        try:
            file.save(file_path)
            source_description = f"file: {original_filename}"
            app.logger.info(f"File {original_filename} saved as {unique_filename} for summarization.")
            
            text_to_summarize = extract_text_from_file(file_path)
            if text_to_summarize is None: 
                app.logger.error(f"Text extraction failed for {original_filename}.")
                return jsonify({"error": "Unable to extract text from the file. The file might be corrupted, password-protected, or in an unsupported format."}), 400
            if not text_to_summarize.strip():
                app.logger.warning(f"No text extracted from {original_filename} or extracted text is empty.")
                return jsonify({"error": "No content found in the file after extraction."}), 400
        except Exception as e: # Catch errors during file save or text extraction call
            app.logger.error(f"Error saving or processing uploaded file {original_filename}: {e}")
            return jsonify({"error": "Error processing uploaded file."}), 500

    if not text_to_summarize.strip(): 
        return jsonify({"error": "No text provided for summarization."}), 400

    if len(text_to_summarize) > MAX_TEXT_INPUT_LENGTH:
        app.logger.warning(f"Input text from {source_description} for summarization exceeds {MAX_TEXT_INPUT_LENGTH} characters. Truncating.")
        text_to_summarize = text_to_summarize[:MAX_TEXT_INPUT_LENGTH]

    if not openai.api_key:
        app.logger.error("OpenAI API key not configured.")
        return jsonify({"error": "Summarization service is not configured. Administrator intervention required."}), 503

    try:
        app.logger.info(f"Requesting summarization from OpenAI for content from {source_description} (length: {len(text_to_summarize)} chars).")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI that summarizes text accurately and concisely."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text_to_summarize}"}
            ],
            timeout=30 
        )
        summary = response.choices[0].message.content.strip()
        app.logger.info(f"Summarization successful for content from {source_description}.")
        return jsonify({"summary": summary})
    except openai.error.OpenAIError as e_openai: 
        app.logger.error(f"OpenAI API error during summarization for {source_description}: {e_openai}")
        return jsonify({"error": f"Summarization service error: {type(e_openai).__name__}. Please try again later."}), 503 
    except requests.exceptions.Timeout:
        app.logger.error(f"OpenAI API call timed out for {source_description}.")
        return jsonify({"error": "Summarization service timed out. Please try again later."}), 504 
    except Exception as e:
        app.logger.error(f"Generic error during summarization for {source_description}: {e}")
        return jsonify({"error": "An unexpected error occurred during summarization."}), 500


@app.route("/convert_pdf_to_word", methods=["POST"])
def convert_pdf_to_word():
    """Convert uploaded PDF to Word (only first 10 pages)."""
    if "pdf-file" not in request.files:
        return jsonify({"error": "No PDF file part in the request."}), 400

    pdf_file = request.files["pdf-file"]
    if pdf_file.filename == "":
        return jsonify({"error": "No PDF file selected."}), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Uploaded file is not a PDF."}), 400

    original_filename = pdf_file.filename
    unique_pdf_filename = generate_unique_filename(original_filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, unique_pdf_filename)

    original_name_no_ext, _ = os.path.splitext(secure_filename(original_filename))
    desired_download_name = original_name_no_ext + ".docx"
    
    unique_docx_internal_name = os.path.splitext(unique_pdf_filename)[0] + ".docx"
    docx_path = os.path.join(CONVERTED_FOLDER, unique_docx_internal_name)

    converter = None
    try:
        pdf_file.save(pdf_path)
        app.logger.info(f"PDF file {original_filename} saved as {unique_pdf_filename} for conversion.")

        converter = Converter(pdf_path)
        converter.convert(docx_path, start=0, end=10)  
        
        app.logger.info(f"PDF {unique_pdf_filename} converted to DOCX {unique_docx_internal_name}.")

        if not os.path.exists(docx_path):
            app.logger.error(f"File conversion failed for {unique_pdf_filename}, DOCX not found at {docx_path}.")
            return jsonify({"error": "File conversion process failed to create output file."}), 500
        
        download_url = url_for("download_file", 
                               filename_internal=unique_docx_internal_name, 
                               filename_original=desired_download_name, 
                               _external=True)
        return jsonify({"download_url": download_url, "message": f"Successfully converted '{original_filename}'."})

    except Exception as e:
        app.logger.error(f"Error during PDF to Word conversion for {original_filename} (saved as {unique_pdf_filename}): {e}")
        # Check if it's a known pdf2docx issue (e.g. password protected)
        if "wrong password" in str(e).lower() or "password required" in str(e).lower():
             return jsonify({"error": "Conversion error: The PDF might be password-protected."}), 400
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500
    finally:
        if converter:
            try:
                converter.close()
            except Exception as e_close: 
                app.logger.error(f"Error closing PDF converter for {unique_pdf_filename}: {e_close}")
        # For production: consider deleting pdf_path here if conversion was successful and it's no longer needed.
        # if os.path.exists(pdf_path):
        #     try:
        #         os.remove(pdf_path)
        #         app.logger.info(f"Cleaned up uploaded PDF: {pdf_path}")
        #     except OSError as e_remove:
        #         app.logger.error(f"Error removing uploaded PDF {pdf_path}: {e_remove}")


@app.route("/download/<path:filename_internal>") 
def download_file(filename_internal):
    """Serve the converted Word file for download."""
    safe_internal_filename = secure_filename(filename_internal)
    if safe_internal_filename != filename_internal:
        app.logger.warning(f"Download attempt with potentially unsafe filename: '{filename_internal}' sanitized to '{safe_internal_filename}'")
        return jsonify({"error": "Invalid filename."}), 400

    # Resolve paths carefully
    base_dir = os.path.abspath(os.path.dirname(__file__))
    converted_folder_abs = os.path.abspath(os.path.join(base_dir, CONVERTED_FOLDER))
    target_path = os.path.abspath(os.path.join(converted_folder_abs, safe_internal_filename))

    # Security check: Ensure the resolved path is strictly within the CONVERTED_FOLDER
    if not target_path.startswith(converted_folder_abs + os.sep): # os.sep for platform independence
        app.logger.error(f"Attempt to download file outside designated folder: {safe_internal_filename} resolved to {target_path}")
        return jsonify({"error": "Access denied or file not found."}), 404

    if os.path.exists(target_path) and os.path.isfile(target_path):
        filename_original_query = request.args.get("filename_original", safe_internal_filename)
        safe_original_filename_download = secure_filename(filename_original_query)

        app.logger.info(f"Serving file '{safe_internal_filename}' for download as '{safe_original_filename_download}'.")
        return send_file(target_path, as_attachment=True, download_name=safe_original_filename_download)
    else:
        app.logger.warning(f"Download request for non-existent or non-file path: '{safe_internal_filename}' (resolved to '{target_path}')")
        return jsonify({"error": "File not found or no longer available."}), 404


def extract_text_from_webpage(url):
    """
    Fetch and extract readable text from a webpage.
    Note: Robust webpage text extraction is complex. This is a basic implementation.
    Consider libraries like Trafilatura or Newspaper3k for more advanced needs.
    Returns (text, error_message_or_none)
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15) 
        response.raise_for_status() 

        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]): # Remove common non-content tags
            script_or_style.decompose()
        
        # Basic attempt to find main content (can be improved)
        texts = []
        main_selectors = ['article', 'main', '.main-content', '#content', '.entry-content', '.post-body']
        content_area = None
        for selector in main_selectors:
            if soup.select_one(selector):
                content_area = soup.select_one(selector)
                break
        
        target_element = content_area if content_area else soup.body # Fallback to body
        
        if target_element:
            paragraphs = target_element.find_all("p", recursive=True) # Get all p tags within target
            if paragraphs:
                for para in paragraphs:
                    texts.append(para.get_text(separator=" ", strip=True))
            else: # If no <p>, get all text from the target_element
                texts.append(target_element.get_text(separator=" ", strip=True))
        
        text_content = "\n".join(filter(None, texts)) # Join non-empty lines
        text_content = ' '.join(text_content.split()) # Normalize whitespace

        if not text_content.strip():
             app.logger.warning(f"No meaningful text extracted from webpage: {url} after parsing.")
             # No error message, but empty text

        return text_content.strip(), None 
    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout fetching webpage {url}.")
        return None, "Failed to fetch webpage: Connection timed out."
    except requests.exceptions.RequestException as e_req:
        app.logger.error(f"Network error fetching webpage {url}: {e_req}")
        return None, f"Failed to fetch webpage: {type(e_req).__name__}."
    except Exception as e:
        app.logger.error(f"Error extracting text from webpage {url}: {e}")
        return None, f"Failed to extract content due to an unexpected error: {str(e)}"


@app.route("/summarize_webpage", methods=["POST"])
def summarize_webpage():
    """Summarize content from a webpage URL."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. JSON payload expected."}), 400
        
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "Invalid URL format. Must start with http:// or https://"}), 400

    app.logger.info(f"Attempting to extract text from webpage: {url}")
    text, error_msg = extract_text_from_webpage(url)
    
    if error_msg: 
        app.logger.error(f"Failed to extract content from {url}: {error_msg}")
        return jsonify({"error": error_msg}), 400 # Use specific error from extraction
    if not text or not text.strip():
        app.logger.warning(f"No text extracted from webpage or extracted text is empty: {url}")
        return jsonify({"error": "No content found on the webpage or content could not be extracted."}), 400

    if len(text) > MAX_TEXT_INPUT_LENGTH: 
        app.logger.warning(f"Webpage text for summarization from {url} (length {len(text)}) exceeds limit {MAX_TEXT_INPUT_LENGTH}. Truncating.")
        text = text[:MAX_TEXT_INPUT_LENGTH]

    if not openai.api_key:
        app.logger.error("OpenAI API key not configured for webpage summarization.")
        return jsonify({"error": "Summarization service is not configured. Administrator intervention required."}), 503
        
    try:
        app.logger.info(f"Requesting summarization from OpenAI for webpage: {url} (text length: {len(text)}).")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI that summarizes webpage content accurately and professionally and doesn't make it too short, make the summarization format like bullet points for each main idea"},
                {"role": "user", "content": f"Please summarize the key information from the following webpage content:\n\n{text}"}
            ],
            timeout=30 
        )
        summary = response.choices[0].message.content.strip()
        app.logger.info(f"Summarization successful for webpage: {url}")
        return jsonify({"summary": summary})
    except openai.error.OpenAIError as e_openai:
        app.logger.error(f"OpenAI API error during webpage summarization for {url}: {e_openai}")
        return jsonify({"error": f"Summarization service error: {type(e_openai).__name__}. Please try again later."}), 503
    except requests.exceptions.Timeout:
        app.logger.error(f"OpenAI API call timed out for webpage {url}.")
        return jsonify({"error": "Summarization service timed out. Please try again later."}), 504
    except Exception as e:
        app.logger.error(f"Generic error during webpage summarization for {url}: {e}")
        return jsonify({"error": "An unexpected error occurred during webpage summarization."}), 500

# --- Main Execution ---
if __name__ == "__main__":
    # For production, use a proper WSGI server (e.g., Gunicorn, uWSGI) instead of Flask's development server.
    # Consider a robust strategy for cleaning up UPLOAD_FOLDER and CONVERTED_FOLDER in a production environment
    # (e.g., a scheduled cron job or task queue worker).
    app.logger.info("Starting Flask development server.")
    # Use PORT environment variable if available (common for cloud platforms like Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=os.environ.get("FLASK_DEBUG", "False").lower() == "true", host="0.0.0.0", port=port)
