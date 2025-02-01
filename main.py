import os
import openai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Set your OpenAI API key (ensure this is configured correctly in your environment)
openai.api_key =os.getenv ("")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get the text input from the request
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Call OpenAI API with the updated method
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text but the output must be same the input language."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ]
        )
        # Extract the summary from the response
        summary = response["choices"][0]["message"]["content"].strip()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Debug mode should be off in production
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
