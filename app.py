from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from pred import OSPredictor, SCPredictor, SDPredictor, MentalPredictor
import os
import json
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)  # Set a secret key

os_predictor = OSPredictor()
sc_predictor = SCPredictor()
sd_predictor = SDPredictor()
mental_predictor = MentalPredictor()

with open("responses.json", "r") as file:
    responses = json.load(file)
# Download NLTK resources (you only need to do this once)
nltk.download("punkt")
nltk.download("wordnet")

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Simple tokenization using split
    tokens = text.split()

    # Convert to lowercase
    processed_tokens = [token.lower() for token in tokens]

    # Join the tokens back into a string
    processed_text = " ".join(processed_tokens)

    return processed_text


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/models")
def models_page():
    return render_template("models.html")


@app.route("/blog")
def blog_page():
    return render_template("blog.html")


@app.route("/about")
def about_page():
    return render_template("AboutUS.html")


@app.route("/contact")
def contact_page():
    return render_template("contact.html")


@app.route("/knee_osteoporosis")
def knee_osteoporosis_page():
    return render_template("Knee_osteoporosis.html")


@app.route("/skin_cancer")
def skin_cancer_page():
    return render_template("skin_cancer.html")


@app.route("/skin_diseases")
def skin_diseases_page():
    return render_template("skin_diseases.html")


@app.route("/mental_health")
def mental_health_page():
    return render_template("mental_health.html")


@app.route("/predict_os", methods=["POST"])
def os_predict_page():
    chart_data = None

    # Get the uploaded image from the form
    image = request.files.get("image")

    if image is None or image.filename == "":
        flash("Please upload an image", "error")
        return redirect(url_for("knee_osteoporosis_page"))

    # Perform prediction using the ImagePredictor
    predictions = os_predictor.predict_image(image)
    chart_data = os_predictor.get_prediction_chart_data(predictions)

    return render_template("Knee_osteoporosis.html", chart_data=chart_data)


@app.route("/predict_sc", methods=["POST"])
def sc_predict_page():
    chart_data = None

    # Get the uploaded image from the form
    image = request.files.get("image")

    if image is None or image.filename == "":
        flash("Please upload an image", "error")
        return redirect(url_for("skin_cancer_page"))

    # Perform prediction using the ImagePredictor
    predictions = sc_predictor.predict_image(image)
    chart_data = sc_predictor.get_prediction_chart_data(predictions)

    return render_template("skin_cancer.html", chart_data=chart_data)


@app.route("/predict_sd", methods=["POST"])
def sd_predict_page():
    chart_data = None

    # Get the uploaded image from the form
    image = request.files.get("image")

    if image is None or image.filename == "":
        flash("Please upload an image", "error")
        return redirect(url_for("skin_diseases_page"))

    # Perform prediction using the ImagePredictor
    predictions = sd_predictor.predict_image(image)
    chart_data = sd_predictor.get_prediction_chart_data(predictions)

    return render_template("skin_diseases.html", chart_data=chart_data)


@app.route("/predict_me", methods=["POST"])
def me_predict_page():
    chart_data = None

    if "audio_data" in request.files:
        audio = request.files["audio_data"]
        audio.save(os.path.join(UPLOAD_FOLDER, audio.filename))
        return "Audio file uploaded successfully", 200

    audio_filename = request.form.get("audio_filename")
    print(f"{audio_filename = }")
    if audio_filename:
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        predictions = mental_predictor.predict_text("", True, audio_path)
        chart_data = mental_predictor.get_prediction_chart_data(predictions)

        return render_template("mental_health.html", chart_data=chart_data)

    input_text = request.form.get("input")

    if input_text != "":
        # Perform prediction using the MentalPredictor
        predictions = mental_predictor.predict_text(input_text)
        chart_data = mental_predictor.get_prediction_chart_data(predictions)

        return render_template("mental_health.html", chart_data=chart_data)

    if input_text == "" and audio_filename == "":
        flash("Please enter a text", "error")
        return redirect(url_for("mental_health_page"))


@app.route("/chatbot", methods=["POST"])
def chatbot_page():
    print("Request received at /chatbot")
    message = request.json.get("message")
    print("Original message:", message)

    if not message:
        return jsonify({"response": "Please provide a message."})

    processed_message = preprocess_text(message)
    print("Processed message:", processed_message)
    print("Responses dictionary:", responses)

    greetings_variations = [
        variation.lower()
        for variation in responses.get("greetings", {}).get("variations", [])
    ]
    osteoporosis = [
        term.lower() for term in responses.get("osteoporosis", {}).get("terms", [])
    ]
    Causes = [term.lower() for term in responses.get("Causes", {}).get("terms", [])]

    print("Lowercase Variations:", greetings_variations)
    print("Lowercase Medical Terms:", osteoporosis)

    if any(variation in processed_message for variation in greetings_variations):
        bot_response = random.choice(
            responses.get("greetings", {}).get("responses", [])
        )
    elif any(term in processed_message for term in osteoporosis):
        bot_response = random.choice(
            responses.get("osteoporosis", {}).get("responses", [])
        )
    elif any(term in processed_message for term in Causes):
        bot_response = random.choice(responses.get("Causes", {}).get("responses", []))

    else:
        bot_response = responses.get("default")

    print("Bot response:", bot_response)

    return jsonify({"response": bot_response})


@app.route("/NeuroBot")
def NeuroBot():
    return render_template("Chatbot.html")


@app.route("/coming_soon")
def coming_soon_page():
    return render_template("Coming_soon.html")


if __name__ == "__main__":
    app.run(debug=True, port=9000)
