from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define base path for .pkl files
BASE_PATH = "D:/Case Study/Project_Arvin/YektaGroupTest/src"

# Load the model, vectorizer, and feature selector
try:
    logger.info("Loading vectorizer...")
    vectorizer = joblib.load(os.path.join(BASE_PATH, 'tfidf_vectorizer_20250505-142227.pkl'))
    logger.info("Loading feature selector...")
    selector = joblib.load(os.path.join(BASE_PATH, 'feature_selector_20250505-142227.pkl'))
    logger.info("Loading model...")
    model = joblib.load(os.path.join(BASE_PATH, 'email_classifier_20250505-142227.pkl'))
    logger.info("All components loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load components: {str(e)}")
    raise Exception(f"Failed to load components: {str(e)}")

# Define classification labels
labels = ['پشتیبانی/شکایت', 'فروش/استعلام', 'همکاری/شراکت', 'هرزنامه/تبلیغات', 'نامربوط']


@app.route("/classify", methods=["POST"])
def classify_email():
    try:
        # Validate input
        if not request.is_json:
            logger.error("Request must be JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if "text" not in data or not isinstance(data["text"], str):
            logger.error("Invalid input: 'text' field is required and must be a string")
            return jsonify({"error": "'text' field is required and must be a string"}), 400

        # Preprocess input text
        logger.info("Preprocessing input text...")
        text_vec = vectorizer.transform([data["text"]])
        text_vec_selected = selector.transform(text_vec)

        # Predict category
        logger.info("Predicting category...")
        pred = model.predict(text_vec_selected)[0]

        # Map prediction to category label
        category = labels[pred]
        logger.info(f"Predicted category: {category}")

        return jsonify({"category": category})
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return jsonify({"error": f"Classification error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
