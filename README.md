##Persian-Email-Classifier##
This project implements a complete pipeline for generating, classifying, and deploying a Persian email classification system. It includes:

Dataset Generation: A script to create a synthetic dataset of 100,000 Persian emails across five categories: پشتیبانی/شکایت مشتری, درخواست فروش/استعلام قیمت, همکاری/پیشنهاد شراکت, هرزنامه/تبلیغات, and نامربوط.
Model Training: A Logistic Regression model trained on the dataset, using TF-IDF vectorization and feature selection for multi-class classification.
API Deployment: A Flask API to serve the trained model, accepting email text via POST requests and returning the predicted category.
Testing: Scripts to test the API and evaluate model performance.

The project is designed for researchers, data scientists, and developers working on Persian NLP tasks, providing a robust baseline for email classification.
Features

Dataset: 100,000 synthetic Persian emails (20,000 per category) with realistic subjects, bodies, noise (30% of emails), and HTML formatting (20% of emails).
Model: Logistic Regression with TF-IDF features (15,000 max, 10,000 selected via chi-squared), achieving expected test accuracy of 85–95%.
API: Flask-based REST API for real-time email classification, deployed on http://127.0.0.1:8000/classify.
Optimization: Hyperparameter tuning with Optuna, 5-fold cross-validation, and feature selection for efficiency.
Evaluation: Comprehensive metrics (accuracy, macro F1-score, per-class precision/recall/F1), confusion matrix, and learning curve.

Project Flow

Generate Dataset: Use generate_emails.py to create persian_emails_robust.csv with 100,000 emails.
Train Model: Run train_model.py to preprocess the dataset, train a Logistic Regression model, and save the model (email_classifier_*.pkl), vectorizer (tfidf_vectorizer_*.pkl), and feature selector (feature_selector_*.pkl).
Deploy API: Use flask_email_classifier.py to serve the model via a Flask API.
Test API: Use test_api.py to send POST requests and verify predictions.

Prerequisites

Python 3.9 or higher
Required Python packages (see Installation)

Installation

Clone the Repository:
git clone https://github.com/your-username/persian-email-classifier.git
cd persian-email-classifier


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install required packages:
pip install -r requirements.txt

See requirements.txt for the full list, including pandas, numpy, scikit-learn, faker, flask, requests, and optuna.

Usage
1. Generate the Dataset
Run the dataset generation script:
python src/generate_emails.py


Output: Creates persian_emails_robust.csv with 100,000 emails (20,000 per category).
Customization: Adjust emails_per_category in generate_emails.py to change the dataset size.

2. Train the Model
Train the Logistic Regression model:
python src/train_model.py


Process:
Loads persian_emails_robust.csv.
Applies TF-IDF vectorization (15,000 features, bigrams, Persian stop words).
Selects top 10,000 features using chi-squared.
Tunes hyperparameters with Optuna (50 trials, L1/L2 penalties, C range: 1e-5 to 1.0).
Performs 5-fold cross-validation.
Saves model, vectorizer, and feature selector with timestamped filenames (e.g., email_classifier_20250505-142227.pkl).


Outputs:
Saved files: tfidf_vectorizer_*.pkl, feature_selector_*.pkl, email_classifier_*.pkl.
Metrics CSV (metrics_*.csv) with accuracy, macro F1, and per-class scores.
Visualizations: confusion_matrix_*.png, learning_curve_*.png.



3. Deploy the Flask API
Run the Flask API to serve the model:
cd src
python flask_email_classifier.py


Details:
Loads the saved model, vectorizer, and feature selector.
Starts a Flask server on http://127.0.0.1:8000.
Exposes the /classify endpoint for POST requests.


Expected Output:INFO:__main__:Loading vectorizer...
INFO:__main__:Loading feature selector...
INFO:__main__:Loading model...
INFO:__main__:All components loaded successfully.
 * Serving Flask app 'flask_email_classifier'
 * Running on http://127.0.0.1:8000 (Press CTRL+C to quit)



4. Test the API
Test the API with a Python script:
python src/test_api.py


Example test_api.py:
import requests

url = "http://127.0.0.1:8000/classify"
emails = [
    "سلام مشکل در نرمافزار خریداری شده دارم.",  # پشتیبانی/شکایت
    "تخفیف ویژه فقط امروز!",  # هرزنامه/تبلیغات
    "سلام، درباره آب و هوا صحبت کنیم.",  # نامربوط
    "درخواست قیمت برای 100 دستگاه لپ‌تاپ.",  # فروش/استعلام
    "پیشنهاد همکاری در پروژه جدید."  # همکاری/شراکت
]

for email in emails:
    data = {"text": email}
    response = requests.post(url, json=data)
    print(f"Email: {email}")
    print(f"Response: {response.json()}\n")


Expected Output:
Email: سلام مشکل در نرمافزار خریداری شده دارم.
Response: {'category': 'پشتیبانی/شکایت'}

Email: تخفیف ویژه فقط امروز!
Response: {'category': 'هرزنامه/تبلیغات'}

Email: سلام، درباره آب و هوا صحبت کنیم.
Response: {'category': 'نامربوط'}

Email: درخواست قیمت برای 100 دستگاه لپ‌تاپ.
Response: {'category': 'فروش/استعلام'}

Email: پیشنهاد همکاری در پروژه جدید.
Response: {'category': 'همکاری/شراکت'}


Alternative Testing:

PowerShell:Invoke-WebRequest -Uri "http://127.0.0.1:8000/classify" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"text": "سلام مشکل در نرمافزار خریداری شده دارم."}' | Select-Object -ExpandProperty Content


Postman: POST to http://127.0.0.1:8000/classify with JSON body {"text": "your email text"}.
Browser: Visit http://127.0.0.1:8000/classify (for GET requests, returns error; use POST).



Dataset Details

Categories: 5 (پشتیبانی/شکایت مشتری, درخواست فروش/استعلام قیمت, همکاری/پیشنهاد شراکت, هرزنامه/تبلیغات, نامربوط)
Emails per Category: 20,000
Total Emails: 100,000
Columns in persian_emails_robust.csv:
text: Email subject + body (separated by newline).
label: Category label.


Features:
Persian-specific content using faker (fa_IR locale).
Noise (typos, random characters) in 30% of emails.
HTML formatting in 20% of emails.
Category-specific templates for realistic variation.



Model Details

Algorithm: Logistic Regression (multinomial, saga or liblinear solver).
Preprocessing:
TF-IDF vectorization (15,000 features, bigrams, Persian stop words, sublinear TF).
Feature selection (top 10,000 features via chi-squared).


Training:
Stratified train/validation/test split (72,000/8,000/20,000).
Hyperparameter tuning with Optuna (C: 1e-5 to 1.0, L1/L2 penalties).
5-fold cross-validation.


Performance: Expected test accuracy of 85–95%, macro F1-score reported in metrics_*.csv.

API Details

Endpoint: POST /classify
Input: JSON payload with text field (e.g., {"text": "سلام مشکل در نرمافزار خریداری شده دارم."}).
Output: JSON response with category field (e.g., {"category": "پشتیبانی/شکایت"}).
Error Handling: Returns 400 for invalid input, 500 for server errors.

Dependencies
See requirements.txt:
pandas>=1.5.0
numpy>=1.21.0
faker>=8.0.0
scikit-learn>=1.0.0
optuna>=3.0.0
flask>=2.0.0
requests>=2.25.0

Install with:
pip install -r requirements.txt

Troubleshooting

FileNotFoundError for .pkl Files:
Ensure tfidf_vectorizer_*.pkl, feature_selector_*.pkl, and email_classifier_*.pkl are in src/.
Update BASE_PATH in flask_email_classifier.py if files are elsewhere.


Server Stops Running:
Run with debug=True in flask_email_classifier.py for detailed logs.
Check for port conflicts:netstat -aon | findstr :8000
taskkill /PID <PID> /F


Use a different port:app.run(host="127.0.0.1", port=8001, debug=False)




ConnectionRefusedError:
Ensure the Flask server is running before testing.
Allow port 8000 in firewall:netsh advfirewall firewall add rule name="Allow Port 8000" dir=in action=allow protocol=TCP localport=8000


⭐ If you find this project useful, please give it a star on GitHub!
