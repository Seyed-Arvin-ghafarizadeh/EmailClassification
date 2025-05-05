# EmailClassification
Persian Email Dataset Generator
This project provides a Python script to generate a synthetic dataset of 100,000 Persian emails, evenly distributed across five categories: پشتیبانی/شکایت مشتری, درخواست فروش/استعلام قیمت, همکاری/پیشنهاد شراکت, هرزنامه/تبلیغات, and نامربوط. The dataset is designed for training and evaluating multi-class email classification models, with realistic email content, subjects, and noise to simulate real-world scenarios.
Features

Generates 20,000 emails per category, totaling 100,000 emails.
Uses Persian-specific templates with varied subjects and bodies for each category.
Incorporates realistic elements like noise (e.g., typos, random characters) in 30% of emails and HTML formatting in 20% of emails.
Outputs a CSV file (emails.csv) with text (subject + body) and label (category) columns.
Leverages the faker library with Persian locale (fa_IR) for authentic names, companies, and other details.

Prerequisites

Python 3.6 or higher
Required Python packages (see Installation)

Installation

Clone the Repository:
git clone https://github.com/your-username/persian-email-dataset-generator.git
cd persian-email-dataset-generator


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required packages using pip:
pip install pandas numpy faker



Usage

Run the Script:Execute the script to generate the dataset:
python generate_emails.py

This will create emails.csv in the project directory, containing 100,000 emails (20,000 per category).

Output:

The script outputs emails.csv with two columns:
text: The email content (subject + body).
label: The category (one of the five categories).

⭐ If you find this project useful, please give it a star on GitHub!
