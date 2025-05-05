import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
import optuna
import joblib
import time

# Comprehensive Persian stop words list (expanded to avoid hazm dependency)
PERSIAN_STOP_WORDS = [
    'و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'برای', 'روی', 'هم',
    'می', 'تا', 'آن', 'ما', 'شما', 'آنها', 'خود', 'یا', 'هر', 'اگر', 'چون',
    'بی', 'پس', 'نه', 'هیچ', 'الان', 'حالا', 'ولی', 'اما', 'چه', 'چی', 'کی',
    'کجا', 'چطور', 'همه', 'یکی', 'دو', 'سه', 'چند', 'دیگر', 'دیگری', 'همین',
    'همان', 'فقط', 'تنها', 'باید', 'نمی', 'بود', 'شده', 'شود', 'کنید', 'کنم',
    'کنند', 'کرد', 'کرده', 'دار', 'دارد', 'دارند', 'داشت', 'داریم', 'باش',
    'باشد', 'باشید', 'باشم', 'باشند', 'بسیار', 'خیلی', 'زیرا', 'چنانچه',
    'گرچه', 'تاکنون', 'هنوز', 'همیشه', 'گاه', 'گاهی', 'پیش', 'پس', 'بالا',
    'پایین', 'جلو', 'عقب', 'داخل', 'خارج', 'نزدیک', 'دور', 'بزرگ', 'کوچک'
]

# Load and prepare data
DATA_PATH = 'D:\Case Study\Project_Arvin\YektaGroupTest\data\emails.csv'  # Update to your dataset path
df = pd.read_csv(DATA_PATH)
texts = df['text'].values
labels = pd.factorize(df['label'])[0]
class_names = list(df['label'].unique())  # Dynamically get class names

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    stratify=labels,
    shuffle=True,
    random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    stratify=y_train,
    shuffle=True,
    random_state=42
)

# Vectorize text using TF-IDF with optimizations
vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased for large dataset
    stop_words=PERSIAN_STOP_WORDS,
    lowercase=True,
    sublinear_tf=True,
    ngram_range=(1, 2)  # Include unigrams and bigrams
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Feature selection to reduce dimensionality and overfitting
selector = SelectKBest(chi2, k=10000)  # Select top 10,000 features
X_train_vec = selector.fit_transform(X_train_vec, y_train)
X_val_vec = selector.transform(X_val_vec)
X_test_vec = selector.transform(X_test_vec)

# Define Optuna objective function for hyperparameter tuning
def objective(trial):
    C = trial.suggest_float('C', 1e-5, 1.0, log=True)  # Narrower range for stronger regularization
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = 'liblinear' if penalty == 'l1' else 'saga'  # Use saga for faster convergence
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=1000,
        tol=1e-3,  # Early stopping tolerance
        random_state=42
    )
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_val_vec)
    return accuracy_score(y_val, y_pred)

# Perform hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=-1)  # Parallelize trials
print('Best hyperparameters:', study.best_params)
print('Best validation accuracy:', study.best_value)

# Train final model with best hyperparameters
best_params = study.best_params
solver = 'liblinear' if best_params['penalty'] == 'l1' else 'saga'
X_train_val = np.concatenate([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])
X_train_val_vec = vectorizer.transform(X_train_val)
X_train_val_vec = selector.transform(X_train_val_vec)
final_model = LogisticRegression(
    C=best_params['C'],
    penalty=best_params['penalty'],
    solver=solver,
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Measure training time
start_time = time.time()
final_model.fit(X_train_val_vec, y_train_val)
training_time = time.time() - start_time
print(f'Training Time (CPU): {training_time:.2f} seconds')

# Cross-validation to assess overfitting
cv_scores = cross_val_score(final_model, X_train_val_vec, y_train_val, cv=5, scoring='accuracy', n_jobs=-1)
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Initialize metrics dictionary dynamically
metrics_data = {
    'Dataset': [],
    'Accuracy': [],
    'Macro_F1_Score': [],
}
for class_name in class_names:
    metrics_data[f'Precision_{class_name}'] = []
    metrics_data[f'Recall_{class_name}'] = []
    metrics_data[f'F1_Score_{class_name}'] = []

# Evaluate on training set
y_train_pred = final_model.predict(X_train_val_vec)
train_accuracy = accuracy_score(y_train_val, y_train_pred)
train_f1 = f1_score(y_train_val, y_train_pred, average='macro')
print("\nTraining Set Metrics:")
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Train Macro F1-Score: {train_f1:.4f}')
print("Training Set Classification Report:")
train_report = classification_report(y_train_val, y_train_pred, target_names=class_names, digits=4, output_dict=True)
print(classification_report(y_train_val, y_train_pred, target_names=class_names, digits=4))

# Add training metrics to dictionary
metrics_data['Dataset'].append('Train')
metrics_data['Accuracy'].append(train_accuracy)
metrics_data['Macro_F1_Score'].append(train_f1)
for class_name in class_names:
    metrics_data[f'Precision_{class_name}'].append(train_report[class_name]['precision'])
    metrics_data[f'Recall_{class_name}'].append(train_report[class_name]['recall'])
    metrics_data[f'F1_Score_{class_name}'].append(train_report[class_name]['f1-score'])

# Evaluate on test set
y_test_pred = final_model.predict(X_test_vec)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro')
print("\nTest Set Metrics:")
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Macro F1-Score: {test_f1:.4f}')
print("Test Set Classification Report:")
test_report = classification_report(y_test, y_test_pred, target_names=class_names, digits=4, output_dict=True)
print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))

# Add test metrics to dictionary
metrics_data['Dataset'].append('Test')
metrics_data['Accuracy'].append(test_accuracy)
metrics_data['Macro_F1_Score'].append(test_f1)
for class_name in class_names:
    metrics_data[f'Precision_{class_name}'].append(test_report[class_name]['precision'])
    metrics_data[f'Recall_{class_name}'].append(test_report[class_name]['recall'])
    metrics_data[f'F1_Score_{class_name}'].append(test_report[class_name]['f1-score'])

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('metrics_results.csv', index=False, encoding='utf-8-sig')
print("Metrics saved to 'metrics_results.csv'")

# Plot confusion matrix (for test set)
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Plot learning curve
train_sizes, train_scores, val_scores = learning_curve(
    final_model,
    X_train_vec,
    y_train,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42,
    n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation accuracy')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.savefig('learning_curve.png')
plt.show()

# Save model and vectorizer with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
joblib.dump(final_model, f'email_classifier_{timestamp}.pkl')
joblib.dump(vectorizer, f'tfidf_vectorizer_{timestamp}.pkl')
joblib.dump(selector, f'feature_selector_{timestamp}.pkl')
print(f"Model, vectorizer, and feature selector saved with timestamp {timestamp}")
