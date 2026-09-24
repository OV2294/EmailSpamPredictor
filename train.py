import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Create required directories if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Starting Spam Classification Model Training...")

# Step 1: Read the dummy JSON dataset
# We load the dataset containing text messages and their corresponding spam/ham labels
with open("data/spam_data.json", "r") as file:
    raw_data = json.load(file)

# Step 2: Convert JSON into a Pandas DataFrame
# Pandas DataFrame allows us to easily manipulate and clean tabular data
df = pd.DataFrame(raw_data)
print(f"Loaded dataset with {len(df)} samples.")

# Step 3: Clean the text data
# We convert all text messages to lowercase and remove trailing spaces so that
# case variations (e.g., 'WIN' vs 'win') do not confuse the vectorizer
df['message'] = df['message'].str.lower().str.strip()

# Step 4: Separate features (X) and target labels (y)
# X contains the input text messages, y contains the target label ('spam' or 'ham')
X = df['message']
y = df['label']

# Step 5: Split data into training and testing sets
# We use 80% of data for training the model and 20% for testing model accuracy
# stratify=y ensures balanced distribution of spam and ham in both train and test sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Create TF-IDF Vectorizer
# Machine learning algorithms require numerical inputs. TF-IDF converts text into
# numbers based on word frequency and importance across messages
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Step 7: Train Multinomial Naive Bayes Model
# Naive Bayes is a simple and effective probabilistic classifier for text classification
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training complete.")

# Step 8: Make predictions on the test dataset
y_pred = model.predict(X_test)

# Step 9: Calculate evaluation metrics
# We calculate Accuracy, Precision, Recall, and F1 Score for positive label 'spam'
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label="spam")
rec = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy : {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall   : {rec:.2f}")
print(f"F1 Score : {f1:.2f}\n")

# Step 10: Create and save Confusion Matrix plot
# Confusion matrix shows how many spam/ham messages were correctly or incorrectly classified
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Spam Classification Confusion Matrix")
cm_path = "results/confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"Saved confusion matrix to {cm_path}")

# Save metrics to JSON for degradation monitoring in monitor.py
metrics_data = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1_score": round(float(f1), 4)
}
with open("results/model_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)
print("Saved baseline metrics to results/model_metrics.json")

# Step 11: Log experiment details with MLflow
# MLflow tracks model training experiments, parameters, metrics, and output files
mlflow.set_experiment("Spam Classification")

with mlflow.start_run():
    # Log hyperparameters and parameters
    mlflow.log_param("model_name", "MultinomialNB")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    
    # Log model metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Log artifact (confusion matrix plot)
    mlflow.log_artifact(cm_path)
    
    print("Logged run parameters, metrics, and artifact to MLflow.")

# Step 12 & 13: Save model and vectorizer binaries
# joblib saves Python objects to disk so FastAPI can load them later for inference
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("Saved model to models/spam_model.pkl")
print("Saved vectorizer to models/tfidf_vectorizer.pkl")
print("\nTraining workflow completed successfully!")
