import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LOGS_PATH = "data/prediction_logs.json"
MONITORING_DATA_PATH = "data/monitoring_data.json"
BASELINE_METRICS_PATH = "results/model_metrics.json"
MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
DEGRADATION_THRESHOLD = 0.05

os.makedirs("results", exist_ok=True)

print("==========================================")
print("       PART 1: PREDICTION MONITORING      ")
print("==========================================")

# Step 1: Read prediction logs recorded by app.py
if os.path.exists(LOGS_PATH):
    with open(LOGS_PATH, "r") as f:
        logs = json.load(f)
else:
    logs = []

total_predictions = len(logs)
print(f"Total Predictions Logged: {total_predictions}")

if total_predictions > 0:
    spam_count = sum(1 for item in logs if item.get("prediction") == "spam")
    ham_count = sum(1 for item in logs if item.get("prediction") == "ham")
    
    # Calculate average probability for spam predictions
    spam_probs = [item["probability"] for item in logs if item.get("prediction") == "spam"]
    avg_spam_prob = sum(spam_probs) / len(spam_probs) if spam_probs else 0.0

    print(f"Spam Predictions        : {spam_count}")
    print(f"Ham Predictions         : {ham_count}")
    print(f"Average Spam Probability: {avg_spam_prob:.2f}")

    # Plot distribution of prediction outcomes
    plt.figure(figsize=(6, 4))
    plt.bar(["Spam", "Ham"], [spam_count, ham_count], color=["red", "green"])
    plt.title("Spam vs Ham Predictions")
    plt.xlabel("Prediction Class")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    dist_path = "results/prediction_distribution.png"
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved prediction distribution graph to {dist_path}\n")
else:
    print("No prediction logs found yet. Run the API and make predictions to view distribution logs.\n")


print("==========================================")
print(" PART 2: MODEL PERFORMANCE DEGRADATION   ")
print("==========================================")

# Step 2: Load evaluation monitoring dataset
if not os.path.exists(MONITORING_DATA_PATH):
    print(f"Error: {MONITORING_DATA_PATH} not found!")
    exit(1)

with open(MONITORING_DATA_PATH, "r") as f:
    monitoring_data = json.load(f)

messages = [item["message"].lower().strip() for item in monitoring_data]
actual_labels = [item["actual_label"] for item in monitoring_data]

# Step 3: Load saved model and vectorizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Error: Saved model or vectorizer not found. Please run train.py first!")
    exit(1)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Step 4: Make predictions on monitoring data
X_mon = vectorizer.transform(messages)
predicted_labels = model.predict(X_mon)

# Step 5: Calculate current metrics on monitoring dataset
curr_acc = accuracy_score(actual_labels, predicted_labels)
curr_prec = precision_score(actual_labels, predicted_labels, pos_label="spam", zero_division=0)
curr_rec = recall_score(actual_labels, predicted_labels, pos_label="spam", zero_division=0)
curr_f1 = f1_score(actual_labels, predicted_labels, pos_label="spam", zero_division=0)

print("Current Monitoring Evaluation:")
print(f"Accuracy : {curr_acc:.2f}")
print(f"Precision: {curr_prec:.2f}")
print(f"Recall   : {curr_rec:.2f}")
print(f"F1 Score : {curr_f1:.2f}\n")

# Step 6: Compare with original baseline F1 score stored in results/model_metrics.json
if os.path.exists(BASELINE_METRICS_PATH):
    with open(BASELINE_METRICS_PATH, "r") as f:
        baseline_metrics = json.load(f)
    orig_f1 = baseline_metrics.get("f1_score", curr_f1)
else:
    orig_f1 = curr_f1

f1_diff = orig_f1 - curr_f1

print("Degradation Analysis:")
print(f"Original F1 Score : {orig_f1:.2f}")
print(f"Current F1 Score  : {curr_f1:.2f}")
print(f"F1 Score Difference: {f1_diff:.2f}")
print(f"Threshold         : {DEGRADATION_THRESHOLD}")

if f1_diff > DEGRADATION_THRESHOLD:
    print("\nWARNING: Model performance may have degraded.")
else:
    print("\nModel performance is stable.")
print("==========================================")
