# Spam Classification Model with FastAPI, MLflow, Docker and Basic Monitoring

An end-to-end, beginner-friendly MLOps project designed for MSc Data Science students and beginners in Python, Machine Learning, and MLOps.

---

## 📌 Project Overview

This project demonstrates the complete end-to-end workflow of building, tracking, serving, monitoring, and containerizing a Machine Learning model without using complex tools or enterprise-level abstractions.

### End-to-End MLOps Flow

```text
Dataset
   ↓
Data Cleaning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow Tracking
   ↓
Save Model
   ↓
FastAPI REST API
   ↓
Prediction
   ↓
JSON Prediction Logging
   ↓
Basic Monitoring
   ↓
Performance Degradation Check
   ↓
Docker Deployment
```

---

## 📁 Project Structure

```text
spam-classification-mlops/
│
├── data/
│   ├── spam_data.json            # Training dataset with 50 spam/ham messages
│   ├── monitoring_data.json      # Dataset for evaluating model performance post-deployment
│   └── prediction_logs.json      # Log file storing real-time API predictions
│
├── models/
│   ├── spam_model.pkl            # Trained Multinomial Naive Bayes model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF Vectorizer
│
├── results/
│   ├── confusion_matrix.png        # Saved confusion matrix plot from training
│   ├── prediction_distribution.png # Distribution plot of predictions made by API
│   └── model_metrics.json         # Baseline evaluation metrics (F1 score, Accuracy, etc.)
│
├── train.py                       # Model training and MLflow experiment tracking script
├── app.py                         # FastAPI REST API serving model predictions
├── monitor.py                     # Script for prediction logging analysis and degradation check
├── requirements.txt               # List of required Python packages
├── Dockerfile                     # Docker container configuration
├── .dockerignore                  # Files excluded from Docker image build
└── README.md                      # Complete project documentation
```

---

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **Pandas & NumPy**: Data loading, cleaning, and array manipulation.
* **Scikit-learn**: TF-IDF text vectorization and Multinomial Naive Bayes model.
* **FastAPI & Uvicorn**: High-performance Python web application framework for building REST APIs.
* **MLflow**: Local experiment tracking for logging parameters, metrics, and confusion matrix artifacts.
* **Joblib**: Model serialization and saving binary `.pkl` files.
* **Matplotlib**: Visualizing confusion matrices and prediction distributions.
* **Docker**: Containerizing the FastAPI application for easy deployment.
* **JSON**: File format for data storage and logging (No databases used!).

---

## 💻 Step-by-Step Local Running Guide

Follow these steps to set up and run the project locally on your machine.

### Step 1 — Create Virtual Environment
Open your terminal/command prompt and run:
```bash
python -m venv venv
```

### Step 2 — Activate Virtual Environment
* **Windows (Command Prompt / PowerShell):**
  ```bash
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train Model & Track with MLflow
Run the training script to load data, train Naive Bayes model, evaluate metrics, save artifacts, and log experiment to MLflow:
```bash
python train.py
```

Expected Terminal Output:
```text
Starting Spam Classification Model Training...
Loaded dataset with 50 samples.
Model training complete.

Model Evaluation
----------------
Accuracy : 0.90
Precision: 1.00
Recall   : 0.80
F1 Score : 0.89

Saved confusion matrix to results/confusion_matrix.png
Saved baseline metrics to results/model_metrics.json
Logged run parameters, metrics, and artifact to MLflow.
Saved model to models/spam_model.pkl
Saved vectorizer to models/tfidf_vectorizer.pkl

Training workflow completed successfully!
```

### Step 5 — Start FastAPI Application
Start the REST API web server using Uvicorn:
```bash
uvicorn app:app --reload
```

### Step 6 — Test API & Swagger UI
1. Open your browser and navigate to the root API endpoint:
   `http://127.0.0.1:8000/`
2. Open interactive API Documentation (Swagger UI):
   `http://127.0.0.1:8000/docs`
3. Click on `POST /predict` -> **Try it out** -> enter a JSON body:
   ```json
   {
       "message": "Congratulations! You won a free prize!"
   }
   ```
4. Click **Execute** to see the model prediction.

### Step 7 — Run Monitoring Script
Open a new terminal tab (keeping Uvicorn running) and execute:
```bash
python monitor.py
```

---

## 📊 MLflow Experiment Tracking Guide

MLflow is used to track model parameters, metrics, and visual artifacts for every training run.

### Running MLflow UI
After executing `python train.py`, launch the MLflow web interface locally:
```bash
mlflow ui
```

Open your web browser and go to:
```text
http://127.0.0.1:5000
```

### Key MLflow Concepts for Students
* **Experiment**: A group of related runs. Here, the experiment is named `Spam Classification`.
* **Runs**: An individual execution of `train.py`.
* **Parameters**: Key hyperparameter settings recorded during training:
  * `model_name`: `"MultinomialNB"`
  * `test_size`: `0.2`
  * `vectorizer`: `"TfidfVectorizer"`
* **Metrics**: Quantifiable evaluation scores:
  * `accuracy`, `precision`, `recall`, `f1_score`
* **Artifacts**: Output files created during training, such as `confusion_matrix.png`.

---

## 🐳 Running the Project with Docker

Docker package your application, dependencies, model files, and runtime environment into a standardized container image.

### Step 1 — Build Docker Image
Run the `docker build` command to compile the Docker image:
```bash
docker build -t spam-classification-api .
```
* **Explanation**: `docker build` reads the `Dockerfile` in the current directory (`.`) and creates an image named `spam-classification-api`.

### Step 2 — Run Docker Container
Start a container from your built image:
```bash
docker run -p 8000:8000 spam-classification-api
```
* **Explanation**: `-p 8000:8000` maps **Host Machine Port 8000** to **Container Port 8000**. Requests sent to `http://localhost:8000` on your computer are routed directly inside the container.

### Step 3 — Open API in Browser
Go to `http://localhost:8000` to verify the containerized API is running.

### Step 4 — Open Interactive Swagger Documentation
Go to `http://localhost:8000/docs` in your web browser.

### Step 5 — Test Containerized Prediction
Test the `/predict` endpoint with:
```json
{
    "message": "You have won 1000 dollars. Claim now!"
}
```
Response:
```json
{
    "message": "You have won 1000 dollars. Claim now!",
    "prediction": "spam",
    "probability": 1.0
}
```

### Step 6 — Stop Container
To stop the running Docker container:
1. Open a new terminal window and list running containers:
   ```bash
   docker ps
   ```
2. Copy the `CONTAINER ID` and run:
   ```bash
   docker stop <CONTAINER_ID>
   ```

---

## 🚫 Why `.dockerignore` is Used

The `.dockerignore` file prevents temporary, local, or secret files from being copied into the Docker image during `docker build`.

Ignored entries:
* `__pycache__` & `*.pyc`: Compiled Python bytecode files.
* `.git`: Version control history (unneeded inside runtime container).
* `venv`: Local virtual environment (Docker creates its own clean Python environment).
* `.env`: Local environment configuration or secrets.
* `mlruns`: Local MLflow experiment runs folder (keeps Docker image lightweight).

---

## 🧪 Complete Student Practical Demonstration

Follow this step-by-step practical demonstration to experience the entire MLOps lifecycle from training to detecting performance degradation:

1. **Install Python 3.11+** on your computer.
2. **Create virtual environment**: `python -m venv venv`
3. **Activate virtual environment**: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux).
4. **Install requirements**: `pip install -r requirements.txt`
5. **Train the model**: Run `python train.py`.
6. **Check model metrics**: Inspect `results/model_metrics.json` and view `results/confusion_matrix.png`.
7. **Check generated model files**: Ensure `models/spam_model.pkl` and `models/tfidf_vectorizer.pkl` exist.
8. **Start MLflow UI**: Run `mlflow ui`.
9. **Open MLflow UI**: Visit `http://127.0.0.1:5000` and inspect Parameters, Metrics, and Confusion Matrix Artifacts.
10. **Start FastAPI**: Run `uvicorn app:app --reload`.
11. **Open Swagger UI**: Visit `http://127.0.0.1:8000/docs`.
12. **Send spam message**: Test POST `/predict` with `"Claim your free cash prize now!"`.
13. **Send normal message**: Test POST `/predict` with `"Are we having class today?"`.
14. **Check prediction logs**: Open `data/prediction_logs.json` in Notepad or VS Code to see logged requests.
15. **Run monitoring script**: Run `python monitor.py`. Observe total predictions, spam vs ham counts, and baseline degradation check.
16. **Check monitoring output image**: Open `results/prediction_distribution.png`.
17. **Demonstrate Model Degradation**:
    * Open `data/monitoring_data.json` in VS Code or Notepad.
    * Modify the `actual_label` values for spam messages (e.g. change `"actual_label": "spam"` to `"actual_label": "ham"` for several messages to simulate concept drift or bad data).
18. **Run monitoring script again**: Execute `python monitor.py` again.
    * Observe that the F1 score drops significantly below the `0.05` threshold and triggers:
      `WARNING: Model performance may have degraded.`
19. **Build Docker image**: Run `docker build -t spam-classification-api .`
20. **Run Docker container**: Run `docker run -p 8000:8000 spam-classification-api`
21. **Test API in Docker**: Visit `http://localhost:8000/docs` and send a test prediction.

---

## 🎓 MLOps Concepts Explained for Beginners

### Machine Learning Model
A mathematical representation learned from data. Here, TF-IDF converts text into numerical word importance scores, and Naive Bayes calculates the probability that a message is "spam" or "ham".

### Model Serving
The process of making a trained ML model available to user applications (web sites, mobile apps) via network requests, usually as a REST API.

### REST API
Representational State Transfer Application Programming Interface. A standardized way for client applications to send JSON requests over HTTP and receive JSON responses.

### FastAPI
A modern, fast web framework for building APIs with Python based on standard Python type hints. It automatically generates interactive Swagger documentation.

### MLflow
An open-source MLOps platform for experiment tracking. It logs hyperparameters, performance metrics, and output files across different model training iterations.

### Prediction Logging
Saving runtime API inputs and prediction outputs into a persistent log file (`data/prediction_logs.json`). This provides data audit trails and raw material for performance monitoring.

### Monitoring
Continuously auditing deployed ML models in production to track request volume, prediction class distributions, and incoming prediction accuracy.

### Model Drift / Performance Degradation
Over time, real-world data patterns change (e.g., spammers use new words), causing model accuracy to decline.
* **Model Performance**: Accuracy/F1 score calculated at training time.
* **Performance Degradation**: The drop in F1 score when evaluating the trained model against newly collected monitoring data. If the drop exceeds a threshold (`DEGRADATION_THRESHOLD = 0.05`), retraining is required.

### Docker
A containerization technology that packages your code, Python runtime, installed libraries, and saved model files into an isolated "container". This guarantees that your application runs identically on any computer or server.
