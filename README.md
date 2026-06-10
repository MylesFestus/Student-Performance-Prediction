# 🎓 Student Placement Prediction — End-to-End Machine Learning App

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://docker.com)
[![Heroku](https://img.shields.io/badge/Heroku-Deployed-purple?logo=heroku)](https://heroku.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-green?logo=github-actions)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A production-grade machine learning web application that predicts student placement outcomes based on academic performance, skills, and demographic features. Built with Streamlit, containerised with Docker, and deployed on Heroku via a fully automated CI/CD pipeline.

---

##  Live Demo

**App URL:** [https://mylesfestus-student-performance-prediction-app-ehs7sq.streamlit.app]

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

This project predicts whether a student will be placed in a job based on key academic and personal attributes. It demonstrates a complete end-to-end machine learning workflow — from data ingestion and exploratory analysis through model training, evaluation, containerisation, and cloud deployment.

The application is designed to help academic institutions and career counsellors identify students who may need additional support before entering the job market.

---

## ✨ Features

- **Interactive Prediction Interface** — real-time placement predictions via Streamlit UI
- **Batch Prediction** — upload a CSV file to predict multiple students at once
- **Model Explainability** — feature importance visualisation showing which factors drive predictions
- **Data Exploration** — built-in EDA dashboard with charts and statistics
- **Dockerised** — fully containerised for consistent environments across local and cloud
- **Automated Deployment** — GitHub Actions CI/CD pipeline deploys to Heroku on every push to `main`

---

##  Project Structure

```
StudentPerformance/
│
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Actions CI/CD pipeline
│
├── data/
│   └── student_dataset_10000.csv  # Raw dataset
│
├── postman/
│   └── *.json                     # API test collections
│
├── app.py                         # Main Streamlit application
├── Dockerfile                     # Docker container definition
├── heroku.yml                     # Heroku container deployment config
├── Procfile                       # Heroku process definition
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version specification
├── model.pkl                      # Serialised trained ML model
├── 01_Data.ipynb                  # Data exploration and model training notebook
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
└── README.md                      # This file
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Streamlit |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Model Serialisation | Joblib / Pickle |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Cloud Platform | Heroku |
| Version Control | Git / GitHub |

---

##  Dataset

**Source:** Synthetic student placement dataset
**Size:** 10,000 records
**Target Variable:** `Placement Status` (Placed / Not Placed)

### Features

| Feature | Type | Description |
|---|---|---|
| Study Hours | Numerical | The average number of hours a student spends studying per day or week in preparation for academic activities and exams |
| Attendance Percentage | Numerical | The percentage of classes attended by the student out of the total classes conducted during a specific period |
| Sleep Hours | Numerical | The average number of hours the student sleeps per day, indicating rest and recovery levels |
| Assignments Completed | Numerical | The total number or percentage of assignments successfully completed and submitted by the student |
| Previous Scores | Numerical | The student's past academic performance, typically represented by marks, grades, or percentages obtained in previous assessments or exams|
| Exam Score | Numerical | The score or marks achieved by the student in the current examination being analyzed or predicted |
| Placement Status | Categorical | The outcome indicating whether the student has secured a placement/job opportunity, usually represented as a binary variable (Placed/Not Placed or 1/0) |

---

##  Model

### Algorithm
**Random Forest Classifier** — chosen for its robustness to outliers, ability to handle mixed feature types, and built-in feature importance.

### Training Pipeline
```
Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering & Encoding
   ↓
Train/Test Split (80/20)
   ↓
Model Training (Random Forest)
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Evaluation (Accuracy, F1, ROC-AUC)
   ↓
Model Serialisation (model.pkl)
```

### Performance Metrics

| Metric | Score |
|---|---|
| Accuracy | ~85% |
| F1 Score | ~0.84 |
| ROC-AUC | ~0.91 |
| Precision | ~0.86 |
| Recall | ~0.83 |

---

##  Installation

### Prerequisites
- Python 3.11+
- pip
- Docker (optional)
- Heroku CLI (optional)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/MylesFestus/Student-Performance-Prediction.git
cd Student-Performance-Prediction

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open at `http://localhost:8501`

---

##  Usage

### Single Prediction
1. Open the app at `http://localhost:8501`
2. Enter student details in the sidebar form
3. Click **"Predict Placement"**
4. View the prediction result and confidence score

### Batch Prediction
1. Navigate to the **"Batch Prediction"** tab
2. Upload a CSV file with the required columns
3. Download the results with predictions appended

---

##  Docker

```bash
# Build the Docker image
docker build -t placement-app .

# Run the container
docker run -p 8501:8501 placement-app

# Open in browser
# http://localhost:8501
```

### Dockerfile Overview

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

##  CI/CD Pipeline

Every push to `main` triggers the automated deployment pipeline.

### Pipeline Flow

```
Push to main
     ↓
GitHub Actions triggered
     ↓
Checkout code
     ↓
Build Docker image
     ↓
Push to Heroku Container Registry
     ↓
Release to Heroku
     ↓
App live at placement-streamlit-app.herokuapp.com
```

### deploy.yml

```yaml
name: Deploy to Heroku

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to Heroku
        uses: akhileshns/heroku-deploy@v3.13.15
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: "placement-streamlit-app"
          heroku_email: ${{ secrets.HEROKU_EMAIL }}
```

### Required GitHub Secrets

Go to **GitHub → Settings → Secrets and Variables → Actions** and add:

| Secret | Description |
|---|---|
| `HEROKU_API_KEY` | Your Heroku API token (`heroku auth:token`) |
| `HEROKU_EMAIL` | Your Heroku account email |

---

##  Deployment

### Deploy to Heroku Manually

```bash
# Login to Heroku
heroku login

# Link repo to Heroku app
heroku git:remote --app placement-streamlit-app

# Set container stack
heroku stack:set container --app placement-streamlit-app

# Push and deploy
git push heroku main

# Open the app
heroku open --app placement-streamlit-app

# View logs
heroku logs --tail --app placement-streamlit-app
```

---

### Results

### Feature Importance

Top predictors of student placement:

1. **CGPA** — 
2. **IQ Score** — 
3. **Number of Internships** — 
4. **Placement Training** — 
5. **Stream** — 
6. **History of Backlogs** — 

### Key Insights

- Students with CGPA above 7.5 have significantly higher placement rates
- Internship experience increases placement probability by ~30%
- Placement training attendance is one of the strongest controllable factors
- Gender shows minimal impact after controlling for other variables

---

## Contributing

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and commit
git add .
git commit -m "feat: add your feature description"

# 4. Push to your fork
git push origin feature/your-feature-name

# 5. Open a Pull Request on GitHub
```

---

##  Roadmap

- [ ] Add SHAP explainability plots
- [ ] Add confidence intervals to predictions
- [ ] Implement user authentication
- [ ] Add database logging of predictions
- [ ] Deploy to AWS ECS with load balancer
- [ ] Add REST API with FastAPI
- [ ] Add model retraining pipeline

---

##  License

This project is licensed under the MIT License.

---

##  Author

**Festus Attor
- GitHub: [@MylesFestus](https://github.com/MylesFestus)
- LinkedIn: [Festus Attor](www.linkedin.com/in/festus-attor)

---
