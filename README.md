# 🚀 End-to-End MLOps Deployment — Diabetes Prediction System

## 📌 Project Objective

This project was developed to **learn and implement real-world MLOps practices** by building, packaging, and deploying a machine learning inference system on cloud infrastructure.

The focus of this work was **not model optimization**, but rather the **end-to-end lifecycle of ML systems**, including experiment tracking, artifact management, containerization, and cloud deployment.

---

## 🧠 Learning Goals

* Understand ML system lifecycle
* Implement experiment tracking
* Manage data & model artifacts
* Containerize inference services
* Deploy ML applications on AWS
* Debug real infrastructure failures
* Separate training vs serving environments

---

## 🏗️ System Architecture

```
Dataset → Training Pipeline → Model Artifact
        → Docker Image → AWS EC2 → Streamlit UI → User Predictions
```

---

## ⚙️ Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas / NumPy

### MLOps & Experimentation

* MLflow (experiment tracking)
* DVC (data & artifact versioning)
* Git / GitHub

### Deployment & Serving

* Docker
* AWS EC2
* Streamlit

---

## 📂 Project Structure

```
diabetes-mlops/
│
├── data/
│   └── raw/diabetes.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   └── train.py
│
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── dvc.yaml
└── README.md
```

---

## 🔄 MLOps Workflow

### 1️⃣ Data Ingestion

Dataset stored outside Git tracking to follow large-file best practices.
DVC was initialized to manage dataset and artifact reproducibility.

---

### 2️⃣ Model Training

Training script:

```
python src/train.py
```

Outputs:

```
models/model.pkl
```

Model artifact is generated reproducibly and used for inference packaging.

---

### 3️⃣ Experiment Tracking (MLflow)

MLflow was integrated to track:

* Model parameters
* Evaluation metrics
* Experiment runs
* Serialized model artifacts

This enabled comparison across training configurations.

However, MLflow was **disabled during deployment** to avoid runtime database dependencies and ensure container portability.

---

### 4️⃣ Artifact Management

* Model artifacts excluded from Git
* Managed via reproducible training
* Avoided committing large binaries

---

### 5️⃣ Containerization

Inference system packaged into Docker:

```
docker build -t diabetes-app .
```

Model artifact baked into image at build time to ensure stateless serving.

---

### 6️⃣ Application Serving

Streamlit UI enables real-time predictions:

```
docker run -p 8501:8501 diabetes-app
```

Users input health indicators and receive diabetes risk classification.

---

## ☁️ AWS EC2 Deployment

Deployment steps:

* Launch Ubuntu EC2 instance
* Configure storage & networking
* Install Docker & Python
* Transfer dataset
* Train model on cloud instance
* Build Docker image
* Run Streamlit container
* Expose public inference endpoint

---

## 🌐 Networking & Access

Application served via:

```
http://<EC2_PUBLIC_IP>:8501
```

Port exposure and security group configuration were validated.

Backend service and container networking were confirmed functional via curl and internal routing tests.

Due to browser/network environment constraints and free-tier instance limitations, the UI experienced loading constraints despite confirmed backend availability.

---

## 🧪 Features

* Interactive Streamlit prediction UI
* Multi-class diabetes risk output
* Containerized inference pipeline
* Cloud-hosted deployment
* Experiment tracking integration
* Reproducible artifact generation

---

## 🛠️ Infrastructure Challenges Solved

This project involved debugging real production-like issues:

* Docker image disk exhaustion
* EBS volume resizing & filesystem expansion
* Memory OOM kills during training
* CPU throttling on micro instances
* MLflow schema conflicts on deployment servers
* Container port binding & exposure
* AWS security group configuration
* Public vs private IP routing
* Streamlit runtime resource constraints

These provided hands-on DevOps and cloud troubleshooting experience.

---

## 🚀 Run Locally

### Install dependencies

```
pip install -r requirements.txt
```

### Train model

```
python src/train.py
```

### Launch UI

```
streamlit run streamlit_app.py
```

---

## 🐳 Run via Docker

```
docker build -t diabetes-app .
docker run -p 8501:8501 diabetes-app
```

---

## 📊 Disclaimer

This project is intended for **MLOps engineering learning and deployment practice**.

The predictive model is not optimized for clinical or production healthcare use.

---

## 👨‍💻 Author

**Om Muddebihal**
M.S. Data Science — IIT Bombay
Research Project Scientist — Disease Modelling

---

## ⭐ Key Takeaway

This project demonstrates the full lifecycle of deploying a machine learning system — from experimentation and artifact management to containerization and cloud inference — emphasizing real-world MLOps practices over model development.
