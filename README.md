# 🧠 Price Optimization ML Platform

A full-stack machine learning platform that predicts the **conversion probability** of a product offer based on pricing and contextual features. Built with FastAPI, Streamlit, MLflow, and deployed on Render.

---

## 🚀 Features

- ✅ Logistic Regression model trained on simulated pricing data  
- 📈 MLflow tracking + artifact logging  
- 🔌 FastAPI for real-time inference  
- 🖥️ Streamlit frontend for live testing  
- ☁️ Cloud deployment (Render for backend, Streamlit Cloud for UI)

---


## 🧪 Getting Started Locally

### 1. Clone and Setup

```bash
git clone https://github.com/rockyglen/price-opt-platform.git
cd price-opt-platform
python -m venv venv
source venv/bin/activate     
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train/train_model.py
```

### 3. Start Backend API

```bash
uvicorn src.inference.inference_api:app --host 0.0.0.0 --port 8000
```

# ☁️ Deployment  
Backend on Render.com  
Connect GitHub repo  

Use render.yaml for config  

Set up environment variables securely on Render dashboard  



