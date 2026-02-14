# 🏦 Bank Customer Churn Prediction using Artificial Neural Network

An end-to-end Deep Learning project that predicts whether a bank customer is likely to churn (leave the bank) or continue using services.  
The model is built using an Artificial Neural Network (ANN) and deployed with Streamlit for an interactive web interface.

---

## 🎯 Project Objective

Customer retention is one of the biggest challenges in the banking sector.  
This project aims to:

- Identify customers who are at high risk of leaving
- Help financial institutions take proactive retention actions
- Use Deep Learning for accurate binary classification

Prediction Output:
- `1` → Customer is likely to churn  
- `0` → Customer is likely to stay  

---

## 🧠 Model Overview

The model is developed using **TensorFlow and Keras** and trained on structured customer data.

### 📌 Features Used

- Credit Score
- Geography (One-Hot Encoded)
- Gender (Label Encoded)
- Age
- Tenure
- Account Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## 🏗 Model Architecture

The Artificial Neural Network consists of:

- Input Layer (11 neurons)
- Hidden Dense Layers with ReLU activation
- Dropout layers for regularization
- Output Layer with Sigmoid activation (Binary classification)

### ⚙️ Training Configuration

- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy
- Train-Test Split: 80/20

---

## 🔧 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📊 Project Workflow

1. Data Cleaning & Preprocessing
2. Encoding Categorical Variables
3. Feature Scaling
4. Model Building (ANN)
5. Model Training & Validation
6. Deployment using Streamlit

---

## 🚀 Streamlit Web App

The project includes an interactive Streamlit application that allows:

- Real-time user input
- Instant churn prediction
- Clean and simple user interface
- Lightweight local deployment

---
