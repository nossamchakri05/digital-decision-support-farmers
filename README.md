# 🌾 Digital Decision Support System for Farmers

In remote agricultural regions, farmers often lack timely access to expert guidance and efficient tools for managing farm data, leading to reduced productivity and delayed decision-making. This project addresses those challenges by replacing outdated paper-based systems with a smart, digital solution that supports data-driven farming decisions.

It is an integrated platform built using Flask, MySQL, and advanced Machine Learning and Deep Learning models. With features like crop yield prediction (XGBoost), disease detection (ResNet-50), and subsidy eligibility assessment (Random Forest), the system delivers real-time, personalized recommendations through a secure and user-friendly interface.

---

## 📌 Overview

This project addresses challenges in remote agricultural regions by replacing outdated paper-based practices with a digital decision-support system. It leverages:

- **MySQL** for secure data storage
- **XGBoost** for crop yield prediction (82.8% accuracy)
- **ResNet-50** for crop disease detection (79.6% accuracy)
- **Random Forest** for subsidy eligibility prediction (81.3% accuracy)

---

## 💡 Key Features

### 👨‍🌾 User Portal
- Secure Registration/Login with validation
- Upload crop images for real-time disease detection
- Input farm details for crop yield predictions
- Check government subsidy eligibility

### 🔐 Admin Portal
- View and manage registered users
- Monitor predictions and data submissions
- Ensure data integrity and log user actions

---

## 🛠️ Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Database**: MySQL
- **ML/DL Models**: XGBoost, ResNet-50, Random Forest

---

## 📁 Folder Structure
digital-decision-support-farmers/
│
├── app/
│ ├── static/
│ │ ├── css/
│ │ ├── js/
│ │ └── images/
│ ├── templates/
│ │ ├── login.html
│ │ ├── register.html
│ │ ├── dashboard.html
│ │ └── admin.html
│ ├── init.py
│ ├── routes.py
│ ├── models.py
│ ├── ml_models/
│ │ ├── xgboost_model.pkl
│ │ ├── resnet50_model.h5
│ │ └── rf_model.pkl
│ └── utils/
│ └── database.py
│
├── dataset/
│ └── crop_images/
│
├── requirements.txt
├── README.md
└── run.py


---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MySQL
- Flask
