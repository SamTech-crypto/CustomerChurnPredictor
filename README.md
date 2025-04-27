# **CustomerChurnPredictor**

**CustomerChurnPredictor** is an AI-powered web application built to predict customer churn and optimize retention strategies. Using machine learning and reinforcement learning, the app forecasts churn likelihood based on customer behavior and sentiment, and suggests tailored retention actions. 

## **Features**

- **Customer Churn Prediction**: 
  - Utilizes a **Random Forest Classifier** to predict customer churn based on key features such as customer tenure, monthly spend, support tickets, and satisfaction score.
  
- **Sentiment Analysis**: 
  - Analyzes customer feedback using **NLTK's VADER** sentiment analysis to gauge customer sentiment, helping determine the likelihood of churn.

- **Retention Strategy Optimization**: 
  - Implements a **reinforcement learning agent** that explores different strategies (email, SMS, chatbot, or discounts) to optimize customer retention. The agent learns and adjusts based on feedback, maximizing customer retention over time.

- **Real-Time Predictions**: 
  - Provides real-time churn predictions and allows for the dynamic monitoring of high-risk customers.
  
- **Model Explainability**: 
  - Uses **Streamlit** for an interactive dashboard, visualizing the importance of features in predicting churn and providing insights into how decisions are made.

- **Interactive Dashboard**:
  - Built with **Streamlit**, the app features an intuitive and user-friendly dashboard for model training, real-time predictions, feature importance visualization, and strategy execution.

## **Technologies Used**

- **Python 3.9+**
- **Streamlit**: For creating interactive web applications.
- **Scikit-learn**: For building and training the machine learning models.
- **NLTK**: For performing sentiment analysis on customer feedback.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For visualizations.
- **Pickle**: For saving and loading the trained model.
- **Smtplib**: For sending automated retention emails (configurable with your SMTP server).

## **Installation & Setup**

1. Clone this repository:

   ```bash
   git clone https://github.com/SamTech-crypto/CustomerChurnPredictor.git
   cd CustomerChurnPredictor
