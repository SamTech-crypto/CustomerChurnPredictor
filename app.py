import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Password for authentication
PASSWORD = "your_password"  # Change this to your desired password

# Simulated customer data (replace with real data in production)
def generate_sample_data(n=1000):
    np.random.seed(42)
    data = {
        'customer_id': [str(uuid.uuid4()) for _ in range(n)],
        'tenure': np.random.randint(1, 60, n),
        'monthly_spend': np.random.uniform(10, 500, n),
        'support_tickets': np.random.randint(0, 10, n),
        'last_interaction_days': np.random.randint(1, 90, n),
        'satisfaction_score': np.random.uniform(1, 5, n),
        'churn': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'feedback': [f"Customer feedback {i}" for i in range(n)]
    }
    return pd.DataFrame(data)

# Reinforcement Learning Agent for Strategy Optimization
class RetentionAgent:
    def __init__(self, actions=['email', 'sms', 'chatbot', 'discount']):
        self.actions = actions
        self.q_table = {action: 0 for action in actions}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return max(self.q_table, key=self.q_table.get)

    def update(self, action, reward):
        self.q_table[action] += self.alpha * (reward + self.gamma * max(self.q_table.values()) - self.q_table[action])

# Email sending function (configure with real SMTP in production)
def send_retention_email(customer_id, strategy):
    msg = MIMEMultipart()
    msg['From'] = os.getenv('SMTP_USER')  # Ensure this is an environment variable
    msg['To'] = f'customer_{customer_id}@example.com'
    msg['Subject'] = 'We Value Your Loyalty!'
    body = f"Dear Customer,\n\nWe've noticed you might need {strategy}. Contact us for a special offer!\n\nBest,\nTeam"
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP(os.getenv('SMTP_HOST'), os.getenv('SMTP_PORT')) as server:
            server.starttls()
            server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        return f"Sent {strategy} email to customer {customer_id}"
    except Exception as e:
        return f"Failed to send email to customer {customer_id}. Error: {e}"

# Model training and prediction
def train_churn_model(data):
    features = ['tenure', 'monthly_spend', 'support_tickets', 'last_interaction_days', 'satisfaction_score']
    X = data[features]
    y = data['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X_test, y_test

# Sentiment analysis
def analyze_sentiment(feedback):
    scores = sid.polarity_scores(feedback)
    return scores['compound']

# Streamlit Dashboard
def main():
    # Password authentication
    password = st.text_input("Enter Password", type="password")
    
    if password != PASSWORD:
        st.error("Incorrect password. Please try again.")
        return  # Prevent access to the rest of the app
    
    st.title("AI-Driven Customer Retention Engine")
    st.sidebar.header("Controls")
    
    # Load or generate data
    if 'data' not in st.session_state:
        st.session_state.data = generate_sample_data()
    
    data = st.session_state.data
    
    # Train model
    if st.sidebar.button("Train Model"):
        model, X_test, y_test = train_churn_model(data)
        st.session_state.model = model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': ['tenure', 'monthly_spend', 'support_tickets', 'last_interaction_days', 'satisfaction_score'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=feature_importance)
        st.pyplot(fig)
    
    # Real-time predictions
    if 'model' in st.session_state:
        st.header("Real-Time Churn Predictions")
        agent = RetentionAgent()
        
        # Segment customers
        data['sentiment'] = data['feedback'].apply(analyze_sentiment)
        data['churn_prob'] = st.session_state.model.predict_proba(data[['tenure', 'monthly_spend', 'support_tickets', 'last_interaction_days', 'satisfaction_score']])[:, 1]
        
        high_risk = data[data['churn_prob'] > 0.7]
        st.write(f"High-Risk Customers: {len(high_risk)}")
        
        # Display high-risk customers
        if not high_risk.empty:
            st.dataframe(high_risk[['customer_id', 'churn_prob', 'sentiment', 'satisfaction_score']])
            
            # Automated retention actions
            for idx, customer in high_risk.iterrows():
                action = agent.choose_action()
                if st.button(f"Execute {action} for Customer {customer['customer_id'][:8]}"):
                    result = send_retention_email(customer['customer_id'], action)
                    reward = 1 if customer['sentiment'] > 0 else -1  # Simplified reward
                    agent.update(action, reward)
                    st.write(result)
        
        # Q-table visualization
        st.header("Retention Strategy Optimization")
        q_df = pd.DataFrame(list(agent.q_table.items()), columns=['Action', 'Q-Value'])
        fig, ax = plt.subplots()
        sns.barplot(x='Q-Value', y='Action', data=q_df)
        st.pyplot(fig)
    
    # Automated model retraining
    if st.sidebar.button("Retrain Model"):
        if os.path.exists('churn_model.pkl'):
            new_data = generate_sample_data(200)  # Simulate new data
            data = pd.concat([data, new_data], ignore_index=True)
            st.session_state.data = data
            model, _, _ = train_churn_model(data)
            st.session_state.model = model
            st.write("Model retrained with new data!")
    
    # Explainability
    if 'model' in st.session_state and st.sidebar.button("Show Explainability"):
        st.header("Model Explainability")
        feature_importance = pd.DataFrame({
            'feature': ['tenure', 'monthly_spend', 'support_tickets', 'last_interaction_days', 'satisfaction_score'],
            'importance': st.session_state.model.feature_importances_
        }).sort_values('importance', ascending=False)
        st.write("Feature Importance for Churn Prediction:")
        st.dataframe(feature_importance)

if __name__ == "__main__":
    main()
