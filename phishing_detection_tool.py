import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import joblib

# Load the trained machine learning model
#model_file = 'phishing_model.joblib'


def load_model(phishing_model):
    return joblib.load(phishing_model)


# Predict whether the URL is phishing or not
def predict_url(url, model):
    # Feature extraction (replace with your feature extraction logic)
    features = [0.1, 0.2, 0.3, 0.4]  # Placeholder features
    prediction = model.predict([features])
    confidence = model.predict_proba([features])[0][1]  # Probability of being phishing
    return prediction[0], confidence


# Send email notification
def send_email(recipient, subject, content):
    # Email configurations (replace with your SMTP server details)
    smtp_server = 'your_smtp_server'
    smtp_port = 587
    sender_email = 'your_email'
    sender_password = 'your_password'

    # Create the email content
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient
    message['Subject'] = subject
    message.attach(MIMEText(content, 'plain'))

    # Connect to the SMTP server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, message.as_string())


if __name__ == "__main__":
    # Load the trained model
    #model_file = 'trained_model.joblib'
    model_file = 'phishing_model.joblib'

    model = load_model(model_file)

    # Get the URL from the user
    url = input("Enter the URL to check for phishing: ")

    # Predict whether the URL is phishing
    prediction, confidence = predict_url(url, model)

    # Display the prediction and confidence
    if prediction == 1:
        print("This URL is likely a phishing attempt.")
    else:
        print("This URL appears to be legitimate.")

    print(f"Model confidence: {confidence * 100:.2f}%")

    # Ask if the user wants to receive an email notification
    send_email_flag = input("Do you want to receive an email notification? (yes/no): ").lower()
    if send_email_flag == 'yes':
        recipient = input("Enter your email address: ")
        send_email(recipient, "Phishing URL Detection",
                   f"URL: {url}\nPrediction: {prediction}\nConfidence: {confidence * 100:.2f}%")
        print("Email notification sent.")
