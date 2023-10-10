import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Generate a larger dataset with a more balanced distribution
data = pd.DataFrame({
    'url': ['http://legit{}.com'.format(i) for i in range(1000)] + ['https://phishing{}.com'.format(i) for i in range(1000)],
    'URL_length': [20] * 2000,  # Hypothetical URL length
    'use_of_special_characters': [0] * 1000 + [1] * 1000,
    'is_https': [1] * 1000 + [0] * 1000,
    'domain_age': [5] * 2000,  # Hypothetical domain age
    'label': [0] * 1000 + [1] * 1000
})

# Preprocess the 'url' column using label encoding
le = LabelEncoder()
data['url'] = le.fit_transform(data['url'])

# Features and labels
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print('Confusion Matrix:\n', confusion_matrix(y_test, predictions))
print('Classification Report:\n', classification_report(y_test, predictions))

# Save the model to a file
joblib.dump(model, 'phishing_model.joblib')
