from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv("mail_data.csv")

# Replace null values
data = data.where((pd.notnull(data)), '')

# Convert labels properly (WRITE THIS HERE)
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})

X = data['Message']
Y = data['Category']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Convert text to numbers
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_count, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    data = cv.transform([email])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Ham (Not Spam)"
    else:
        result = "Spam"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":

    app.run(debug=True)
