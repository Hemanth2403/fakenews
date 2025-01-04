import joblib
from flask import Flask, request, render_template, redirect, url_for
import re
import string

app = Flask(__name__)

# Load the vectorizer and model
vectorizer = joblib.load(r'model/vectorizer1.pkl')
model = joblib.load(r'model/model.pkl')

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Show input form

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    news_article = request.form['news_article']

    # Preprocess the input
    processed_article = wordopt(news_article)

    # Transform the input and predict
    input_features = vectorizer.transform([processed_article])
    prediction = model.predict(input_features)

    # Determine the prediction result
    output = "Fake News" if prediction[0] == 0 else "Real News"

    # Redirect to the result page and pass the prediction
    return redirect(url_for('result', prediction_text=output))

# Result route
@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text', None)
    return render_template('result.html', prediction_text=f"This news article is: {prediction_text}")

if __name__ == '__main__':
    app.run(debug=True)
