import joblib
import re
import string
import streamlit as st

# Load the vectorizer and model
vectorizer = joblib.load('model/vectorizer1.pkl')
model = joblib.load('model/model.pkl')

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

# Streamlit UI
st.title('Fake News Detection')

# Input form
news_article = st.text_area('Enter the news article:')

if st.button('Predict'):
    if news_article:
        # Preprocess the input
        processed_article = wordopt(news_article)

        # Transform the input and predict
        input_features = vectorizer.transform([processed_article])
        prediction = model.predict(input_features)

        # Display result
        output = "Fake News" if prediction[0] == 0 else "Real News"
        st.write(f"This news article is: {output}")
    else:
        st.write("Please enter a news article to predict.")
