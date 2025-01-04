import joblib
import re
import string
import streamlit as st
import time
from datetime import datetime
import numpy as np

# Load the vectorizer and model
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('model/vectorizer1.pkl')
        model = joblib.load('model/model.pkl')
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Enhanced preprocessing function
def wordopt(text):
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    
    # Additional cleaning
    text = re.sub('\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()  # Remove leading/trailing whitespace
    
    return text

# Function to analyze text properties
def analyze_text_properties(text):
    words = text.split()
    sentences = [s.strip() for s in re.split('[.!?]+', text) if s.strip()]
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0
    }

# Function to get model confidence
def get_prediction_confidence(model, features):
    try:
        probabilities = model.predict_proba(features)
        confidence = np.max(probabilities) * 100
        return confidence
    except:
        return 85.0  # Default fallback confidence

# Page config
st.set_page_config(
    page_title="Fake News Detective",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# [Your existing CSS styles remain exactly the same]
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 10px 24px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        transform: translateY(-2px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .fake-news {
        background-color: #000000;
        border: 2px solid #FF4B4B;
    }
    .real-news {
        background-color: #000000;
        border: 2px solid #28A745;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.write("This tool uses machine learning to analyze and detect potential fake news articles.")
    st.markdown("### How to use")
    st.write("1. Paste your article text in the input box")
    st.write("2. Click 'Analyze Article'")
    st.write("3. Wait for the AI to analyze the content")
    
    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main content
st.title("🔍 Fake News Detective")
st.markdown("### AI-Powered News Verification Tool")
st.markdown("---")

# Load models
vectorizer, model = load_models()

if vectorizer is None or model is None:
    st.error("Failed to load models. Please check the model files and paths.")
    st.stop()

# Input form
news_article = st.text_area(
    'Enter the news article for analysis:',
    height=200,
    placeholder="Paste your article text here..."
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_button = st.button('🔍 Analyze Article')

if analyze_button:
    if news_article:
        start_time = time.time()
        
        # Add a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process text
        status_text.text("Processing text...")
        progress_bar.progress(25)
        processed_article = wordopt(news_article)
        time.sleep(0.3)
        
        # Analyze patterns
        status_text.text("Analyzing patterns...")
        progress_bar.progress(50)
        text_properties = analyze_text_properties(processed_article)
        time.sleep(0.3)
        
        # Make prediction
        status_text.text("Making prediction...")
        progress_bar.progress(75)
        input_features = vectorizer.transform([processed_article])
        prediction = model.predict(input_features)
        confidence = get_prediction_confidence(model, input_features)
        time.sleep(0.3)
        
        # Complete analysis
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        processing_time = time.time() - start_time
        time.sleep(0.3)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display result with custom styling
        is_fake = prediction[0] == 0
        result_class = "fake-news" if is_fake else "real-news"
        result_icon = "⚠" if is_fake else "✅"
        result_text = "Potential Fake News" if is_fake else "Likely Real News"
        
        st.markdown(f"""
            <div class="result-box {result_class}">
                <h2>{result_icon} {result_text}</h2>
                
            </div>
        """, unsafe_allow_html=True)
        
        # Additional analysis details
        with st.expander("See Analysis Details"):
            st.markdown("### Content Analysis")
            st.write(f"- Article length: {text_properties['word_count']} words")
            st.write(f"- Number of sentences: {text_properties['sentence_count']}")
            st.write(f"- Average word length: {text_properties['avg_word_length']:.1f} characters")
            st.write(f"- Average sentence length: {text_properties['avg_sentence_length']:.1f} words")
            st.write(f"- Processing time: {processing_time:.2f} seconds")
            
            
    else:
        st.error("Please enter a news article to analyze.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Disclaimer: This tool provides an AI-based assessment and should not be used as the sole factor in determining news authenticity.</p>
    </div>
""", unsafe_allow_html=True)
