import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

# --- 1. Load All Models & Data ONCE ---
# Use @st.cache_resource to load models only once
# This is the key to a fast Streamlit app.

# Define paths relative to this file
# 'utils/' -> 'streamlit_app/' -> 'models/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_FILE = os.path.join(MODELS_DIR, 'quality_model.pkl')
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, 'embeddings.npy')
FEATURES_FILE = os.path.join(MODELS_DIR, 'features.csv')

@st.cache_resource
def load_model():
    """Loads the trained ML model."""
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_sbert_model():
    """Loads the SentenceTransformer model."""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading SBERT model: {e}")
        return None

@st.cache_data
def load_corpus_data():
    """Loads the corpus embeddings and features for duplicate checking."""
    try:
        corpus_embeddings = np.load(EMBEDDINGS_FILE)
        df_corpus = pd.read_csv(FEATURES_FILE)
        corpus_urls = df_corpus['url'].tolist()
        return corpus_embeddings, corpus_urls
    except Exception as e:
        st.error(f"Error loading corpus data: {e}")
        return None, None

# Load everything
model_pipeline = load_model()
sbert_model = load_sbert_model()
corpus_embeddings, corpus_urls = load_corpus_data()

# --- 2. Main Analysis Function ---

def predict_quality(features_df):
    """Predicts quality using the loaded model."""
    if model_pipeline is None:
        raise ValueError("Model not loaded.")
    
    try:
        quality_label = model_pipeline.predict(features_df)[0]
        return quality_label
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

def check_for_duplicates(clean_text, url):
    """Checks for duplicates against the loaded corpus."""
    if sbert_model is None or corpus_embeddings is None:
        raise ValueError("Corpus or SBERT model not loaded.")
        
    try:
        new_embedding = sbert_model.encode([clean_text])
        similarities = cosine_similarity(new_embedding, corpus_embeddings)[0]
        
        SIMILARITY_THRESHOLD = 0.80
        similar_to = []
        for i, score in enumerate(similarities):
            # Don't compare the page to itself if it's in the corpus
            if score > SIMILARITY_THRESHOLD and corpus_urls[i] != url:
                similar_to.append({
                    "url": corpus_urls[i],
                    "similarity": float(score)
                })
        
        return sorted(similar_to, key=lambda x: x['similarity'], reverse=True)
        
    except Exception as e:
        raise ValueError(f"Error checking duplicates: {e}")

print("File 'streamlit_app/utils/scorer.py' written.")
