import nltk
import textstat
import pandas as pd

# Download NLTK data (Streamlit Cloud needs this)
nltk.download('punkt', quiet=True)

def extract_features(clean_text):
    """
    Extracts word count, sentence count, readability, and thin content flag.
    """
    word_count = len(clean_text.split())
    sentence_count = len(nltk.sent_tokenize(clean_text))
    readability = textstat.flesch_reading_ease(clean_text)
    is_thin = word_count < 500
    
    # Create a DataFrame for the model
    features_df = pd.DataFrame([[word_count, sentence_count, readability]], 
                               columns=['word_count', 'sentence_count', 'flesch_reading_ease'])
    
    feature_dict = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "readability": round(readability, 2),
        "is_thin": is_thin
    }
    
    return features_df, feature_dict

print("File 'streamlit_app/utils/features.py' written.")
