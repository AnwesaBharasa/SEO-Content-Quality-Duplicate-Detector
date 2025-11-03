import streamlit as st
import json
from utils.parser import scrape_and_parse
from utils.features import extract_features
from utils.scorer import predict_quality, check_for_duplicates
import nltk

# Download necessary tokenizer models at runtime
nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)  # fix for newer NLTK versions


# --- 1. App Title and Description ---
st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
st.title("🤖 SEO Content Quality & Duplicate Detector")
st.markdown("""
This app analyzes a live URL to assess its SEO quality and check for near-duplicates 
against a pre-computed corpus. This is a demo for the LeadWalnut/CodeWalnut assignment.
""")

# --- 2. URL Input ---
url = st.text_input("Enter a URL to analyze:", placeholder="https://example.com/blog-post")

# --- 3. Analyze Button ---
if st.button("Analyze Content", type="primary"):
    if not url:
        st.warning("Please enter a URL to analyze.")
    else:
        try:
            # --- 4. Analysis Pipeline ---
            with st.spinner("Analyzing... This may take a moment..."):
                
                # Step 1: Scrape and Parse
                st.subheader("1. Parsing Content")
                title, clean_text = scrape_and_parse(url)
                st.success(f"Successfully scraped and parsed content.")
                st.text(f"Page Title: {title}")

                # Step 2: Extract Features
                st.subheader("2. Extracting Features")
                features_df, feature_dict = extract_features(clean_text)
                st.success("Features extracted.")
                
                cols = st.columns(3)
                cols[0].metric("Word Count", feature_dict['word_count'])
                cols[1].metric("Sentence Count", feature_dict['sentence_count'])
                cols[2].metric("Readability (Flesch)", f"{feature_dict['readability']:.2f}")

                # Step 3: Predict Quality
                st.subheader("3. Scoring Quality")
                quality_label = predict_quality(features_df)
                st.success("Content quality scored.")
                
                if quality_label == "High":
                    st.markdown(f"**Predicted Quality: <span style='color:green; font-weight:bold;'>{quality_label}</span>**", unsafe_allow_html=True)
                elif quality_label == "Medium":
                    st.markdown(f"**Predicted Quality: <span style='color:orange; font-weight:bold;'>{quality_label}</span>**", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Predicted Quality: <span style='color:red; font-weight:bold;'>{quality_label}</span>**", unsafe_allow_html=True)
                
                st.info(f"Thin Content (< 500 words): **{feature_dict['is_thin']}**")

                # Step 4: Check Duplicates
                st.subheader("4. Checking for Duplicates")
                similar_to = check_for_duplicates(clean_text, url)
                st.success("Duplicate check complete.")
                
                if not similar_to:
                    st.info("No near-duplicates found in the corpus (similarity > 80%).")
                else:
                    st.warning(f"Found {len(similar_to)} similar page(s):")
                    for item in similar_to:
                        st.markdown(f"- **URL:** `{item['url']}` \n - **Similarity:** `{item['similarity']:.2%}`")

                # Step 5: Show Full Result
                st.subheader("5. Full Analysis Result (JSON)")
                result = {
                    "url": url,
                    "title": title,
                    "quality_label": quality_label,
                    **feature_dict,
                    "similar_to": similar_to
                }
                st.json(json.dumps(result, indent=2))

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

print("File 'streamlit_app/app.py' written.")
