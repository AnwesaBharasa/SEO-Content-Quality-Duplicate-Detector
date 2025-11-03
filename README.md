SEO Content Quality & Duplicate Detector

This project analyzes webpage content to detect duplicate pages, flag thin content, and assess SEO content quality using NLP and machine learning.
Developed as part of an AI/ML role assignment, it combines text extraction, embedding similarity, and rule-based + ML classification for automated content evaluation.

🚀 Overview

The system takes web page HTML or URLs as input and performs:

HTML Parsing – Extracts title, main body text, and word count.

Feature Engineering – Calculates readability (Flesch), sentence count, and top TF-IDF keywords.

Embedding Generation – Uses SentenceTransformer (all-MiniLM-L6-v2) for semantic vectorization.

Duplicate Detection – Computes cosine similarity between embeddings to flag near-duplicates.

Quality Labeling – Categorizes each page as High, Medium, or Low quality based on rules and an ML classifier.

Real-Time Analysis – analyze_url() function allows instant evaluation of any new URL.

🧰 Tech Stack

Language: Python 3.9+

Core Environment: Jupyter / Kaggle Notebook

Libraries:
pandas, numpy, beautifulsoup4, lxml, scikit-learn,
sentence-transformers, textstat, nltk, tqdm, joblib, matplotlib

Optional: streamlit for interactive demo

📂 Project Structure
seo-content-quality-duplicate-detector/
├── data/
│   ├── data.csv                      # Input dataset (URL or HTML)
│   ├── extracted_content.csv         # Parsed text data
│   ├── features_with_embeddings.csv  # Features + embeddings
│   ├── duplicates.csv                # Duplicate URL pairs
│   └── final_features.csv            # All engineered features + labels
├── notebooks/
│   └── SEO_Content_Quality_and_Duplicate_Detector.ipynb
├── models/
│   ├── quality_model.pkl
│   └── label_encoder.pkl
├── streamlit_app/ (optional)
│   └── app.py
└── README.md

⚙️ How to Run (Kaggle Setup)

Upload your dataset → /kaggle/input/dataset-for-assignment/data.csv

Upload this notebook: SEO_Content_Quality_and_Duplicate_Detector.ipynb

In Kaggle Notebook Settings → enable Internet + GPU (T4)

Run all cells top-to-bottom.

Output files will be saved in /kaggle/working/seo_outputs/.

📈 Outputs
File	Description
extracted_content.csv	Parsed titles, text, and word counts
features_with_embeddings.csv	Engineered features + embeddings
duplicates.csv	URL pairs with cosine similarity > 0.8
final_features.csv	All computed features + quality_label
quality_model.pkl	Trained RandomForest quality predictor
🧩 Key Insights

High Quality → >1500 words, readability between 50–70

Low Quality → <500 words or readability <30

Duplicates → cosine similarity ≥ 0.8 between embeddings

Thin Content Ratio → pages with <500 words

🌐 Example Test URLs

Use these for demo/testing:

https://www.example.com/
 (short content → Low quality)

https://en.wikipedia.org/wiki/Artificial_intelligence
 (long, detailed → High quality)

https://www.ibm.com/topics/machine-learning
 (medium-length → Medium quality)

🏆 Results Summary

Model Accuracy (RandomForest): ~75–80% on rule-based labels

Duplicate Detection Threshold: 0.80

Thin Pages Flagged: automatically labeled

⚡ Future Enhancements

Integrate topic-based clustering for duplicate group detection

Add grammar/keyword density metrics

Deploy Streamlit dashboard for visual analysis
