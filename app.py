import streamlit as st
from transformers import pipeline
import pandas as pd

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI
st.title("📊 Sentiment Analysis App")

# User input for multiple sentences
user_input = st.text_area("Enter sentences (one per line):", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentences = user_input.split("\n")  # Split input into multiple sentences
        results = sentiment_analyzer(sentences)  # Analyze sentiment

        # Convert results to DataFrame
        df = pd.DataFrame(results, index=sentences)
        df.index.name = "Text"
        df.rename(columns={"label": "Sentiment", "score": "Confidence"}, inplace=True)

        # Display the results
        st.write("### Sentiment Analysis Results")
        st.dataframe(df)

        # Display results as a progress bar
        for i, row in df.iterrows():
            st.progress(float(row["Confidence"]))

    else:
        st.warning("Please enter some text before analyzing!!")
