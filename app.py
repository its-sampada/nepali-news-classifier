
import streamlit as st
import pandas as pd

from src.predict import predict_news


# Page configuration
st.set_page_config(
    page_title="Nepali News Classifier",
    page_icon="🇳🇵",
    layout="centered"
)


# Application heading
st.title("🇳🇵 Nepali News Classifier")

st.write(
    "An ML-powered application that classifies "
    "Nepali news articles into 16 categories."
)

st.divider()


# News input
st.subheader("Enter your news article")

news_text = st.text_area(
    "Paste Nepali news here:",
    height=250,
    placeholder="यहाँ नेपाली समाचार लेख्नुहोस्..."
)


# Prediction
if st.button("Classify News", type="primary"):

    if not news_text.strip():

        st.warning("Please enter a news article.")

    else:

        try:
            result = predict_news(news_text)

            st.success("Classification completed!")

            st.subheader("Predicted Category")

            st.markdown(
                f"## {result['category']}"
            )

            st.divider()

            st.subheader("Top 3 Predictions")

            scores_df = pd.DataFrame(
                result["top_predictions"]
            )

            scores_df["score"] *= 100

            st.bar_chart(
                scores_df.set_index("category")["score"]
            )

            st.caption(
                "Scores represent model-estimated probabilities, "
                "not necessarily calibrated confidence."
            )

            st.divider()

            st.write(
                f"Article length: {len(news_text)} characters"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


st.divider()

st.caption(
    "Built using Python, Scikit-learn, TF-IDF "
    "and Logistic Regression."
)
