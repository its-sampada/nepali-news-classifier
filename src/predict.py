
import joblib
from scipy.sparse import hstack


MODEL_PATH = "models/nepali_news_pipeline.joblib"


# Load the saved model components
pipeline = joblib.load(MODEL_PATH)

word_vectorizer = pipeline["word_vectorizer"]
char_vectorizer = pipeline["char_vectorizer"]
classifier = pipeline["classifier"]


def predict_news(text):

    if not text or not text.strip():
        raise ValueError("Please enter a news article.")

    # Convert text into word features
    word_features = word_vectorizer.transform([text])

    # Convert text into character features
    char_features = char_vectorizer.transform([text])

    # Combine both feature matrices
    combined_features = hstack([
        word_features,
        char_features
    ]).tocsr()

    # Predict category
    prediction = classifier.predict(combined_features)[0]

    # Get model scores
    scores = classifier.predict_proba(combined_features)[0]

    # Get top 3 predictions
    top_indices = scores.argsort()[::-1][:3]

    top_predictions = [
        {
            "category": classifier.classes_[i],
            "score": float(scores[i])
        }
        for i in top_indices
    ]

    return {
        "category": prediction,
        "top_predictions": top_predictions
    }
