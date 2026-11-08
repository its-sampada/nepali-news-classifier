# 🇳🇵 Nepali News Classifier

# Nepali News Topic Classifier

🚀 **Live Demo:** [Try the Nepali News Classifier](https://nepali-news-classifier-npo3slpxctyfyd9u642gno.streamlit.app/)

A machine learning application that automatically classifies Nepali news articles into 16 different categories using Natural Language Processing (NLP) and Scikit-learn.

The project explores different text representation techniques and classification algorithms, with an interactive Streamlit application for making predictions.

## Features

- Classifies Nepali news articles into 16 categories.
- Word-level and character-level TF-IDF feature extraction.
- Multiple machine learning models compared.
- Hyperparameter tuning and model evaluation.
- Top-three category predictions.
- Interactive Streamlit web application.

## Categories

National News, Sports, Economy, Entertainment, Opinion, Education, Tourism, Bank, Interview, Employment, World, Literature, Blog, Technology, Business Interview and Auto.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- SciPy
- Joblib
- Streamlit
- Google Colab

## Machine Learning Workflow

1. Dataset loading and exploration.
2. Data cleaning and text preprocessing.
3. Exploratory Data Analysis (EDA).
4. Class distribution and imbalance analysis.
5. Word-level TF-IDF feature extraction.
6. Character-level TF-IDF feature extraction.
7. Feature combination using sparse matrices.
8. Model training and comparison.
9. Hyperparameter tuning.
10. Error analysis and final evaluation.
11. Model serialization using Joblib.

## Model Experiments

Three classification algorithms were evaluated:

| Model | Features | Accuracy | Macro F1 |
|---|---|---:|---:|
| Naive Bayes | Original TF-IDF | 58.82% | 0.093 |
| Logistic Regression | Word TF-IDF | 76.66% | 0.672 |
| Linear SVM | Word TF-IDF | 82.37% | 0.619 |
| Logistic Regression | Character TF-IDF | 76.59% | 0.626 |
| Logistic Regression | Combined TF-IDF | 78.89% | 0.663 |

After hyperparameter tuning, the selected model was:

**Combined TF-IDF + Logistic Regression**

- Word-level TF-IDF features: 50,000
- Character-level TF-IDF features: 50,000
- Regularization parameter (C): 5.0
- Class weights: balanced

## Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 82.02% |
| Macro F1 | 0.708 |
| Weighted F1 | 0.823 |

The model was evaluated on a separate test set of 1,435 articles.

## Project Structure

```text
nepali-news-classifier/
│
├── models/
│   └── nepali_news_pipeline.joblib
│
├── notebooks/
│   └── nepali_news_classification.ipynb
│
├── src/
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/its-sampada/nepali-news-classifier.git
cd nepali-news-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Model Persistence

The trained model components are saved using Joblib:

- Word TF-IDF vectorizer
- Character TF-IDF vectorizer
- Logistic Regression classifier

This allows predictions without retraining the model.

## Limitations

- The dataset has an imbalanced class distribution.
- Some categories have very few training examples.
- Similar articles can have inconsistent labels.
- Predictions may be less reliable for underrepresented categories.
- Model-estimated probabilities are not necessarily calibrated confidence scores.

## Future Improvements

- Improve data quality and investigate inconsistent labels.
- Experiment with Nepali-specific transformer models.
- Increase representation of minority categories.
- Improve prediction reliability.
- Explore hierarchical news classification.

## Author

**Sampada Dahal**

GitHub: [its-sampada](https://github.com/its-sampada)

---

*Built as a hands-on machine learning and NLP project to explore text classification for the Nepali language.*