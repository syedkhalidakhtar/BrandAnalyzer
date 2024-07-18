# BrandAnalyzer
## Overview
Brand Analyzer is a web application built using Python and Flask for backend functionality, and HTML and CSS for frontend design. It allows users to analyze comments from past buyers, filter products based on their preferences, and access brand rankings that cater to the selected filters.

## Features
- **Sentiment Analysis**: Analyze customer reviews to determine the sentiment (positive or negative).
- **Product Filtering**: Users can filter products based on gender, category, and price range.
- **Brand Ranking**: View rankings of brands based on positive reviews for the selected filters.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brand-analyzer.git
   cd brand-analyzer
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('omw-1.4')
   nltk.download('stopwords')
   ```

4. Ensure you have the dataset `male_female.csv` in the project directory.

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

## Project Structure
```
brand-analyzer/
├── templates/
│   ├── home.html
│   ├── about.html
│   ├── gender.html
│   ├── male.html
│   ├── female.html
│   ├── pricerange.html
│   └── display.html
├── app.py
├── male_female.csv
├── requirements.txt
└── README.md
```

## Code Explanation
### Import Libraries
```python
from flask import jsonify, Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import emoji
import random
import nltk
from nltk.tokenize import word_tokenize
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
```

### Data Preprocessing
- Functions for cleaning and preprocessing the reviews, including replacing URLs, hashtags, emojis, and contractions, converting to lowercase, removing repetitions, and tokenizing and stemming words.

### Sentiment Analysis
- **process_review**: Cleans and processes a review text.
- **fit_tfidf**: Fits a TF-IDF vectorizer on the review corpus.
- **fit_lr**: Fits a Logistic Regression model on training data.
- **predict_review**: Preprocesses, transforms, and predicts the sentiment of a review.

### Product Filtering
- Functions to filter products based on gender, category, and price range.

### Flask Routes
- Various routes to handle page rendering and form submissions for gender selection, product category, price range, and displaying results.

## Contributing
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/feature-name`).
5. Open a pull request.
