# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Text processing function
def text_processing(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')])
    return text

# Main function to run the project
def main():
    # Load comments from CSV
    input_csv = 'comments.csv'  # Change this to the path of your comments CSV
    data = pd.read_csv(input_csv)

    print("Comments DataFrame structure:\n", data.head())  # Display the first few comments

    # Check if 'body' or equivalent exists
    if 'body' not in data.columns:
        print("Available columns:", data.columns)
        print("Error: 'body' column not found. Please check the structure of the comments.")
        return
    
    # Sentiment analysis
    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["body"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["body"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["body"]]
    data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["body"]]

    # Classify sentiment
    score = data["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05:
            sentiment.append('Positive')
        elif i <= -0.05:
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')

    data["Sentiment"] = sentiment

    # Apply text processing
    data["Processed_Text"] = data["body"].apply(text_processing)

    # Create sentiment classes
    df_neutral = data[data['Sentiment'] == 'Neutral']
    df_negative = data[data['Sentiment'] == 'Negative']
    df_positive = data[data['Sentiment'] == 'Positive']

    # Upsample minority classes
    df_negative_upsampled = resample(df_negative, replace=True, n_samples=100, random_state=42)
    df_neutral_upsampled = resample(df_neutral, replace=True, n_samples=100, random_state=42)
    df_positive_upsampled = resample(df_positive, replace=True, n_samples=100, random_state=42)

    # Concatenate the upsampled dataframes
    final_data = pd.concat([df_negative_upsampled, df_neutral_upsampled, df_positive_upsampled])

    # Prepare features and labels
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(final_data["Processed_Text"]).toarray()
    y = final_data["Sentiment"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Confusion matrix and accuracy
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    nb_score = accuracy_score(y_test, y_pred)
    print('Accuracy:', nb_score * 100)

    # Result in the form of a pie chart
    labels = ["Positive", "Negative", "Neutral"]
    x = [len(df_positive), len(df_negative), len(df_neutral)]
    myexplode = [0.2, 0.2, 0]
    colors = ['orange', 'darkblue', 'red']

    plt.pie(x, labels=labels, colors=colors, explode=myexplode, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

    # Save to CSV
    output_csv = input("Output CSV file name: ")
    data.to_csv(output_csv, index=False)
    print(f"Comments and sentiments saved to {output_csv}")

if __name__ == "__main__":
    main()
