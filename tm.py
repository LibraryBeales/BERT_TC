#EXTENSIVE COMMENTS ADDED BY CHATGPT


import pandas as pd  # Import pandas for handling CSV files and data manipulation
import numpy as np  # Import numpy (not used explicitly, but commonly used for numerical operations)
from bertopic import BERTopic  # Import BERTopic for topic modeling
import nltk  # Import NLTK for natural language processing tasks
from nltk.corpus import stopwords  # Import stopwords to remove common words that don't contribute to topics

# Download stopwords if they haven't been downloaded before
nltk.download("stopwords")
# Create a set of stopwords for English language
stop_words = set(stopwords.words("english"))

# Load the dataset from a CSV file. 'engine="python"' is used to handle potential parsing issues.
df = pd.read_csv("scopus.csv", engine='python')

# Fill any missing values (NaN) with an empty string to avoid errors when concatenating text columns
df.fillna("", inplace=True)

# Combine multiple text columns into one string per document
# This ensures that the topic model considers the title, abstract, and keywords together
# Each row represents a single document for BERTopic
# Spaces are added between concatenated values to ensure proper token separation
df["combined_text"] = df["Title"] + " " + df["Abstract"] + " " + df["Author Keywords"] + " " + df["Index Keywords"]

# Initialize the BERTopic model with verbose=True to display progress messages during training
model = BERTopic(verbose=True) 

# Define a function to remove stopwords from a given text
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

# Apply the stopword removal function to the combined text column
# This ensures that common words (e.g., "the", "is", "and") are removed before topic modeling
df["clean_text"] = df["combined_text"].apply(remove_stopwords)

# Convert the cleaned text column into a list
# This list of documents will be used as input for the BERTopic model
docs = df["clean_text"].tolist() 

# Fit the BERTopic model to the documents and extract topics
# `topics` contains the assigned topic number for each document
# `probabilities` contains the confidence scores for each topic assignment
topics, probabilities = model.fit_transform(docs)

# Get information about the identified topics
# This includes the topic number, frequency, and top words for each topic
topic_info = model.get_topic_info()

# Extract the most important words for each topic (excluding the outlier topic -1)
# The outlier topic (-1) contains documents that were not assigned to a meaningful topic
topic_words = {topic: model.get_topic(topic) for topic in topic_info["Topic"].tolist() if topic != -1}  

# Save the topics and their top words (with scores) to a text file
with open("topics.txt", "w", encoding="utf-8") as f:
    for topic, words in topic_words.items():
        # Format the words and their relevance scores
        words_scores = ", ".join([f"{word[0]} ({word[1]:.4f})" for word in words])
        f.write(f"Topic {topic}: {words_scores}\n")

print("Topics saved to topics.txt")

# Generate visualizations
# These will help understand the distribution and importance of topics

# Create an interactive visualization of the topics
topics_fig = model.visualize_topics()
# Create a bar chart showing the most important words for each topic
barchart_fig = model.visualize_barchart()

# Save the topic visualization as an interactive HTML file
topics_fig.write_html("topics.html")  
# Save the bar chart visualization as an interactive HTML file
barchart_fig.write_html("barchart.html") 

print("Visualizations saved as topics.html and barchart.html")