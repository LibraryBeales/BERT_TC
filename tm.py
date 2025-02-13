import pandas as pd 
import numpy as np
from bertopic import BERTopic

df = pd.read_csv("scopus.csv", engine='python')

df.fillna("", inplace=True)
df["combined_text"] = df["Title"] + " " + df["Abstract"] + " " + df["Author Keywords"] + " " + df["Index Keywords"]

model = BERTopic(min_topic_size=5, verbose=True) 
docs = df["combined_text"].tolist() 
topics, probabilities = model.fit_transform(docs)

topic_info = model.get_topic_info()
topic_words = {topic: model.get_topic(topic) for topic in topic_info["Topic"].tolist() if topic != -1}  

with open("topics.txt", "w", encoding="utf-8") as f:
    for topic, words in topic_words.items():
        words_scores = ", ".join([f"{word[0]} ({word[1]:.4f})" for word in words])
        f.write(f"Topic {topic}: {words_scores}\n")

print("Topics saved to topics.txt")

topics_fig = model.visualize_topics()
barchart_fig = model.visualize_barchart()

topics_fig.write_html("topics.html")  
barchart_fig.write_html("barchart.html") 

print("Visualizations saved as topics.html and barchart.html")