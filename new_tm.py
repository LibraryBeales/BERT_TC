import os
import numpy as np
import random
import pandas as pd
import nltk
from bertopic import BERTopic
from nltk.corpus import stopwords
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update(["information", "system", "implementation", "systems"])

# Prompt for output directory
output_dir = input("Enter the directory where you want to save the files: ")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("scopus.csv", engine='python')
df.fillna("", inplace=True)

df["Publication_Period"] = df["Year"].astype(str)  # Use Year as the time period
dates = df["Publication_Period"].tolist()

df["combined_text"] = df["Title"] + " " + df["Abstract"] + " " + df["Author Keywords"] + " " + df["Index Keywords"]

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

# Could use month or quarter as granularity if we had complete dates.
# df["Publication_Period"] = df["Publication_Date"].dt.to_period("M")  # Monthly granularity
# df["Publication_Period"] = df["Publication_Date"].dt.to_period("Q")  # Quarterly granularity
# df["Publication_Period"] = df["Publication_Date"].dt.to_period("Y")  # Yearly granularity


df["clean_text"] = df["combined_text"].apply(remove_stopwords)
docs = df["clean_text"].tolist()

seed = random.randint(0, 2**32 - 1)
random.seed(seed)
np.random.seed(seed)

with open(os.path.join(output_dir, "seed.txt"), "w") as f:
    f.write(f"Random Seed: {seed}\n")
print(f"Random Seed used: {seed}")

hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

vectorizer_model = CountVectorizer(stop_words="english", max_features=10000)

umap_model = UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=seed)

model = BERTopic(
    verbose=True,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    nr_topics=12
)

topics, probabilities = model.fit_transform(docs)

topics_fig = model.visualize_topics()
topics_fig.write_html(os.path.join(output_dir, "topics.html"))

barchart_fig = model.visualize_barchart()
barchart_fig.write_html(os.path.join(output_dir, "barchart.html"))

hierarchical_fig = model.visualize_hierarchy()
hierarchical_fig.write_html(os.path.join(output_dir, "hierarchical.html"))

term_freq_fig = model.visualize_term_rank()
term_freq_fig.write_html(os.path.join(output_dir, "term_frequency.html"))

topics_over_time = model.topics_over_time(docs, dates)
topics_over_time.to_csv(os.path.join(output_dir, "topics_over_time.csv"), index=False)
topics_over_time_fig = model.visualize_topics_over_time(topics_over_time)
topics_over_time_fig.write_html(os.path.join(output_dir, "topics_over_time.html"))

if isinstance(probabilities, list) and isinstance(probabilities[0], (list, np.ndarray)):
    num_topics = len(probabilities[0])
    topic_scores_df = pd.DataFrame(probabilities, columns=[f"Topic_{i}" for i in range(num_topics)])
else:
    topic_scores_df = pd.DataFrame({"Dominant_Topic_Probability": probabilities})

df = pd.concat([df, topic_scores_df], axis=1)
df.to_csv(os.path.join(output_dir, "topic_scores.csv"), index=False)

topic_info = model.get_topic_info()
topic_words = {topic: model.get_topic(topic) for topic in topic_info["Topic"].tolist() if topic != -1}
topic_words_df = pd.DataFrame([{"Topic": topic, "Words": ", ".join([f"{word[0]} ({word[1]:.4f})" for word in words])} for topic, words in topic_words.items()])
topic_words_df.to_csv(os.path.join(output_dir, "topics.csv"), index=False)
