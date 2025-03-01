

import pandas as pd
import nltk
from bertopic import BERTopic
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update(["information", "system", "implementation", "systems"])

df = pd.read_csv("scopus.csv", engine='python')
df.fillna("", inplace=True)

df["Publication_Period"] = df["Year"].astype(str)  # Use Year as the time period
dates = df["Publication_Period"].tolist()

df["combined_text"] = df["Title"] + " " + df["Abstract"] + " " + df["Author Keywords"] + " " + df["Index Keywords"]

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

#Could use month or quarter as granularity if we had complete dates.
# df["Publication_Period"] = df["Publication_Date"].dt.to_period("M")  # Monthly granularity
# df["Publication_Period"] = df["Publication_Date"].dt.to_period("Q")  # Quarterly granularity
# df["Publication_Period"] = df["Publication_Date"].dt.to_period("Y")  # Yearly granularity


df["clean_text"] = df["combined_text"].apply(remove_stopwords)
docs = df["clean_text"].tolist()

model = BERTopic(verbose=True, nr_topics=8) 

topics, probabilities = model.fit_transform(docs)
topics_over_time = model.topics_over_time(docs, dates)
topics_over_time.to_csv("topics_over_time.csv", index=False)
topics_over_time_fig = model.visualize_topics_over_time(topics_over_time)
topics_over_time_fig.write_html("topics_over_time.html")

if isinstance(probabilities, list) and isinstance(probabilities[0], (list, np.ndarray)):
    num_topics = len(probabilities[0])  
    topic_scores_df = pd.DataFrame(probabilities, columns=[f"Topic_{i}" for i in range(num_topics)])
else:
    topic_scores_df = pd.DataFrame({"Dominant_Topic_Probability": probabilities})  # Fallback for single probability case
df = pd.concat([df, topic_scores_df], axis=1)
df.to_csv("document_topic_scores.csv", index=False)

topic_info = model.get_topic_info()
topic_words = {topic: model.get_topic(topic) for topic in topic_info["Topic"].tolist() if topic != -1}
topic_words_df = pd.DataFrame([{"Topic": topic, "Words": ", ".join([f"{word[0]} ({word[1]:.4f})" for word in words])} for topic, words in topic_words.items()])
topic_words_df.to_csv("new_topics.csv", index=False)

print("Topic evolution saved as 'topics_over_time.html'")
print("Document topic scores saved as 'document_topic_scores.csv'")
print("Topics and their words saved as 'new_topics.csv'")
