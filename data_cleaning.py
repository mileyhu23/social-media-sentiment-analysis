from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

df = pd.read_csv('sentimentdataset.csv')
df['Sentiment'] = df['Sentiment'].str.strip()

analyzer = SentimentIntensityAnalyzer()

def get_compound(label):
    return analyzer.polarity_scores(label)['compound']

def classify(compound):
    if compound >= 0.05:  return 'Positive'
    if compound <= -0.05: return 'Negative'
    return 'Neutral'


df['compound']        = df['Sentiment'].apply(get_compound)
df['Sentiment_Group'] = df['compound'].apply(classify)
