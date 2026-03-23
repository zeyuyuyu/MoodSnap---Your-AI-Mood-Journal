import numpy as np
from transformers import pipeline

class MoodAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline('sentiment-analysis')

    def analyze_mood(self, journal_entry):
        sentiment = self.sentiment_analyzer(journal_entry)[0]
        score = sentiment['score']
        label = sentiment['label']
        return {'score': score, 'label': label}

    def get_mood_trends(self, journal_entries):
        scores = [self.analyze_mood(entry)['score'] for entry in journal_entries]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        return {'mean_score': mean_score, 'std_score': std_score}
