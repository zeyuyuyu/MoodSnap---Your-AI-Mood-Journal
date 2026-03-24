import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class MoodAnalysis:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_mood(self, journal_entry):
        """Analyze the sentiment of a mood journal entry."""
        sentiment_scores = self.sia.polarity_scores(journal_entry)
        if sentiment_scores['compound'] >= 0.05:
            mood = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            mood = 'Negative'
        else:
            mood = 'Neutral'
        return mood

if __name__ == '__main__':
    analyzer = MoodAnalysis()
    journal_entry = "Today was a great day! I felt really happy and energetic."
    mood = analyzer.analyze_mood(journal_entry)
    print(f"The mood for this entry is: {mood}")