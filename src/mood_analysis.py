import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MoodAnalyzer:
    def __init__(self):
        self.mood_data = pd.DataFrame(columns=['date', 'mood_score', 'journal_text', 'music_genre'])
    
    def add_entry(self, journal_text: str, music_genre: str) -> None:
        """Add a new mood entry with automatic sentiment analysis"""
        sentiment = TextBlob(journal_text).sentiment.polarity
        new_entry = {
            'date': datetime.now(),
            'mood_score': sentiment,
            'journal_text': journal_text,
            'music_genre': music_genre
        }
        self.mood_data = pd.concat([self.mood_data, pd.DataFrame([new_entry])], ignore_index=True)
    
    def get_mood_trends(self, days: int = 30) -> dict:
        """Analyze mood trends over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = self.mood_data[self.mood_data['date'] >= cutoff_date]
        
        return {
            'average_mood': recent_data['mood_score'].mean(),
            'mood_volatility': recent_data['mood_score'].std(),
            'top_genres': recent_data['music_genre'].value_counts().head(3).to_dict()
        }
    
    def visualize_mood_timeline(self, days: int = 30) -> None:
        """Generate mood timeline visualization"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = self.mood_data[self.mood_data['date'] >= cutoff_date]
        
        plt.figure(figsize=(12, 6))
        plt.plot(recent_data['date'], recent_data['mood_score'], marker='o')
        plt.title('Mood Timeline')
        plt.xlabel('Date')
        plt.ylabel('Mood Score')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('mood_timeline.png')
        plt.close()
    
    def get_genre_mood_correlation(self) -> dict:
        """Analyze correlation between music genres and mood"""
        genre_moods = {}
        for genre in self.mood_data['music_genre'].unique():
            genre_data = self.mood_data[self.mood_data['music_genre'] == genre]
            genre_moods[genre] = {
                'avg_mood': genre_data['mood_score'].mean(),
                'count': len(genre_data)
            }
        return genre_moods
    
    def export_data(self, filepath: str) -> None:
        """Export mood data to CSV"""
        self.mood_data.to_csv(filepath, index=False)
    
    def import_data(self, filepath: str) -> None:
        """Import mood data from CSV"""
        self.mood_data = pd.read_csv(filepath)
        self.mood_data['date'] = pd.to_datetime(self.mood_data['date'])
