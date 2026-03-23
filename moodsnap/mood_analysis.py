# Mood analysis and visualization module
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from typing import List, Dict, Optional

class MoodAnalyzer:
    def __init__(self):
        self.mood_scale = {
            'Very Happy': 5,
            'Happy': 4,
            'Neutral': 3,
            'Sad': 2,
            'Very Sad': 1
        }

    def calculate_mood_trend(self, entries: List[Dict]) -> pd.DataFrame:
        """Convert mood entries to time series data and calculate trends."""
        if not entries:
            return pd.DataFrame()

        # Convert entries to DataFrame
        df = pd.DataFrame(entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['mood_score'] = df['mood'].map(self.mood_scale)

        # Calculate 7-day rolling average
        df = df.sort_values('timestamp')
        df['mood_trend'] = df['mood_score'].rolling(window=7, min_periods=1).mean()

        return df

    def generate_mood_visualization(self, df: pd.DataFrame) -> dict:
        """Create interactive mood trend visualization using plotly."""
        if df.empty:
            return None

        fig = px.line(df, 
                      x='timestamp', 
                      y=['mood_score', 'mood_trend'],
                      title='Mood Trend Analysis',
                      labels={
                          'timestamp': 'Date',
                          'value': 'Mood Score',
                          'variable': 'Metric'
                      })

        fig.add_scatter(x=df['timestamp'],
                       y=df['mood_score'],
                       mode='markers',
                       name='Daily Mood')

        return fig.to_dict()

    def get_mood_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights from mood data."""
        if df.empty:
            return {'error': 'No mood data available'}

        recent_trend = df.tail(7)['mood_trend'].mean()
        overall_avg = df['mood_score'].mean()

        # Calculate mood stability
        mood_stability = 5 - df['mood_score'].std()

        # Find best and worst days
        best_day = df.loc[df['mood_score'].idxmax()]
        worst_day = df.loc[df['mood_score'].idxmin()]

        return {
            'recent_trend': round(recent_trend, 2),
            'overall_average': round(overall_avg, 2),
            'mood_stability': round(mood_stability, 2),
            'best_day': best_day['timestamp'].strftime('%Y-%m-%d'),
            'worst_day': worst_day['timestamp'].strftime('%Y-%m-%d'),
            'total_entries': len(df)
        }

    def analyze_mood_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in mood variations."""
        if df.empty:
            return {'error': 'No mood data available'}

        # Add day of week
        df['day_of_week'] = df['timestamp'].dt.day_name()

        # Calculate average mood by day of week
        day_averages = df.groupby('day_of_week')['mood_score'].mean().round(2)

        # Find most common moods
        mood_counts = df['mood'].value_counts()

        return {
            'day_averages': day_averages.to_dict(),
            'most_common_mood': mood_counts.index[0],
            'mood_frequency': mood_counts.to_dict()
        }
