import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class MoodAnalyzer:
    def __init__(self):
        self.mood_scale = {
            'very_happy': 5,
            'happy': 4,
            'neutral': 3,
            'sad': 2,
            'very_sad': 1
        }

    def calculate_mood_trends(self, mood_entries: List[Dict]) -> Dict:
        """Analyze mood patterns and generate statistical insights.

        Args:
            mood_entries: List of dictionaries containing mood data
                         with 'timestamp' and 'mood' keys

        Returns:
            Dictionary containing trend analysis results
        """
        if not mood_entries:
            return {'error': 'No mood data available'}

        df = pd.DataFrame(mood_entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['mood_score'] = df['mood'].map(self.mood_scale)

        # Basic statistics
        stats = {
            'average_mood': round(df['mood_score'].mean(), 2),
            'mood_volatility': round(df['mood_score'].std(), 2),
            'total_entries': len(df),
            'time_span_days': (df['timestamp'].max() - df['timestamp'].min()).days
        }

        # Weekly patterns
        weekly_avg = df.groupby(df['timestamp'].dt.dayofweek)['mood_score'].mean()
        stats['weekly_patterns'] = {
            day: round(score, 2) for day, score in weekly_avg.items()
        }

        # Mood stability
        df['mood_change'] = df['mood_score'].diff()
        stats['mood_stability'] = {
            'rapid_changes': len(df[abs(df['mood_change']) >= 2]),
            'gradual_changes': len(df[abs(df['mood_change']).between(1, 2)])
        }

        # Trend detection
        if len(df) >= 7:
            recent_trend = df.tail(7)['mood_score'].values
            stats['weekly_trend'] = self._detect_trend(recent_trend)

        return stats

    def _detect_trend(self, values: np.ndarray) -> str:
        """Detect the trend direction in a series of mood scores."""
        slope, _ = np.polyfit(range(len(values)), values, 1)
        if abs(slope) < 0.1:
            return 'stable'
        return 'improving' if slope > 0 else 'declining'

    def get_mood_suggestions(self, recent_moods: List[str]) -> List[str]:
        """Generate personalized suggestions based on recent mood patterns."""
        if not recent_moods:
            return ['Start tracking your mood regularly to get personalized suggestions']

        suggestions = []
        avg_score = sum(self.mood_scale[mood] for mood in recent_moods) / len(recent_moods)

        if avg_score <= 2.5:
            suggestions.extend([
                'Consider talking to a trusted friend or professional',
                'Try incorporating short walks or light exercise',
                'Practice mindfulness or meditation'
            ])
        elif avg_score <= 3.5:
            suggestions.extend([
                'Maintain a regular sleep schedule',
                'Set small, achievable goals for the day',
                'Connect with friends or family'
            ])
        else:
            suggestions.extend([
                'Keep up your positive routines',
                'Share your good energy with others',
                'Document what\'s working well for you'
            ])

        return suggestions

    def generate_mood_report(self, 
                           mood_entries: List[Dict],
                           days: int = 30) -> Dict:
        """Generate a comprehensive mood report for a specified time period."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        filtered_entries = [
            entry for entry in mood_entries
            if start_date <= datetime.fromisoformat(entry['timestamp']) <= end_date
        ]

        trends = self.calculate_mood_trends(filtered_entries)
        recent_moods = [entry['mood'] for entry in filtered_entries[-7:]]
        suggestions = self.get_mood_suggestions(recent_moods)

        return {
            'period': f'Last {days} days',
            'trend_analysis': trends,
            'suggestions': suggestions,
            'data_points': len(filtered_entries)
        }
