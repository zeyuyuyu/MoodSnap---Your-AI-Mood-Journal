import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class MoodAnalyzer:
    def __init__(self):
        self.mood_scale = {
            'very_negative': 1,
            'negative': 2,
            'neutral': 3, 
            'positive': 4,
            'very_positive': 5
        }

    def calculate_mood_trends(self, mood_entries: List[Dict]) -> Dict:
        """Analyze mood trends and provide statistical insights.
        
        Args:
            mood_entries: List of dictionaries containing mood data
                        [{timestamp: datetime, mood: str, notes: str}, ...]
        
        Returns:
            Dictionary containing trend analysis results
        """
        if not mood_entries:
            return {
                'average_mood': None,
                'trend': None,
                'volatility': None,
                'patterns': None
            }

        # Convert to DataFrame for analysis
        df = pd.DataFrame(mood_entries)
        df['mood_score'] = df['mood'].map(self.mood_scale)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Calculate key metrics
        avg_mood = df['mood_score'].mean()
        
        # Calculate trend using linear regression
        x = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        slope, _ = np.polyfit(x, df['mood_score'], 1)
        
        # Calculate mood volatility
        volatility = df['mood_score'].std()
        
        # Identify patterns
        patterns = self._identify_patterns(df)
        
        return {
            'average_mood': round(avg_mood, 2),
            'trend': {
                'direction': 'improving' if slope > 0.1 else 'declining' if slope < -0.1 else 'stable',
                'slope': round(slope, 4)
            },
            'volatility': round(volatility, 2),
            'patterns': patterns
        }

    def _identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify recurring patterns in mood data."""
        patterns = {
            'weekday_effect': self._analyze_weekday_effect(df),
            'time_of_day': self._analyze_time_of_day(df),
            'streak_analysis': self._analyze_streaks(df)
        }
        return patterns

    def _analyze_weekday_effect(self, df: pd.DataFrame) -> Dict:
        """Analyze how mood varies by day of week."""
        weekday_means = df.groupby(df['timestamp'].dt.day_name())['mood_score'].mean()
        best_day = weekday_means.idxmax()
        worst_day = weekday_means.idxmin()
        
        return {
            'best_day': best_day,
            'worst_day': worst_day,
            'variation': round(weekday_means.max() - weekday_means.min(), 2)
        }

    def _analyze_time_of_day(self, df: pd.DataFrame) -> Dict:
        """Analyze mood patterns throughout the day."""
        df['hour'] = df['timestamp'].dt.hour
        hourly_means = df.groupby('hour')['mood_score'].mean()
        
        return {
            'best_hour': int(hourly_means.idxmax()),
            'worst_hour': int(hourly_means.idxmin()),
            'variation': round(hourly_means.max() - hourly_means.min(), 2)
        }

    def _analyze_streaks(self, df: pd.DataFrame) -> Dict:
        """Analyze positive and negative mood streaks."""
        positive_threshold = 4  # mood_score >= 4 is considered positive
        negative_threshold = 2  # mood_score <= 2 is considered negative
        
        current_positive_streak = 0
        max_positive_streak = 0
        current_negative_streak = 0
        max_negative_streak = 0
        
        for score in df['mood_score']:
            if score >= positive_threshold:
                current_positive_streak += 1
                current_negative_streak = 0
                max_positive_streak = max(max_positive_streak, current_positive_streak)
            elif score <= negative_threshold:
                current_negative_streak += 1
                current_positive_streak = 0
                max_negative_streak = max(max_negative_streak, current_negative_streak)
            else:
                current_positive_streak = 0
                current_negative_streak = 0
        
        return {
            'longest_positive_streak': max_positive_streak,
            'longest_negative_streak': max_negative_streak,
            'current_streak': current_positive_streak if current_positive_streak > 0 else -current_negative_streak
        }

    def generate_insights(self, analysis_results: Dict) -> List[str]:
        """Generate human-readable insights from analysis results."""
        insights = []
        
        if analysis_results['average_mood']:
            insights.append(f"Your average mood is {analysis_results['average_mood']}/5")
            
            trend = analysis_results['trend']['direction']
            insights.append(f"Your mood is {trend} over time")
            
            if analysis_results['volatility'] > 1.5:
                insights.append("Your mood shows significant variation")
            
            patterns = analysis_results['patterns']
            if patterns['weekday_effect']['variation'] > 0.5:
                insights.append(f"You tend to feel best on {patterns['weekday_effect']['best_day']}s")
            
            if patterns['streak_analysis']['current_streak'] > 3:
                insights.append(f"You're on a {patterns['streak_analysis']['current_streak']}-day positive streak!")
            elif patterns['streak_analysis']['current_streak'] < -3:
                insights.append(f"You're having a rough patch lasting {abs(patterns['streak_analysis']['current_streak'])} days")
        
        return insights