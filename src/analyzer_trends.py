"""
Trends analysis module for analyzing YouTube trending content and patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter, defaultdict
import re
from datetime import datetime, timedelta
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class TrendsAnalyzer:
    """Analyze YouTube trending content and identify patterns."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.trends_data = {}
        self.historical_trends = []
    
    def analyze_trending_content(self, trending_videos: List[Dict], category: str = "all") -> Dict[str, Any]:
        """
        Analyze current trending videos to identify patterns and insights.
        
        Args:
            trending_videos: List of trending video dictionaries
            category: Trending category (all, music, gaming, movies, news)
            
        Returns:
            Dictionary with trending analysis results
        """
        if not trending_videos:
            return {"error": "No trending videos data provided"}
        
        logger.info(f"Analyzing {len(trending_videos)} trending videos in category: {category}")
        
        # Process the data
        videos_df = self.data_processor.process_videos(trending_videos)
        
        analysis = {
            'category': category,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overview': self._generate_trending_overview(videos_df),
            'content_patterns': self._analyze_content_patterns(videos_df),
            'channel_analysis': self._analyze_trending_channels(videos_df),
            'title_trends': self._analyze_title_trends(videos_df),
            'duration_trends': self._analyze_duration_trends(videos_df),
            'viral_factors': self._identify_viral_factors(videos_df),
            'competitive_landscape': self._analyze_competitive_landscape(videos_df),
            'predictions': self._generate_trend_predictions(videos_df),
            'recommendations': self._generate_trending_recommendations(videos_df)
        }
        
        # Store for historical analysis
        self.trends_data[f"{category}_{datetime.now().strftime('%Y%m%d')}"] = analysis
        self.historical_trends.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'category': category,
            'video_count': len(trending_videos),
            'avg_views': videos_df['views'].mean(),
            'dominant_channels': videos_df['channel_name'].value_counts().head(5).to_dict()
        })
        
        return analysis
    
    def compare_trending_categories(self, categories_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Compare trending content across different categories.
        
        Args:
            categories_data: Dictionary with category names as keys and trending videos as values
            
        Returns:
            Dictionary with category comparison results
        """
        logger.info(f"Comparing trending content across {len(categories_data)} categories")
        
        if len(categories_data) < 2:
            return {"error": "At least 2 categories required for comparison"}
        
        # Analyze each category individually
        category_analyses = {}
        for category, videos in categories_data.items():
            category_analyses[category] = self.analyze_trending_content(videos, category)
        
        # Perform comparison
        comparison = {
            'comparison_overview': {
                'categories_compared': list(categories_data.keys()),
                'total_videos_analyzed': sum(len(videos) for videos in categories_data.values()),
                'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_comparison': self._compare_category_performance(category_analyses),
            'content_strategy_comparison': self._compare_content_strategies(category_analyses),
            'audience_preferences': self._analyze_audience_preferences(category_analyses),
            'cross_category_insights': self._generate_cross_category_insights(category_analyses),
            'recommendations': self._generate_category_recommendations(category_analyses)
        }
        
        return comparison
    
    def track_trending_evolution(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Track how trending content evolves over time.
        
        Args:
            historical_data: List of historical trending data snapshots
            
        Returns:
            Dictionary with evolution analysis
        """
        logger.info(f"Analyzing trending evolution across {len(historical_data)} time periods")
        
        if len(historical_data) < 2:
            return {"error": "At least 2 time periods required for evolution analysis"}
        
        evolution = {
            'time_period': {
                'start_date': historical_data[0].get('analysis_date', 'Unknown'),
                'end_date': historical_data[-1].get('analysis_date', 'Unknown'),
                'periods_analyzed': len(historical_data)
            },
            'content_evolution': self._analyze_content_evolution(historical_data),
            'channel_evolution': self._analyze_channel_evolution(historical_data),
            'performance_evolution': self._analyze_performance_evolution(historical_data),
            'trend_lifecycle': self._analyze_trend_lifecycle(historical_data),
            'emerging_patterns': self._identify_emerging_patterns(historical_data),
            'predictions': self._predict_future_trends(historical_data)
        }
        
        return evolution
    
    def _generate_trending_overview(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overview of trending content."""
        overview = {
            'total_trending_videos': len(videos_df),
            'unique_channels': videos_df['channel_name'].nunique(),
            'total_views': int(videos_df['views'].sum()),
            'average_views': int(videos_df['views'].mean()),
            'median_views': int(videos_df['views'].median()),
            'view_range': {
                'min_views': int(videos_df['views'].min()),
                'max_views': int(videos_df['views'].max())
            },
            'top_performing_video': {
                'title': videos_df.loc[videos_df['views'].idxmax(), 'title'],
                'channel': videos_df.loc[videos_df['views'].idxmax(), 'channel_name'],
                'views': int(videos_df['views'].max())
            },
            'channel_dominance': self._calculate_channel_dominance(videos_df)
        }
        
        return overview
    
    def _analyze_content_patterns(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in trending content."""
        patterns = {
            'duration_patterns': {
                'average_duration_minutes': round(videos_df['duration_seconds'].mean() / 60, 2),
                'duration_distribution': videos_df['duration_category'].value_counts().to_dict(),
                'optimal_trending_duration': videos_df.groupby('duration_category')['views'].mean().idxmax()
            },
            'upload_timing': self._analyze_upload_timing_patterns(videos_df),
            'content_freshness': self._analyze_content_freshness(videos_df),
            'viral_velocity': self._calculate_viral_velocity(videos_df)
        }
        
        return patterns
    
    def _analyze_trending_channels(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which channels dominate trending."""
        channel_stats = videos_df.groupby('channel_name').agg({
            'views': ['count', 'sum', 'mean'],
            'duration_seconds': 'mean'
        }).round(2)
        
        channel_stats.columns = ['video_count', 'total_views', 'avg_views', 'avg_duration']
        channel_stats = channel_stats.reset_index()
        
        # Calculate trending score (combination of frequency and performance)
        channel_stats['trending_score'] = (
            channel_stats['video_count'] * 0.4 + 
            (channel_stats['avg_views'] / channel_stats['avg_views'].max()) * 100 * 0.6
        )
        
        top_channels = channel_stats.nlargest(10, 'trending_score')
        
        channel_analysis = {
            'total_unique_channels': len(channel_stats),
            'top_trending_channels': top_channels.to_dict('records'),
            'channel_concentration': {
                'top_5_channels_share': round((top_channels.head(5)['video_count'].sum() / len(videos_df)) * 100, 1),
                'top_10_channels_share': round((top_channels.head(10)['video_count'].sum() / len(videos_df)) * 100, 1)
            },
            'new_vs_established': self._analyze_new_vs_established_channels(videos_df),
            'channel_diversity_score': self._calculate_channel_diversity(videos_df)
        }
        
        return channel_analysis
    
    def _analyze_title_trends(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trending title patterns."""
        # Extract common words and phrases
        all_titles = ' '.join(videos_df['title'].astype(str))
        words = re.findall(r'\b\w+\b', all_titles.lower())
        
        # Filter meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        word_counts = Counter(meaningful_words)
        
        # Analyze title characteristics
        title_trends = {
            'trending_keywords': dict(word_counts.most_common(20)),
            'title_characteristics': {
                'average_length': round(videos_df['title_length'].mean(), 1),
                'titles_with_numbers': int(videos_df['has_numbers_in_title'].sum()),
                'titles_with_caps': int(videos_df['has_caps_in_title'].sum()),
                'percentage_with_numbers': round((videos_df['has_numbers_in_title'].sum() / len(videos_df)) * 100, 1),
                'percentage_with_caps': round((videos_df['has_caps_in_title'].sum() / len(videos_df)) * 100, 1)
            },
            'clickbait_analysis': self._analyze_clickbait_in_trending(videos_df),
            'emotional_triggers': self._identify_emotional_triggers(videos_df),
            'trending_title_formulas': self._identify_title_formulas(videos_df)
        }
        
        return title_trends
    
    def _analyze_duration_trends(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duration trends in trending content."""
        duration_analysis = {
            'duration_statistics': {
                'mean_duration_minutes': round(videos_df['duration_seconds'].mean() / 60, 2),
                'median_duration_minutes': round(videos_df['duration_seconds'].median() / 60, 2),
                'std_duration_minutes': round(videos_df['duration_seconds'].std() / 60, 2)
            },
            'duration_distribution': videos_df['duration_category'].value_counts().to_dict(),
            'performance_by_duration': videos_df.groupby('duration_category')['views'].agg(['mean', 'count']).to_dict(),
            'optimal_duration_insights': self._find_optimal_trending_duration(videos_df),
            'duration_vs_performance_correlation': round(videos_df['duration_seconds'].corr(videos_df['views']), 3)
        }
        
        return duration_analysis
    
    def _identify_viral_factors(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify factors that contribute to viral success."""
        # Define viral threshold (top 20% of views)
        viral_threshold = videos_df['views'].quantile(0.8)
        viral_videos = videos_df[videos_df['views'] >= viral_threshold]
        non_viral_videos = videos_df[videos_df['views'] < viral_threshold]
        
        viral_factors = {
            'viral_threshold_views': int(viral_threshold),
            'viral_video_count': len(viral_videos),
            'viral_characteristics': {
                'avg_title_length': round(viral_videos['title_length'].mean(), 1),
                'avg_duration_minutes': round(viral_videos['duration_seconds'].mean() / 60, 2),
                'numbers_in_title_rate': round((viral_videos['has_numbers_in_title'].sum() / len(viral_videos)) * 100, 1),
                'caps_in_title_rate': round((viral_videos['has_caps_in_title'].sum() / len(viral_videos)) * 100, 1)
            },
            'non_viral_characteristics': {
                'avg_title_length': round(non_viral_videos['title_length'].mean(), 1),
                'avg_duration_minutes': round(non_viral_videos['duration_seconds'].mean() / 60, 2),
                'numbers_in_title_rate': round((non_viral_videos['has_numbers_in_title'].sum() / len(non_viral_videos)) * 100, 1),
                'caps_in_title_rate': round((non_viral_videos['has_caps_in_title'].sum() / len(non_viral_videos)) * 100, 1)
            },
            'viral_success_factors': self._calculate_viral_success_factors(viral_videos, non_viral_videos),
            'viral_channels': viral_videos['channel_name'].value_counts().head(5).to_dict()
        }
        
        return viral_factors
    
    def _analyze_competitive_landscape(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the competitive landscape of trending content."""
        # Group similar content by keywords
        content_clusters = self._cluster_content_by_similarity(videos_df)
        
        competitive_analysis = {
            'content_clusters': content_clusters,
            'market_saturation': self._calculate_market_saturation(videos_df),
            'niche_opportunities': self._identify_niche_opportunities(videos_df),
            'competitive_intensity': self._measure_competitive_intensity(videos_df),
            'barrier_to_entry': self._assess_barrier_to_entry(videos_df)
        }
        
        return competitive_analysis
    
    def _generate_trend_predictions(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions about future trends."""
        predictions = {
            'emerging_topics': self._identify_emerging_topics(videos_df),
            'declining_topics': self._identify_declining_topics(videos_df),
            'content_format_predictions': self._predict_content_formats(videos_df),
            'channel_growth_predictions': self._predict_channel_growth(videos_df),
            'seasonal_patterns': self._identify_seasonal_patterns(videos_df)
        }
        
        return predictions
    
    def _calculate_channel_dominance(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate how dominated the trending list is by top channels."""
        channel_counts = videos_df['channel_name'].value_counts()
        total_videos = len(videos_df)
        
        dominance = {
            'top_channel_share': round((channel_counts.iloc[0] / total_videos) * 100, 1),
            'top_3_channels_share': round((channel_counts.head(3).sum() / total_videos) * 100, 1),
            'top_5_channels_share': round((channel_counts.head(5).sum() / total_videos) * 100, 1),
            'herfindahl_index': round(sum((count / total_videos) ** 2 for count in channel_counts), 3),
            'dominance_level': 'High' if channel_counts.iloc[0] / total_videos > 0.2 else 'Moderate' if channel_counts.iloc[0] / total_videos > 0.1 else 'Low'
        }
        
        return dominance
    
    def _analyze_upload_timing_patterns(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when trending videos were uploaded."""
        timing_patterns = {
            'timing_data_available': 'upload_date' in videos_df.columns
        }
        
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df['days_since_upload'] = (datetime.now() - videos_df['upload_date']).dt.days
                
                timing_patterns.update({
                    'average_days_to_trend': round(videos_df['days_since_upload'].mean(), 1),
                    'median_days_to_trend': round(videos_df['days_since_upload'].median(), 1),
                    'fastest_to_trend': int(videos_df['days_since_upload'].min()),
                    'slowest_to_trend': int(videos_df['days_since_upload'].max()),
                    'trending_speed_distribution': {
                        'same_day': len(videos_df[videos_df['days_since_upload'] == 0]),
                        'within_week': len(videos_df[videos_df['days_since_upload'] <= 7]),
                        'within_month': len(videos_df[videos_df['days_since_upload'] <= 30])
                    }
                })
            except Exception as e:
                logger.warning(f"Could not analyze upload timing: {e}")
                timing_patterns['error'] = str(e)
        
        return timing_patterns
    
    def _analyze_content_freshness(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how fresh the trending content is."""
        freshness = {
            'freshness_data_available': 'upload_date' in videos_df.columns
        }
        
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df['days_old'] = (datetime.now() - videos_df['upload_date']).dt.days
                
                freshness.update({
                    'average_content_age_days': round(videos_df['days_old'].mean(), 1),
                    'fresh_content_percentage': round((len(videos_df[videos_df['days_old'] <= 7]) / len(videos_df)) * 100, 1),
                    'evergreen_content_percentage': round((len(videos_df[videos_df['days_old'] > 30]) / len(videos_df)) * 100, 1),
                    'freshness_score': self._calculate_freshness_score(videos_df['days_old'])
                })
            except Exception as e:
                logger.warning(f"Could not analyze content freshness: {e}")
        
        return freshness
    
    def _calculate_viral_velocity(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate how quickly content goes viral."""
        velocity = {
            'velocity_data_available': 'upload_date' in videos_df.columns and 'views' in videos_df.columns
        }
        
        if velocity['velocity_data_available']:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df['days_old'] = (datetime.now() - videos_df['upload_date']).dt.days
                videos_df['days_old'] = videos_df['days_old'].replace(0, 1)  # Avoid division by zero
                videos_df['views_per_day'] = videos_df['views'] / videos_df['days_old']
                
                velocity.update({
                    'average_views_per_day': int(videos_df['views_per_day'].mean()),
                    'median_views_per_day': int(videos_df['views_per_day'].median()),
                    'fastest_growing_video': {
                        'title': videos_df.loc[videos_df['views_per_day'].idxmax(), 'title'],
                        'views_per_day': int(videos_df['views_per_day'].max())
                    },
                    'velocity_distribution': self._categorize_viral_velocity(videos_df['views_per_day'])
                })
            except Exception as e:
                logger.warning(f"Could not calculate viral velocity: {e}")
        
        return velocity
    
    def _analyze_clickbait_in_trending(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clickbait patterns in trending videos."""
        clickbait_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 'secret', 'revealed', 'exposed', 'must', 'never', 'always', 'everyone', 'nobody']
        
        videos_df['has_clickbait'] = videos_df['title'].str.lower().str.contains('|'.join(clickbait_words), na=False)
        videos_df['question_marks'] = videos_df['title'].str.count(r'\?')
        videos_df['exclamations'] = videos_df['title'].str.count(r'!')
        
        clickbait_analysis = {
            'clickbait_prevalence': round((videos_df['has_clickbait'].sum() / len(videos_df)) * 100, 1),
            'avg_views_clickbait': int(videos_df[videos_df['has_clickbait']]['views'].mean()) if videos_df['has_clickbait'].any() else 0,
            'avg_views_non_clickbait': int(videos_df[~videos_df['has_clickbait']]['views'].mean()) if (~videos_df['has_clickbait']).any() else 0,
            'question_titles_percentage': round((videos_df[videos_df['question_marks'] > 0].shape[0] / len(videos_df)) * 100, 1),
            'exclamation_titles_percentage': round((videos_df[videos_df['exclamations'] > 0].shape[0] / len(videos_df)) * 100, 1),
            'clickbait_effectiveness': 'Effective' if videos_df[videos_df['has_clickbait']]['views'].mean() > videos_df[~videos_df['has_clickbait']]['views'].mean() else 'Not Effective'
        }
        
        return clickbait_analysis
    
    def _identify_emotional_triggers(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify emotional triggers in trending titles."""
        emotion_words = {
            'excitement': ['amazing', 'incredible', 'awesome', 'fantastic', 'epic', 'mind-blowing'],
            'curiosity': ['secret', 'hidden', 'revealed', 'mystery', 'unknown', 'discovered'],
            'urgency': ['now', 'today', 'immediately', 'urgent', 'breaking', 'latest'],
            'exclusivity': ['exclusive', 'first', 'only', 'rare', 'limited', 'special'],
            'fear': ['dangerous', 'scary', 'terrifying', 'warning', 'avoid', 'never'],
            'controversy': ['shocking', 'controversial', 'banned', 'exposed', 'truth', 'lies']
        }
        
        emotion_counts = {}
        for emotion, words in emotion_words.items():
            count = videos_df['title'].str.lower().str.contains('|'.join(words), na=False).sum()
            emotion_counts[emotion] = count
        
        emotional_triggers = {
            'emotion_usage': emotion_counts,
            'most_used_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'None',
            'emotional_content_percentage': round((sum(emotion_counts.values()) / len(videos_df)) * 100, 1),
            'emotion_effectiveness': self._calculate_emotion_effectiveness(videos_df, emotion_words)
        }
        
        return emotional_triggers
    
    def _identify_title_formulas(self, videos_df: pd.DataFrame) -> List[str]:
        """Identify common title formulas in trending content."""
        formulas = []
        
        # Common patterns
        patterns = [
            (r'\d+\s+(things|ways|tips|secrets|facts)', 'Number + Category (e.g., "5 Things...")'),
            (r'how\s+to\s+', 'How-to Format'),
            (r'why\s+', 'Why Questions'),
            (r'what\s+(happens|if)', 'What-if Scenarios'),
            (r'(first|last)\s+time', 'First/Last Time Stories'),
            (r'(before|after)', 'Before/After Comparisons'),
            (r'vs\s+', 'Versus Comparisons'),
            (r'(top|best|worst)\s+\d+', 'Ranked Lists'),
            (r'(trying|testing)', 'Experiment/Review Format'),
            (r'(reaction|reacting)', 'Reaction Content')
        ]
        
        for pattern, description in patterns:
            matches = videos_df['title'].str.lower().str.contains(pattern, na=False).sum()
            if matches > 0:
                percentage = round((matches / len(videos_df)) * 100, 1)
                formulas.append(f"{description}: {matches} videos ({percentage}%)")
        
        return formulas
    
    def _find_optimal_trending_duration(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Find the optimal duration for trending content."""
        duration_performance = videos_df.groupby('duration_category').agg({
            'views': ['mean', 'count'],
            'duration_seconds': 'mean'
        }).round(2)
        
        duration_performance.columns = ['avg_views', 'video_count', 'avg_duration_seconds']
        duration_performance = duration_performance.reset_index()
        
        # Find best performing duration category
        best_duration = duration_performance.loc[duration_performance['avg_views'].idxmax()]
        
        optimal_insights = {
            'optimal_duration_category': best_duration['duration_category'],
            'optimal_avg_views': int(best_duration['avg_views']),
            'optimal_avg_duration_minutes': round(best_duration['avg_duration_seconds'] / 60, 2),
            'duration_performance_breakdown': duration_performance.to_dict('records'),
            'sweet_spot_range': self._find_duration_sweet_spot(videos_df)
        }
        
        return optimal_insights
    
    def _calculate_viral_success_factors(self, viral_videos: pd.DataFrame, non_viral_videos: pd.DataFrame) -> Dict[str, Any]:
        """Calculate factors that differentiate viral from non-viral content."""
        factors = {}
        
        # Title length factor
        viral_title_len = viral_videos['title_length'].mean()
        non_viral_title_len = non_viral_videos['title_length'].mean()
        factors['title_length_factor'] = round(viral_title_len / non_viral_title_len, 2) if non_viral_title_len > 0 else 1
        
        # Duration factor
        viral_duration = viral_videos['duration_seconds'].mean()
        non_viral_duration = non_viral_videos['duration_seconds'].mean()
        factors['duration_factor'] = round(viral_duration / non_viral_duration, 2) if non_viral_duration > 0 else 1
        
        # Numbers in title factor
        viral_numbers_rate = viral_videos['has_numbers_in_title'].mean()
        non_viral_numbers_rate = non_viral_videos['has_numbers_in_title'].mean()
        factors['numbers_effectiveness'] = round(viral_numbers_rate / non_viral_numbers_rate, 2) if non_viral_numbers_rate > 0 else 1
        
        # Caps in title factor
        viral_caps_rate = viral_videos['has_caps_in_title'].mean()
        non_viral_caps_rate = non_viral_videos['has_caps_in_title'].mean()
        factors['caps_effectiveness'] = round(viral_caps_rate / non_viral_caps_rate, 2) if non_viral_caps_rate > 0 else 1
        
        return factors
    
    def _cluster_content_by_similarity(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster similar content together."""
        # Simple keyword-based clustering
        all_titles = ' '.join(videos_df['title'].str.lower())
        words = re.findall(r'\b\w+\b', all_titles)
        word_counts = Counter(words)
        
        # Get top keywords
        top_keywords = [word for word, count in word_counts.most_common(20) if len(word) > 3]
        
        clusters = {}
        for keyword in top_keywords:
            matching_videos = videos_df[videos_df['title'].str.lower().str.contains(keyword, na=False)]
            if len(matching_videos) > 1:
                clusters[keyword] = {
                    'video_count': len(matching_videos),
                    'avg_views': int(matching_videos['views'].mean()),
                    'channels': matching_videos['channel_name'].nunique()
                }
        
        return clusters
    
    def _calculate_market_saturation(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market saturation for different content types."""
        # Analyze channel concentration
        channel_counts = videos_df['channel_name'].value_counts()
        
        saturation = {
            'channel_concentration_ratio': round(channel_counts.head(5).sum() / len(videos_df), 3),
            'unique_channels_ratio': round(videos_df['channel_name'].nunique() / len(videos_df), 3),
            'saturation_level': 'High' if channel_counts.head(5).sum() / len(videos_df) > 0.5 else 'Medium' if channel_counts.head(5).sum() / len(videos_df) > 0.3 else 'Low',
            'market_leaders': channel_counts.head(5).to_dict()
        }
        
        return saturation
    
    def _identify_niche_opportunities(self, videos_df: pd.DataFrame) -> List[str]:
        """Identify potential niche opportunities."""
        opportunities = []
        
        # Analyze underrepresented content types
        duration_dist = videos_df['duration_category'].value_counts()
        total_videos = len(videos_df)
        
        for category, count in duration_dist.items():
            percentage = (count / total_videos) * 100
            if percentage < 15:  # Less than 15% representation
                opportunities.append(f"Underrepresented duration category: {category} ({percentage:.1f}% of trending)")
        
        # Analyze title patterns
        numbers_percentage = (videos_df['has_numbers_in_title'].sum() / total_videos) * 100
        if numbers_percentage < 30:
            opportunities.append(f"Opportunity for numbered titles (only {numbers_percentage:.1f}% use numbers)")
        
        return opportunities
    
    def _measure_competitive_intensity(self, videos_df: pd.DataFrame) -> str:
        """Measure the competitive intensity of the trending landscape."""
        unique_channels = videos_df['channel_name'].nunique()
        total_videos = len(videos_df)
        
        # Calculate competition ratio
        competition_ratio = unique_channels / total_videos
        
        if competition_ratio > 0.8:
            return "Low Competition (High Diversity)"
        elif competition_ratio > 0.5:
            return "Medium Competition"
        else:
            return "High Competition (Low Diversity)"
    
    def _assess_barrier_to_entry(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess barriers to entry for trending content."""
        # Analyze subscriber requirements (if data available)
        barrier_assessment = {
            'view_threshold_estimate': int(videos_df['views'].quantile(0.1)),  # Bottom 10% threshold
            'duration_flexibility': len(videos_df['duration_category'].unique()),
            'channel_diversity': videos_df['channel_name'].nunique(),
            'barrier_level': 'Low' if videos_df['channel_name'].nunique() / len(videos_df) > 0.7 else 'Medium' if videos_df['channel_name'].nunique() / len(videos_df) > 0.4 else 'High'
        }
        
        return barrier_assessment
    
    def _identify_emerging_topics(self, videos_df: pd.DataFrame) -> List[str]:
        """Identify emerging topics in trending content."""
        # This is a simplified version - in practice, you'd compare with historical data
        all_titles = ' '.join(videos_df['title'].str.lower())
        words = re.findall(r'\b\w+\b', all_titles)
        
        # Filter for potentially trending topics
        emerging_indicators = ['new', 'latest', '2024', '2025', 'first', 'breaking', 'just', 'now']
        emerging_topics = []
        
        for word in set(words):
            if any(indicator in videos_df[videos_df['title'].str.lower().str.contains(word, na=False)]['title'].str.lower().str.cat(sep=' ') for indicator in emerging_indicators):
                count = words.count(word)
                if count >= 3 and len(word) > 3:
                    emerging_topics.append(f"{word} ({count} mentions)")
        
        return emerging_topics[:10]  # Top 10 emerging topics
    
    def _identify_declining_topics(self, videos_df: pd.DataFrame) -> List[str]:
        """Identify potentially declining topics."""
        # This would require historical comparison - simplified version
        return ["Requires historical data for accurate analysis"]
    
    def _predict_content_formats(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict future content format trends."""
        current_trends = {
            'short_form_trend': round((len(videos_df[videos_df['duration_seconds'] < 300]) / len(videos_df)) * 100, 1),
            'long_form_trend': round((len(videos_df[videos_df['duration_seconds'] > 1800]) / len(videos_df)) * 100, 1),
            'question_format_trend': round((videos_df['title'].str.contains(r'\?', na=False).sum() / len(videos_df)) * 100, 1),
            'numbered_content_trend': round((videos_df['has_numbers_in_title'].sum() / len(videos_df)) * 100, 1)
        }
        
        predictions = {
            'format_trends': current_trends,
            'predicted_growth_areas': [],
            'format_recommendations': []
        }
        
        # Generate predictions based on current trends
        if current_trends['short_form_trend'] > 40:
            predictions['predicted_growth_areas'].append("Short-form content (under 5 minutes)")
        
        if current_trends['question_format_trend'] > 20:
            predictions['predicted_growth_areas'].append("Question-based titles")
        
        return predictions
    
    def _predict_channel_growth(self, videos_df: pd.DataFrame) -> List[str]:
        """Predict which channels might see growth."""
        # Analyze channels with multiple trending videos
        channel_performance = videos_df['channel_name'].value_counts()
        rising_channels = channel_performance[channel_performance >= 2].head(5)
        
        return [f"{channel} ({count} trending videos)" for channel, count in rising_channels.items()]
    
    def _identify_seasonal_patterns(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify seasonal patterns in content."""
        seasonal_patterns = {
            'seasonal_data_available': 'upload_date' in videos_df.columns
        }
        
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df['month'] = videos_df['upload_date'].dt.month
                videos_df['day_of_week'] = videos_df['upload_date'].dt.day_name()
                
                seasonal_patterns.update({
                    'monthly_distribution': videos_df['month'].value_counts().to_dict(),
                    'weekly_distribution': videos_df['day_of_week'].value_counts().to_dict(),
                    'peak_upload_month': videos_df['month'].mode().iloc[0] if not videos_df.empty else 'Unknown',
                    'peak_upload_day': videos_df['day_of_week'].mode().iloc[0] if not videos_df.empty else 'Unknown'
                })
            except Exception as e:
                logger.warning(f"Could not analyze seasonal patterns: {e}")
        
        return seasonal_patterns
    
    def _calculate_freshness_score(self, days_old_series: pd.Series) -> float:
        """Calculate a freshness score (0-100) where 100 is freshest."""
        avg_age = days_old_series.mean()
        # Inverse relationship: newer content gets higher score
        freshness_score = max(0, 100 - (avg_age * 2))  # Decrease by 2 points per day
        return round(freshness_score, 1)
    
    def _categorize_viral_velocity(self, views_per_day_series: pd.Series) -> Dict[str, int]:
        """Categorize viral velocity into different speed categories."""
        return {
            'explosive_growth': len(views_per_day_series[views_per_day_series > 1000000]),  # >1M views/day
            'rapid_growth': len(views_per_day_series[(views_per_day_series > 100000) & (views_per_day_series <= 1000000)]),  # 100K-1M views/day
            'steady_growth': len(views_per_day_series[(views_per_day_series > 10000) & (views_per_day_series <= 100000)]),  # 10K-100K views/day
            'slow_growth': len(views_per_day_series[views_per_day_series <= 10000])  # <10K views/day
        }
    
    def _calculate_emotion_effectiveness(self, videos_df: pd.DataFrame, emotion_words: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate the effectiveness of different emotional triggers."""
        effectiveness = {}
        
        for emotion, words in emotion_words.items():
            has_emotion = videos_df['title'].str.lower().str.contains('|'.join(words), na=False)
            
            if has_emotion.any():
                avg_views_with = videos_df[has_emotion]['views'].mean()
                avg_views_without = videos_df[~has_emotion]['views'].mean()
                
                if avg_views_without > 0:
                    effectiveness[emotion] = round(avg_views_with / avg_views_without, 2)
                else:
                    effectiveness[emotion] = 1.0
            else:
                effectiveness[emotion] = 0.0
        
        return effectiveness
    
    def _find_duration_sweet_spot(self, videos_df: pd.DataFrame) -> str:
        """Find the duration sweet spot for trending content."""
        # Create more granular duration bins
        videos_df['duration_minutes'] = videos_df['duration_seconds'] / 60
        
        # Create bins for analysis
        bins = [0, 2, 5, 8, 12, 20, 30, float('inf')]
        labels = ['0-2 min', '2-5 min', '5-8 min', '8-12 min', '12-20 min', '20-30 min', '30+ min']
        
        videos_df['duration_bin'] = pd.cut(videos_df['duration_minutes'], bins=bins, labels=labels)
        
        # Find best performing bin
        performance_by_bin = videos_df.groupby('duration_bin')['views'].mean()
        best_bin = performance_by_bin.idxmax()
        
        return f"Sweet spot appears to be {best_bin} with {int(performance_by_bin.max())} average views"
    
    def _compare_category_performance(self, category_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare performance across different trending categories."""
        performance_comparison = {
            'category_rankings': [],
            'view_performance': {},
            'content_characteristics': {}
        }
        
        for category, analysis in category_analyses.items():
            overview = analysis['overview']
            
            performance_comparison['category_rankings'].append({
                'category': category,
                'avg_views': overview['average_views'],
                'total_views': overview['total_views'],
                'unique_channels': overview['unique_channels']
            })
            
            performance_comparison['view_performance'][category] = {
                'average': overview['average_views'],
                'median': overview['median_views'],
                'range': overview['view_range']
            }
        
        # Sort by average views
        performance_comparison['category_rankings'].sort(key=lambda x: x['avg_views'], reverse=True)
        
        return performance_comparison
    
    def _compare_content_strategies(self, category_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare content strategies across categories."""
        strategy_comparison = {}
        
        for category, analysis in category_analyses.items():
            content_patterns = analysis['content_patterns']
            title_trends = analysis['title_trends']
            
            strategy_comparison[category] = {
                'avg_duration': content_patterns['duration_patterns']['average_duration_minutes'],
                'optimal_duration': content_patterns['duration_patterns']['optimal_trending_duration'],
                'clickbait_usage': title_trends['clickbait_analysis']['clickbait_prevalence'],
                'title_characteristics': title_trends['title_characteristics']
            }
        
        return strategy_comparison
    
    def _analyze_audience_preferences(self, category_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze audience preferences across categories."""
        preferences = {
            'duration_preferences': {},
            'content_type_preferences': {},
            'engagement_patterns': {}
        }
        
        for category, analysis in category_analyses.items():
            duration_trends = analysis['duration_trends']
            preferences['duration_preferences'][category] = duration_trends['optimal_duration_insights']
        
        return preferences
    
    def _generate_cross_category_insights(self, category_analyses: Dict[str, Dict]) -> List[str]:
        """Generate insights that apply across categories."""
        insights = []
        
        # Analyze common patterns
        all_avg_durations = []
        all_clickbait_rates = []
        
        for analysis in category_analyses.values():
            all_avg_durations.append(analysis['content_patterns']['duration_patterns']['average_duration_minutes'])
            all_clickbait_rates.append(analysis['title_trends']['clickbait_analysis']['clickbait_prevalence'])
        
        # Generate insights
        avg_duration_across_categories = np.mean(all_avg_durations)
        insights.append(f"Average trending video duration across all categories: {avg_duration_across_categories:.1f} minutes")
        
        avg_clickbait_rate = np.mean(all_clickbait_rates)
        insights.append(f"Average clickbait usage across categories: {avg_clickbait_rate:.1f}%")
        
        if max(all_avg_durations) - min(all_avg_durations) > 5:
            insights.append("Significant duration variation between categories - tailor content length to your niche")
        
        return insights
    
    def _generate_trending_recommendations(self, videos_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for trending success."""
        recommendations = []
        
        # Duration recommendations
        optimal_duration = videos_df.groupby('duration_category')['views'].mean().idxmax()
        recommendations.append(f"Optimal duration category for trending: {optimal_duration}")
        
        # Title recommendations
        avg_title_length = videos_df['title_length'].mean()
        if avg_title_length > 60:
            recommendations.append("Consider shorter titles - trending videos average shorter titles")
        
        numbers_effectiveness = videos_df.groupby('has_numbers_in_title')['views'].mean()
        if len(numbers_effectiveness) > 1 and numbers_effectiveness[True] > numbers_effectiveness[False]:
            recommendations.append("Include numbers in titles - they perform better in trending")
        
        # Channel diversity insights
        channel_concentration = videos_df['channel_name'].value_counts().iloc[0] / len(videos_df)
        if channel_concentration < 0.2:
            recommendations.append("Trending list shows good diversity - new creators have opportunities")
        else:
            recommendations.append("Trending dominated by few channels - focus on unique content to stand out")
        
        # Timing recommendations
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df['days_to_trend'] = (datetime.now() - videos_df['upload_date']).dt.days
                avg_days = videos_df['days_to_trend'].mean()
                
                if avg_days < 3:
                    recommendations.append("Content trends quickly - focus on timely, relevant topics")
                else:
                    recommendations.append("Content takes time to trend - be patient and promote consistently")
            except:
                pass
        
        return recommendations
    
    def _generate_category_recommendations(self, category_analyses: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on category comparison."""
        recommendations = []
        
        # Find best performing category
        best_category = max(category_analyses.items(), 
                          key=lambda x: x[1]['overview']['average_views'])
        
        recommendations.append(f"Highest performing category: {best_category[0]} with {best_category[1]['overview']['average_views']:,} average views")
        
        # Duration insights across categories
        duration_insights = {}
        for category, analysis in category_analyses.items():
            duration_insights[category] = analysis['content_patterns']['duration_patterns']['average_duration_minutes']
        
        optimal_category = max(duration_insights.items(), key=lambda x: x[1])
        recommendations.append(f"Consider {optimal_category[0]} category approach - {optimal_category[1]:.1f} minute average duration")
        
        return recommendations
    
    # Historical analysis methods for evolution tracking
    def _analyze_content_evolution(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how content has evolved over time."""
        evolution = {
            'duration_evolution': [],
            'title_length_evolution': [],
            'keyword_evolution': []
        }
        
        for data in historical_data:
            date = data.get('analysis_date', 'Unknown')
            content_patterns = data.get('content_patterns', {})
            
            if 'duration_patterns' in content_patterns:
                evolution['duration_evolution'].append({
                    'date': date,
                    'avg_duration': content_patterns['duration_patterns'].get('average_duration_minutes', 0)
                })
            
            title_trends = data.get('title_trends', {})
            if 'title_characteristics' in title_trends:
                evolution['title_length_evolution'].append({
                    'date': date,
                    'avg_title_length': title_trends['title_characteristics'].get('average_length', 0)
                })
        
        return evolution
    
    def _analyze_channel_evolution(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how channel landscape has evolved."""
        channel_evolution = {
            'dominance_evolution': [],
            'diversity_evolution': []
        }
        
        for data in historical_data:
            date = data.get('analysis_date', 'Unknown')
            channel_analysis = data.get('channel_analysis', {})
            
            if 'channel_concentration' in channel_analysis:
                concentration = channel_analysis['channel_concentration']
                channel_evolution['dominance_evolution'].append({
                    'date': date,
                    'top_5_share': concentration.get('top_5_channels_share', 0)
                })
        
        return channel_evolution
    
    def _analyze_performance_evolution(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how performance metrics have evolved."""
        performance_evolution = {
            'view_evolution': [],
            'engagement_evolution': []
        }
        
        for data in historical_data:
            date = data.get('analysis_date', 'Unknown')
            overview = data.get('overview', {})
            
            performance_evolution['view_evolution'].append({
                'date': date,
                'avg_views': overview.get('average_views', 0),
                'total_views': overview.get('total_views', 0)
            })
        
        return performance_evolution
    
    def _analyze_trend_lifecycle(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze the lifecycle of trends."""
        return {
            'trend_persistence': 'Requires more sophisticated analysis with longer time series',
            'emerging_trend_success_rate': 'Needs tracking of specific trends over time',
            'trend_decay_patterns': 'Requires historical keyword tracking'
        }
    
    def _identify_emerging_patterns(self, historical_data: List[Dict]) -> List[str]:
        """Identify emerging patterns from historical data."""
        patterns = []
        
        if len(historical_data) >= 2:
            latest = historical_data[-1]
            previous = historical_data[-2]
            
            # Compare duration trends
            latest_duration = latest.get('content_patterns', {}).get('duration_patterns', {}).get('average_duration_minutes', 0)
            previous_duration = previous.get('content_patterns', {}).get('duration_patterns', {}).get('average_duration_minutes', 0)
            
            if latest_duration > previous_duration * 1.1:
                patterns.append("Trend toward longer content detected")
            elif latest_duration < previous_duration * 0.9:
                patterns.append("Trend toward shorter content detected")
            
            # Compare view performance
            latest_views = latest.get('overview', {}).get('average_views', 0)
            previous_views = previous.get('overview', {}).get('average_views', 0)
            
            if latest_views > previous_views * 1.2:
                patterns.append("Significant increase in trending video performance")
            elif latest_views < previous_views * 0.8:
                patterns.append("Decline in trending video performance")
        
        return patterns
    
    def _predict_future_trends(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Predict future trends based on historical data."""
        predictions = {
            'confidence_level': 'Low - requires more data points',
            'predicted_changes': [],
            'recommendation': 'Collect more historical data for accurate predictions'
        }
        
        if len(historical_data) >= 3:
            predictions['confidence_level'] = 'Medium'
            predictions['predicted_changes'] = [
                'Based on limited data - predictions would improve with more historical snapshots'
            ]
        
        return predictions
