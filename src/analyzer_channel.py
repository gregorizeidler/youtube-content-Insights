"""
Channel analysis module for comparing and analyzing YouTube channels.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import re
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class ChannelAnalyzer:
    """Analyze and compare YouTube channels."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.analysis_results = {}
    
    def analyze_single_channel(self, channel_data: Dict, videos_data: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single channel.
        
        Args:
            channel_data: Channel information dictionary
            videos_data: List of video dictionaries from the channel
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing channel: {channel_data.get('name', 'Unknown')}")
        
        # Process the data
        videos_df = self.data_processor.process_videos(videos_data)
        channel_df = self.data_processor.process_channels([channel_data])
        
        if videos_df.empty:
            return {"error": "No video data available for analysis"}
        
        analysis = {
            'channel_info': {
                'name': channel_data.get('name', 'Unknown'),
                'subscribers': self.data_processor.clean_subscriber_count(channel_data.get('subscribers', '0')),
                'total_videos_analyzed': len(videos_data),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_metrics': self._calculate_performance_metrics(videos_df),
            'content_analysis': self._analyze_content_patterns(videos_df),
            'engagement_analysis': self._analyze_engagement(videos_df, channel_data),
            'upload_patterns': self._analyze_upload_patterns(videos_df),
            'title_analysis': self._analyze_titles(videos_df),
            'recommendations': self._generate_recommendations(videos_df, channel_data)
        }
        
        self.analysis_results[channel_data.get('name', 'Unknown')] = analysis
        return analysis
    
    def compare_channels(self, channels_data: List[Dict]) -> Dict[str, Any]:
        """
        Compare multiple channels across various metrics.
        
        Args:
            channels_data: List of channel dictionaries with their video data
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(channels_data)} channels")
        
        if len(channels_data) < 2:
            return {"error": "At least 2 channels required for comparison"}
        
        # Analyze each channel individually first
        individual_analyses = []
        for channel_info in channels_data:
            channel_data = channel_info.get('channel_data', {})
            videos_data = channel_info.get('videos_data', [])
            analysis = self.analyze_single_channel(channel_data, videos_data)
            individual_analyses.append(analysis)
        
        # Perform comparison
        comparison = {
            'comparison_overview': {
                'channels_compared': len(channels_data),
                'comparison_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'channels': [analysis['channel_info']['name'] for analysis in individual_analyses]
            },
            'performance_comparison': self._compare_performance(individual_analyses),
            'content_comparison': self._compare_content_strategies(individual_analyses),
            'engagement_comparison': self._compare_engagement_rates(individual_analyses),
            'competitive_analysis': self._perform_competitive_analysis(individual_analyses),
            'recommendations': self._generate_comparison_recommendations(individual_analyses)
        }
        
        return comparison
    
    def _calculate_performance_metrics(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance metrics for a channel."""
        metrics = {
            'total_views': int(videos_df['views'].sum()),
            'average_views': int(videos_df['views'].mean()),
            'median_views': int(videos_df['views'].median()),
            'view_consistency': float(videos_df['views'].std() / videos_df['views'].mean()) if videos_df['views'].mean() > 0 else 0,
            'top_performing_video': {
                'title': videos_df.loc[videos_df['views'].idxmax(), 'title'],
                'views': int(videos_df['views'].max()),
                'url': videos_df.loc[videos_df['views'].idxmax(), 'url']
            },
            'view_distribution': {
                'top_10_percent_views': int(videos_df.nlargest(max(1, len(videos_df) // 10), 'views')['views'].sum()),
                'bottom_50_percent_views': int(videos_df.nsmallest(len(videos_df) // 2, 'views')['views'].sum())
            }
        }
        
        # Calculate growth trend (if upload dates are available)
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df = videos_df.sort_values('upload_date')
                
                # Simple trend analysis
                recent_videos = videos_df.tail(5)
                older_videos = videos_df.head(5)
                
                if len(recent_videos) > 0 and len(older_videos) > 0:
                    recent_avg = recent_videos['views'].mean()
                    older_avg = older_videos['views'].mean()
                    
                    if older_avg > 0:
                        growth_rate = ((recent_avg - older_avg) / older_avg) * 100
                        metrics['growth_trend'] = {
                            'growth_rate_percent': round(growth_rate, 2),
                            'trend': 'Growing' if growth_rate > 5 else 'Declining' if growth_rate < -5 else 'Stable'
                        }
            except Exception as e:
                logger.warning(f"Could not calculate growth trend: {e}")
                metrics['growth_trend'] = {'error': 'Unable to calculate'}
        
        return metrics
    
    def _analyze_content_patterns(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content patterns and preferences."""
        patterns = {
            'duration_analysis': {
                'average_duration_minutes': round(videos_df['duration_seconds'].mean() / 60, 2),
                'duration_distribution': videos_df['duration_category'].value_counts().to_dict(),
                'optimal_duration': videos_df.groupby('duration_category')['views'].mean().idxmax()
            },
            'title_patterns': {
                'average_title_length': round(videos_df['title_length'].mean(), 1),
                'titles_with_numbers': int(videos_df['has_numbers_in_title'].sum()),
                'titles_with_caps': int(videos_df['has_caps_in_title'].sum()),
                'number_usage_effectiveness': self._calculate_title_effectiveness(videos_df, 'has_numbers_in_title'),
                'caps_usage_effectiveness': self._calculate_title_effectiveness(videos_df, 'has_caps_in_title')
            },
            'upload_frequency': self._calculate_upload_frequency(videos_df)
        }
        
        return patterns
    
    def _analyze_engagement(self, videos_df: pd.DataFrame, channel_data: Dict) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        subscribers = self.data_processor.clean_subscriber_count(channel_data.get('subscribers', '0'))
        
        engagement = {
            'subscriber_count': subscribers,
            'average_views_per_video': int(videos_df['views'].mean()),
            'engagement_rate': 0,
            'view_to_subscriber_ratio': 0,
            'consistency_score': 0
        }
        
        if subscribers > 0:
            engagement['engagement_rate'] = round((videos_df['views'].mean() / subscribers) * 100, 2)
            engagement['view_to_subscriber_ratio'] = round(videos_df['views'].mean() / subscribers, 2)
        
        # Calculate consistency score (lower coefficient of variation = higher consistency)
        if videos_df['views'].mean() > 0:
            cv = videos_df['views'].std() / videos_df['views'].mean()
            engagement['consistency_score'] = round(max(0, 100 - (cv * 50)), 1)
        
        return engagement
    
    def _analyze_upload_patterns(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze upload timing and frequency patterns."""
        patterns = {
            'total_videos_analyzed': len(videos_df),
            'upload_frequency': 'Unable to determine',
            'most_successful_upload_pattern': 'Unable to determine'
        }
        
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                videos_df = videos_df.sort_values('upload_date')
                
                # Calculate time differences between uploads
                time_diffs = videos_df['upload_date'].diff().dt.days.dropna()
                
                if len(time_diffs) > 0:
                    avg_days_between = time_diffs.mean()
                    
                    if avg_days_between <= 1:
                        frequency = "Daily"
                    elif avg_days_between <= 3:
                        frequency = "Every 2-3 days"
                    elif avg_days_between <= 7:
                        frequency = "Weekly"
                    elif avg_days_between <= 14:
                        frequency = "Bi-weekly"
                    elif avg_days_between <= 30:
                        frequency = "Monthly"
                    else:
                        frequency = "Irregular"
                    
                    patterns['upload_frequency'] = frequency
                    patterns['average_days_between_uploads'] = round(avg_days_between, 1)
                
            except Exception as e:
                logger.warning(f"Could not analyze upload patterns: {e}")
        
        return patterns
    
    def _analyze_titles(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze title characteristics and their effectiveness."""
        # Extract common words from titles
        all_titles = ' '.join(videos_df['title'].astype(str))
        words = re.findall(r'\b\w+\b', all_titles.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        word_counts = Counter(filtered_words)
        
        title_analysis = {
            'most_common_words': dict(word_counts.most_common(10)),
            'average_title_length': round(videos_df['title_length'].mean(), 1),
            'title_length_vs_views': self._calculate_correlation(videos_df, 'title_length', 'views'),
            'optimal_title_length_range': self._find_optimal_title_length(videos_df),
            'clickbait_indicators': self._analyze_clickbait_patterns(videos_df)
        }
        
        return title_analysis
    
    def _calculate_title_effectiveness(self, videos_df: pd.DataFrame, feature_column: str) -> Dict[str, Any]:
        """Calculate effectiveness of title features (numbers, caps, etc.)."""
        with_feature = videos_df[videos_df[feature_column] == True]['views'].mean()
        without_feature = videos_df[videos_df[feature_column] == False]['views'].mean()
        
        if pd.isna(with_feature):
            with_feature = 0
        if pd.isna(without_feature):
            without_feature = 0
        
        effectiveness = {
            'avg_views_with_feature': int(with_feature),
            'avg_views_without_feature': int(without_feature),
            'effectiveness_ratio': round(with_feature / without_feature, 2) if without_feature > 0 else 0,
            'recommendation': 'Use' if with_feature > without_feature else 'Avoid'
        }
        
        return effectiveness
    
    def _calculate_upload_frequency(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate upload frequency metrics."""
        frequency_data = {
            'total_videos': len(videos_df),
            'estimated_frequency': 'Unable to determine'
        }
        
        if 'upload_date' in videos_df.columns and len(videos_df) > 1:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                date_range = (videos_df['upload_date'].max() - videos_df['upload_date'].min()).days
                
                if date_range > 0:
                    videos_per_day = len(videos_df) / date_range
                    frequency_data['videos_per_day'] = round(videos_per_day, 3)
                    frequency_data['estimated_frequency'] = f"{round(videos_per_day * 7, 1)} videos per week"
                
            except Exception as e:
                logger.warning(f"Could not calculate upload frequency: {e}")
        
        return frequency_data
    
    def _calculate_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """Calculate correlation between two columns."""
        try:
            return round(df[col1].corr(df[col2]), 3)
        except:
            return 0.0
    
    def _find_optimal_title_length(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Find the optimal title length range based on view performance."""
        # Create title length bins
        videos_df['title_length_bin'] = pd.cut(videos_df['title_length'], 
                                             bins=[0, 30, 50, 70, 100, float('inf')], 
                                             labels=['Very Short (0-30)', 'Short (30-50)', 'Medium (50-70)', 'Long (70-100)', 'Very Long (100+)'])
        
        length_performance = videos_df.groupby('title_length_bin')['views'].agg(['mean', 'count']).reset_index()
        
        # Find the best performing length category
        best_length = length_performance.loc[length_performance['mean'].idxmax(), 'title_length_bin']
        
        return {
            'optimal_range': str(best_length),
            'performance_by_length': length_performance.to_dict('records')
        }
    
    def _analyze_clickbait_patterns(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential clickbait patterns in titles."""
        clickbait_words = ['amazing', 'incredible', 'shocking', 'unbelievable', 'secret', 'hidden', 'revealed', 'exposed', 'must', 'need', 'should', 'never', 'always', 'everyone', 'nobody', 'everything', 'nothing']
        
        videos_df['has_clickbait_words'] = videos_df['title'].str.lower().str.contains('|'.join(clickbait_words), na=False)
        videos_df['question_mark_count'] = videos_df['title'].str.count(r'\?')
        videos_df['exclamation_count'] = videos_df['title'].str.count(r'!')
        
        clickbait_analysis = {
            'videos_with_clickbait_words': int(videos_df['has_clickbait_words'].sum()),
            'avg_views_with_clickbait': int(videos_df[videos_df['has_clickbait_words']]['views'].mean()) if videos_df['has_clickbait_words'].any() else 0,
            'avg_views_without_clickbait': int(videos_df[~videos_df['has_clickbait_words']]['views'].mean()) if (~videos_df['has_clickbait_words']).any() else 0,
            'videos_with_questions': int((videos_df['question_mark_count'] > 0).sum()),
            'videos_with_exclamations': int((videos_df['exclamation_count'] > 0).sum())
        }
        
        return clickbait_analysis
    
    def _compare_performance(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare performance metrics across channels."""
        comparison = {
            'subscriber_ranking': [],
            'view_performance': [],
            'engagement_rates': [],
            'consistency_scores': []
        }
        
        for analysis in analyses:
            channel_name = analysis['channel_info']['name']
            
            comparison['subscriber_ranking'].append({
                'channel': channel_name,
                'subscribers': analysis['engagement_analysis']['subscriber_count']
            })
            
            comparison['view_performance'].append({
                'channel': channel_name,
                'avg_views': analysis['performance_metrics']['average_views'],
                'total_views': analysis['performance_metrics']['total_views']
            })
            
            comparison['engagement_rates'].append({
                'channel': channel_name,
                'engagement_rate': analysis['engagement_analysis']['engagement_rate']
            })
            
            comparison['consistency_scores'].append({
                'channel': channel_name,
                'consistency': analysis['engagement_analysis']['consistency_score']
            })
        
        # Sort rankings
        comparison['subscriber_ranking'].sort(key=lambda x: x['subscribers'], reverse=True)
        comparison['view_performance'].sort(key=lambda x: x['avg_views'], reverse=True)
        comparison['engagement_rates'].sort(key=lambda x: x['engagement_rate'], reverse=True)
        comparison['consistency_scores'].sort(key=lambda x: x['consistency'], reverse=True)
        
        return comparison
    
    def _compare_content_strategies(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare content strategies across channels."""
        strategies = {
            'duration_strategies': [],
            'title_strategies': [],
            'upload_frequencies': []
        }
        
        for analysis in analyses:
            channel_name = analysis['channel_info']['name']
            content = analysis['content_analysis']
            
            strategies['duration_strategies'].append({
                'channel': channel_name,
                'avg_duration_minutes': content['duration_analysis']['average_duration_minutes'],
                'optimal_duration': content['duration_analysis']['optimal_duration']
            })
            
            strategies['title_strategies'].append({
                'channel': channel_name,
                'avg_title_length': content['title_patterns']['average_title_length'],
                'uses_numbers': content['title_patterns']['titles_with_numbers'] > 0,
                'uses_caps': content['title_patterns']['titles_with_caps'] > 0
            })
            
            strategies['upload_frequencies'].append({
                'channel': channel_name,
                'frequency': content['upload_frequency']['estimated_frequency']
            })
        
        return strategies
    
    def _compare_engagement_rates(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare engagement rates and identify best practices."""
        engagement_data = []
        
        for analysis in analyses:
            channel_name = analysis['channel_info']['name']
            engagement = analysis['engagement_analysis']
            
            engagement_data.append({
                'channel': channel_name,
                'engagement_rate': engagement['engagement_rate'],
                'consistency_score': engagement['consistency_score'],
                'view_to_subscriber_ratio': engagement['view_to_subscriber_ratio']
            })
        
        # Find best practices
        best_engagement = max(engagement_data, key=lambda x: x['engagement_rate'])
        most_consistent = max(engagement_data, key=lambda x: x['consistency_score'])
        
        return {
            'engagement_comparison': engagement_data,
            'best_practices': {
                'highest_engagement': best_engagement,
                'most_consistent': most_consistent
            }
        }
    
    def _perform_competitive_analysis(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Perform competitive analysis to identify opportunities."""
        competitive_insights = {
            'market_gaps': [],
            'content_opportunities': [],
            'performance_benchmarks': {}
        }
        
        # Calculate benchmarks
        all_avg_views = [analysis['performance_metrics']['average_views'] for analysis in analyses]
        all_engagement_rates = [analysis['engagement_analysis']['engagement_rate'] for analysis in analyses]
        
        competitive_insights['performance_benchmarks'] = {
            'avg_views_benchmark': int(np.mean(all_avg_views)),
            'top_quartile_views': int(np.percentile(all_avg_views, 75)),
            'avg_engagement_benchmark': round(np.mean(all_engagement_rates), 2),
            'top_quartile_engagement': round(np.percentile(all_engagement_rates, 75), 2)
        }
        
        # Identify content gaps and opportunities
        all_durations = []
        all_title_lengths = []
        
        for analysis in analyses:
            content = analysis['content_analysis']
            all_durations.append(content['duration_analysis']['average_duration_minutes'])
            all_title_lengths.append(content['title_patterns']['average_title_length'])
        
        competitive_insights['content_opportunities'] = {
            'duration_range_gap': f"Consider videos between {min(all_durations):.1f} and {max(all_durations):.1f} minutes",
            'title_length_optimization': f"Optimal title length appears to be around {np.mean(all_title_lengths):.0f} characters"
        }
        
        return competitive_insights
    
    def _generate_recommendations(self, videos_df: pd.DataFrame, channel_data: Dict) -> List[str]:
        """Generate actionable recommendations for channel improvement."""
        recommendations = []
        
        # View performance recommendations
        avg_views = videos_df['views'].mean()
        median_views = videos_df['views'].median()
        
        if avg_views > median_views * 2:
            recommendations.append("You have some high-performing videos. Analyze what made them successful and replicate those elements.")
        
        # Title recommendations
        if videos_df['title_length'].mean() > 70:
            recommendations.append("Consider shortening your video titles. Titles under 60 characters tend to perform better.")
        
        if videos_df['has_numbers_in_title'].sum() / len(videos_df) < 0.3:
            recommendations.append("Try including numbers in your titles (e.g., '5 Tips', 'Top 10'). They often increase click-through rates.")
        
        # Duration recommendations
        avg_duration = videos_df['duration_seconds'].mean() / 60
        if avg_duration > 15:
            recommendations.append("Consider creating shorter videos (8-12 minutes) to improve audience retention.")
        elif avg_duration < 5:
            recommendations.append("Your videos might be too short. Consider expanding content to 8-10 minutes for better engagement.")
        
        # Consistency recommendations
        view_cv = videos_df['views'].std() / videos_df['views'].mean()
        if view_cv > 1:
            recommendations.append("Your view counts are inconsistent. Focus on maintaining quality and identifying your most successful content types.")
        
        # Upload frequency (if data available)
        if len(videos_df) < 10:
            recommendations.append("Increase your upload frequency. Consistent uploads help build audience engagement.")
        
        return recommendations
    
    def _generate_comparison_recommendations(self, analyses: List[Dict]) -> List[str]:
        """Generate recommendations based on channel comparison."""
        recommendations = []
        
        # Find the best performing channel
        best_channel = max(analyses, key=lambda x: x['engagement_analysis']['engagement_rate'])
        best_channel_name = best_channel['channel_info']['name']
        
        recommendations.append(f"Study {best_channel_name}'s content strategy - they have the highest engagement rate.")
        
        # Duration analysis
        durations = [analysis['content_analysis']['duration_analysis']['average_duration_minutes'] for analysis in analyses]
        optimal_duration = np.mean(durations)
        recommendations.append(f"The average optimal video duration across compared channels is {optimal_duration:.1f} minutes.")
        
        # Title strategy
        title_lengths = [analysis['content_analysis']['title_patterns']['average_title_length'] for analysis in analyses]
        optimal_title_length = np.mean(title_lengths)
        recommendations.append(f"Aim for title lengths around {optimal_title_length:.0f} characters based on competitor analysis.")
        
        return recommendations
