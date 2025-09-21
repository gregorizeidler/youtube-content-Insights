"""
Data processing module for cleaning and structuring scraped YouTube data.
"""

import re
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean scraped YouTube data."""
    
    def __init__(self):
        self.processed_data = {}
    
    def clean_view_count(self, view_string: str) -> int:
        """
        Convert view count string to integer.
        
        Args:
            view_string: String like "1.2M views" or "1,234 views"
            
        Returns:
            Integer view count
        """
        if not view_string or view_string == "N/A":
            return 0
        
        # Remove "views" and extra spaces
        view_string = re.sub(r'\s*views?\s*', '', view_string, flags=re.IGNORECASE)
        view_string = view_string.replace(',', '')
        
        # Handle K, M, B suffixes
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        
        for suffix, multiplier in multipliers.items():
            if suffix in view_string.upper():
                number = float(re.findall(r'[\d.]+', view_string)[0])
                return int(number * multiplier)
        
        # Extract just the number
        numbers = re.findall(r'\d+', view_string)
        return int(numbers[0]) if numbers else 0
    
    def clean_subscriber_count(self, subscriber_string: str) -> int:
        """
        Convert subscriber count string to integer.
        
        Args:
            subscriber_string: String like "1.2M subscribers"
            
        Returns:
            Integer subscriber count
        """
        if not subscriber_string or subscriber_string == "N/A":
            return 0
        
        # Remove "subscribers" and extra spaces
        subscriber_string = re.sub(r'\s*subscribers?\s*', '', subscriber_string, flags=re.IGNORECASE)
        
        # Handle K, M, B suffixes
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        
        for suffix, multiplier in multipliers.items():
            if suffix in subscriber_string.upper():
                number = float(re.findall(r'[\d.]+', subscriber_string)[0])
                return int(number * multiplier)
        
        # Extract just the number
        numbers = re.findall(r'\d+', subscriber_string.replace(',', ''))
        return int(numbers[0]) if numbers else 0
    
    def parse_duration(self, duration_string: str) -> int:
        """
        Convert duration string to seconds.
        
        Args:
            duration_string: String like "10:30" or "1:05:30"
            
        Returns:
            Duration in seconds
        """
        if not duration_string or duration_string == "N/A":
            return 0
        
        parts = duration_string.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        return 0
    
    def parse_upload_time(self, time_string: str) -> str:
        """
        Standardize upload time format.
        
        Args:
            time_string: String like "2 days ago" or "1 week ago"
            
        Returns:
            Standardized time string
        """
        if not time_string or time_string == "N/A":
            return "Unknown"
        
        # Convert relative time to approximate date
        now = datetime.now()
        
        if "minute" in time_string:
            minutes = int(re.findall(r'\d+', time_string)[0])
            date = now - timedelta(minutes=minutes)
        elif "hour" in time_string:
            hours = int(re.findall(r'\d+', time_string)[0])
            date = now - timedelta(hours=hours)
        elif "day" in time_string:
            days = int(re.findall(r'\d+', time_string)[0])
            date = now - timedelta(days=days)
        elif "week" in time_string:
            weeks = int(re.findall(r'\d+', time_string)[0])
            date = now - timedelta(weeks=weeks)
        elif "month" in time_string:
            months = int(re.findall(r'\d+', time_string)[0])
            date = now - timedelta(days=months*30)
        elif "year" in time_string:
            years = int(re.findall(r'\d+', time_string)[0])
            date = now - timedelta(days=years*365)
        else:
            return time_string
        
        return date.strftime('%Y-%m-%d')
    
    def process_videos(self, videos: List[Dict]) -> pd.DataFrame:
        """
        Process video data into a clean DataFrame.
        
        Args:
            videos: List of video dictionaries
            
        Returns:
            Processed DataFrame
        """
        if not videos:
            return pd.DataFrame()
        
        processed_videos = []
        
        for video in videos:
            processed_video = {
                'title': video.get('title', ''),
                'url': video.get('url', ''),
                'channel_name': video.get('channel_name', ''),
                'channel_url': video.get('channel_url', ''),
                'views': self.clean_view_count(video.get('views', '0')),
                'duration_seconds': self.parse_duration(video.get('duration', '0:00')),
                'upload_date': self.parse_upload_time(video.get('upload_time', '')),
                'scraped_at': video.get('scraped_at', ''),
                'title_length': len(video.get('title', '')),
                'has_numbers_in_title': bool(re.search(r'\d', video.get('title', ''))),
                'has_caps_in_title': bool(re.search(r'[A-Z]{2,}', video.get('title', ''))),
            }
            
            # Add trending category if available
            if 'trending_category' in video:
                processed_video['trending_category'] = video['trending_category']
            
            processed_videos.append(processed_video)
        
        df = pd.DataFrame(processed_videos)
        
        # Add calculated metrics
        if not df.empty:
            df['views_per_day'] = df.apply(self._calculate_views_per_day, axis=1)
            df['duration_category'] = df['duration_seconds'].apply(self._categorize_duration)
        
        return df
    
    def process_channels(self, channels: List[Dict]) -> pd.DataFrame:
        """
        Process channel data into a clean DataFrame.
        
        Args:
            channels: List of channel dictionaries
            
        Returns:
            Processed DataFrame
        """
        if not channels:
            return pd.DataFrame()
        
        processed_channels = []
        
        for channel in channels:
            recent_videos = channel.get('recent_videos', [])
            video_df = self.process_videos(recent_videos)
            
            processed_channel = {
                'name': channel.get('name', ''),
                'url': channel.get('url', ''),
                'subscribers': self.clean_subscriber_count(channel.get('subscribers', '0')),
                'total_videos_analyzed': len(recent_videos),
                'avg_views': video_df['views'].mean() if not video_df.empty else 0,
                'avg_duration': video_df['duration_seconds'].mean() if not video_df.empty else 0,
                'total_views': video_df['views'].sum() if not video_df.empty else 0,
                'scraped_at': channel.get('scraped_at', ''),
            }
            
            # Calculate engagement metrics
            if processed_channel['subscribers'] > 0 and processed_channel['avg_views'] > 0:
                processed_channel['engagement_rate'] = (processed_channel['avg_views'] / processed_channel['subscribers']) * 100
            else:
                processed_channel['engagement_rate'] = 0
            
            processed_channels.append(processed_channel)
        
        return pd.DataFrame(processed_channels)
    
    def process_comments(self, comments: List[Dict]) -> pd.DataFrame:
        """
        Process comment data into a clean DataFrame.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Processed DataFrame
        """
        if not comments:
            return pd.DataFrame()
        
        processed_comments = []
        
        for comment in comments:
            processed_comment = {
                'author': comment.get('author', ''),
                'text': comment.get('text', ''),
                'likes': self.clean_view_count(comment.get('likes', '0')),
                'time': comment.get('time', ''),
                'scraped_at': comment.get('scraped_at', ''),
                'text_length': len(comment.get('text', '')),
                'word_count': len(comment.get('text', '').split()),
                'has_emoji': bool(re.search(r'[^\w\s]', comment.get('text', ''))),
                'is_question': '?' in comment.get('text', ''),
                'is_exclamation': '!' in comment.get('text', ''),
            }
            
            processed_comments.append(processed_comment)
        
        return pd.DataFrame(processed_comments)
    
    def _calculate_views_per_day(self, row) -> float:
        """Calculate approximate views per day based on upload date."""
        try:
            upload_date = datetime.strptime(row['upload_date'], '%Y-%m-%d')
            days_since_upload = (datetime.now() - upload_date).days
            if days_since_upload > 0:
                return row['views'] / days_since_upload
            return row['views']
        except:
            return 0
    
    def _categorize_duration(self, seconds: int) -> str:
        """Categorize video duration."""
        if seconds < 60:
            return "Very Short (< 1 min)"
        elif seconds < 300:
            return "Short (1-5 min)"
        elif seconds < 600:
            return "Medium (5-10 min)"
        elif seconds < 1800:
            return "Long (10-30 min)"
        else:
            return "Very Long (> 30 min)"
    
    def generate_summary_stats(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Generate summary statistics for processed data.
        
        Args:
            df: Processed DataFrame
            data_type: Type of data ('videos', 'channels', 'comments')
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_records': len(df),
            'data_type': data_type,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if data_type == 'videos':
            summary.update({
                'total_views': df['views'].sum(),
                'avg_views': df['views'].mean(),
                'median_views': df['views'].median(),
                'avg_duration_minutes': df['duration_seconds'].mean() / 60,
                'most_common_duration_category': df['duration_category'].mode().iloc[0] if not df.empty else 'N/A',
                'channels_represented': df['channel_name'].nunique(),
                'avg_title_length': df['title_length'].mean(),
            })
        
        elif data_type == 'channels':
            summary.update({
                'total_subscribers': df['subscribers'].sum(),
                'avg_subscribers': df['subscribers'].mean(),
                'avg_engagement_rate': df['engagement_rate'].mean(),
                'total_videos_analyzed': df['total_videos_analyzed'].sum(),
            })
        
        elif data_type == 'comments':
            summary.update({
                'total_likes': df['likes'].sum(),
                'avg_likes': df['likes'].mean(),
                'avg_comment_length': df['text_length'].mean(),
                'avg_word_count': df['word_count'].mean(),
                'percentage_with_emoji': (df['has_emoji'].sum() / len(df)) * 100,
                'percentage_questions': (df['is_question'].sum() / len(df)) * 100,
            })
        
        return summary
