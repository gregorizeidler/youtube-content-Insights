"""
Playlist generator module for creating curated playlists based on analysis results.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict
import random
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class PlaylistGenerator:
    """Generate curated playlists based on various criteria and analysis results."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.generated_playlists = {}
    
    def generate_best_performing_playlist(self, videos_data: List[Dict], 
                                        criteria: str = "views", 
                                        max_videos: int = 20,
                                        min_duration: int = 0,
                                        max_duration: int = 3600) -> Dict[str, Any]:
        """
        Generate a playlist of best performing videos based on specified criteria.
        
        Args:
            videos_data: List of video dictionaries
            criteria: Sorting criteria ('views', 'engagement', 'recent', 'duration')
            max_videos: Maximum number of videos in playlist
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            
        Returns:
            Dictionary with playlist information
        """
        if not videos_data:
            return {"error": "No video data provided"}
        
        logger.info(f"Generating best performing playlist with {criteria} criteria")
        
        # Process the data
        videos_df = self.data_processor.process_videos(videos_data)
        
        if videos_df.empty:
            return {"error": "No valid video data to process"}
        
        # Apply duration filters
        filtered_df = videos_df[
            (videos_df['duration_seconds'] >= min_duration) & 
            (videos_df['duration_seconds'] <= max_duration)
        ]
        
        if filtered_df.empty:
            return {"error": "No videos match the duration criteria"}
        
        # Sort based on criteria
        if criteria == "views":
            sorted_df = filtered_df.nlargest(max_videos, 'views')
        elif criteria == "engagement":
            # Calculate engagement score (views per day if upload date available)
            if 'views_per_day' in filtered_df.columns:
                sorted_df = filtered_df.nlargest(max_videos, 'views_per_day')
            else:
                sorted_df = filtered_df.nlargest(max_videos, 'views')
        elif criteria == "recent":
            # Sort by upload date if available, otherwise by scraped date
            if 'upload_date' in filtered_df.columns:
                filtered_df['upload_date'] = pd.to_datetime(filtered_df['upload_date'])
                sorted_df = filtered_df.nlargest(max_videos, 'upload_date')
            else:
                sorted_df = filtered_df.head(max_videos)  # Take first N as "recent"
        elif criteria == "duration":
            # Sort by duration (longest first)
            sorted_df = filtered_df.nlargest(max_videos, 'duration_seconds')
        else:
            sorted_df = filtered_df.nlargest(max_videos, 'views')  # Default to views
        
        # Generate playlist
        playlist = self._create_playlist_structure(
            sorted_df, 
            f"Best {criteria.title()} Videos",
            f"Curated playlist of top {max_videos} videos sorted by {criteria}"
        )
        
        # Add analysis
        playlist['analysis'] = self._analyze_playlist_composition(sorted_df)
        
        # Store playlist
        playlist_id = f"best_{criteria}_{len(self.generated_playlists) + 1}"
        self.generated_playlists[playlist_id] = playlist
        
        return playlist
    
    def generate_themed_playlist(self, videos_data: List[Dict], 
                                theme_keywords: List[str],
                                max_videos: int = 15,
                                diversity_factor: float = 0.3) -> Dict[str, Any]:
        """
        Generate a themed playlist based on keywords.
        
        Args:
            videos_data: List of video dictionaries
            theme_keywords: List of keywords to match
            max_videos: Maximum number of videos in playlist
            diversity_factor: Factor to ensure channel diversity (0.0 = no diversity, 1.0 = max diversity)
            
        Returns:
            Dictionary with themed playlist information
        """
        if not videos_data or not theme_keywords:
            return {"error": "No video data or theme keywords provided"}
        
        logger.info(f"Generating themed playlist for keywords: {theme_keywords}")
        
        # Process the data
        videos_df = self.data_processor.process_videos(videos_data)
        
        if videos_df.empty:
            return {"error": "No valid video data to process"}
        
        # Filter videos matching theme keywords
        theme_pattern = '|'.join(theme_keywords)
        matching_videos = videos_df[
            videos_df['title'].str.contains(theme_pattern, case=False, na=False)
        ]
        
        if matching_videos.empty:
            return {"error": f"No videos found matching keywords: {theme_keywords}"}
        
        # Apply diversity factor
        if diversity_factor > 0:
            selected_videos = self._apply_diversity_selection(
                matching_videos, max_videos, diversity_factor
            )
        else:
            selected_videos = matching_videos.nlargest(max_videos, 'views')
        
        # Generate playlist
        theme_name = " & ".join(theme_keywords).title()
        playlist = self._create_playlist_structure(
            selected_videos,
            f"{theme_name} Collection",
            f"Curated playlist focused on {theme_name.lower()} content"
        )
        
        # Add theme analysis
        playlist['theme_analysis'] = self._analyze_theme_coverage(selected_videos, theme_keywords)
        playlist['analysis'] = self._analyze_playlist_composition(selected_videos)
        
        # Store playlist
        playlist_id = f"themed_{theme_name.lower().replace(' ', '_')}_{len(self.generated_playlists) + 1}"
        self.generated_playlists[playlist_id] = playlist
        
        return playlist
    
    def generate_discovery_playlist(self, videos_data: List[Dict],
                                  exclude_popular: bool = True,
                                  max_videos: int = 25,
                                  min_quality_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Generate a discovery playlist featuring underrated or hidden gems.
        
        Args:
            videos_data: List of video dictionaries
            exclude_popular: Whether to exclude highly popular videos
            max_videos: Maximum number of videos in playlist
            min_quality_threshold: Minimum quality score (0.0 to 1.0)
            
        Returns:
            Dictionary with discovery playlist information
        """
        if not videos_data:
            return {"error": "No video data provided"}
        
        logger.info("Generating discovery playlist for hidden gems")
        
        # Process the data
        videos_df = self.data_processor.process_videos(videos_data)
        
        if videos_df.empty:
            return {"error": "No valid video data to process"}
        
        # Calculate quality score (combination of views and other factors)
        videos_df['quality_score'] = self._calculate_quality_score(videos_df)
        
        # Filter by quality threshold
        quality_videos = videos_df[videos_df['quality_score'] >= min_quality_threshold]
        
        if quality_videos.empty:
            return {"error": "No videos meet the quality threshold"}
        
        # Exclude popular videos if requested
        if exclude_popular:
            # Define popular as top 20% by views
            popular_threshold = videos_df['views'].quantile(0.8)
            discovery_candidates = quality_videos[quality_videos['views'] < popular_threshold]
        else:
            discovery_candidates = quality_videos
        
        if discovery_candidates.empty:
            return {"error": "No discovery candidates found"}
        
        # Select diverse set of videos
        selected_videos = self._select_discovery_videos(discovery_candidates, max_videos)
        
        # Generate playlist
        playlist = self._create_playlist_structure(
            selected_videos,
            "Hidden Gems Discovery",
            "Discover underrated videos that deserve more attention"
        )
        
        # Add discovery analysis
        playlist['discovery_analysis'] = self._analyze_discovery_potential(selected_videos, videos_df)
        playlist['analysis'] = self._analyze_playlist_composition(selected_videos)
        
        # Store playlist
        playlist_id = f"discovery_{len(self.generated_playlists) + 1}"
        self.generated_playlists[playlist_id] = playlist
        
        return playlist
    
    def generate_balanced_playlist(self, videos_data: List[Dict],
                                 duration_target: int = 3600,
                                 variety_weight: float = 0.4,
                                 quality_weight: float = 0.6) -> Dict[str, Any]:
        """
        Generate a balanced playlist optimized for total duration and variety.
        
        Args:
            videos_data: List of video dictionaries
            duration_target: Target total duration in seconds
            variety_weight: Weight for variety in selection (0.0 to 1.0)
            quality_weight: Weight for quality in selection (0.0 to 1.0)
            
        Returns:
            Dictionary with balanced playlist information
        """
        if not videos_data:
            return {"error": "No video data provided"}
        
        logger.info(f"Generating balanced playlist with {duration_target}s target duration")
        
        # Process the data
        videos_df = self.data_processor.process_videos(videos_data)
        
        if videos_df.empty:
            return {"error": "No valid video data to process"}
        
        # Calculate composite scores
        videos_df['quality_score'] = self._calculate_quality_score(videos_df)
        videos_df['variety_score'] = self._calculate_variety_score(videos_df)
        videos_df['composite_score'] = (
            quality_weight * videos_df['quality_score'] + 
            variety_weight * videos_df['variety_score']
        )
        
        # Select videos using knapsack-like approach
        selected_videos = self._optimize_playlist_selection(
            videos_df, duration_target, 'composite_score'
        )
        
        if selected_videos.empty:
            return {"error": "Could not create balanced playlist"}
        
        # Generate playlist
        playlist = self._create_playlist_structure(
            selected_videos,
            "Balanced Mix Playlist",
            f"Optimized playlist targeting {duration_target//60} minutes with balanced variety and quality"
        )
        
        # Add balance analysis
        playlist['balance_analysis'] = self._analyze_playlist_balance(selected_videos, duration_target)
        playlist['analysis'] = self._analyze_playlist_composition(selected_videos)
        
        # Store playlist
        playlist_id = f"balanced_{len(self.generated_playlists) + 1}"
        self.generated_playlists[playlist_id] = playlist
        
        return playlist
    
    def generate_trending_playlist(self, trending_videos: List[Dict],
                                 category_filter: Optional[str] = None,
                                 max_videos: int = 30,
                                 recency_weight: float = 0.3) -> Dict[str, Any]:
        """
        Generate a playlist from trending content.
        
        Args:
            trending_videos: List of trending video dictionaries
            category_filter: Optional category to filter by
            max_videos: Maximum number of videos in playlist
            recency_weight: Weight for recency in selection
            
        Returns:
            Dictionary with trending playlist information
        """
        if not trending_videos:
            return {"error": "No trending video data provided"}
        
        logger.info(f"Generating trending playlist with {max_videos} videos")
        
        # Process the data
        videos_df = self.data_processor.process_videos(trending_videos)
        
        if videos_df.empty:
            return {"error": "No valid trending video data to process"}
        
        # Apply category filter if specified
        if category_filter and 'trending_category' in videos_df.columns:
            videos_df = videos_df[videos_df['trending_category'] == category_filter]
            if videos_df.empty:
                return {"error": f"No trending videos found in category: {category_filter}"}
        
        # Calculate trending score (combination of views and recency)
        videos_df['trending_score'] = self._calculate_trending_score(videos_df, recency_weight)
        
        # Select top videos
        selected_videos = videos_df.nlargest(max_videos, 'trending_score')
        
        # Generate playlist
        category_suffix = f" - {category_filter}" if category_filter else ""
        playlist = self._create_playlist_structure(
            selected_videos,
            f"Trending Now{category_suffix}",
            f"Current trending videos{category_suffix.lower()}"
        )
        
        # Add trending analysis
        playlist['trending_analysis'] = self._analyze_trending_composition(selected_videos)
        playlist['analysis'] = self._analyze_playlist_composition(selected_videos)
        
        # Store playlist
        playlist_id = f"trending_{category_filter or 'all'}_{len(self.generated_playlists) + 1}"
        self.generated_playlists[playlist_id] = playlist
        
        return playlist
    
    def _create_playlist_structure(self, videos_df: pd.DataFrame, 
                                 title: str, description: str) -> Dict[str, Any]:
        """Create the basic playlist structure."""
        playlist = {
            'title': title,
            'description': description,
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'video_count': len(videos_df),
            'total_duration_seconds': int(videos_df['duration_seconds'].sum()),
            'total_duration_formatted': self._format_duration(videos_df['duration_seconds'].sum()),
            'estimated_watch_time': self._format_duration(videos_df['duration_seconds'].sum()),
            'videos': self._format_video_list(videos_df),
            'statistics': {
                'total_views': int(videos_df['views'].sum()),
                'average_views': int(videos_df['views'].mean()),
                'unique_channels': videos_df['channel_name'].nunique(),
                'average_duration': self._format_duration(videos_df['duration_seconds'].mean()),
                'duration_range': {
                    'shortest': self._format_duration(videos_df['duration_seconds'].min()),
                    'longest': self._format_duration(videos_df['duration_seconds'].max())
                }
            }
        }
        
        return playlist
    
    def _format_video_list(self, videos_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format video list for playlist."""
        video_list = []
        
        for idx, row in videos_df.iterrows():
            video_info = {
                'position': len(video_list) + 1,
                'title': row['title'],
                'channel': row['channel_name'],
                'duration': self._format_duration(row['duration_seconds']),
                'views': int(row['views']),
                'views_formatted': self._format_number(row['views']),
                'url': row.get('url', ''),
                'upload_date': row.get('upload_date', 'Unknown')
            }
            
            # Add quality indicators
            if 'quality_score' in row:
                video_info['quality_score'] = round(row['quality_score'], 2)
            
            video_list.append(video_info)
        
        return video_list
    
    def _calculate_quality_score(self, videos_df: pd.DataFrame) -> pd.Series:
        """Calculate quality score for videos."""
        # Normalize views (0-1 scale)
        max_views = videos_df['views'].max()
        min_views = videos_df['views'].min()
        
        if max_views == min_views:
            view_score = pd.Series([0.5] * len(videos_df), index=videos_df.index)
        else:
            view_score = (videos_df['views'] - min_views) / (max_views - min_views)
        
        # Duration score (prefer 5-15 minute videos)
        duration_minutes = videos_df['duration_seconds'] / 60
        duration_score = np.where(
            (duration_minutes >= 5) & (duration_minutes <= 15), 1.0,
            np.where(duration_minutes < 5, duration_minutes / 5,
                    np.maximum(0.1, 1.0 - (duration_minutes - 15) / 30))
        )
        duration_score = pd.Series(duration_score, index=videos_df.index)
        
        # Title quality score (reasonable length, not too many caps)
        title_score = np.where(
            (videos_df['title_length'] >= 30) & (videos_df['title_length'] <= 70), 1.0,
            np.where(videos_df['title_length'] < 30, videos_df['title_length'] / 30,
                    np.maximum(0.3, 1.0 - (videos_df['title_length'] - 70) / 50))
        )
        title_score = pd.Series(title_score, index=videos_df.index)
        
        # Combine scores
        quality_score = (view_score * 0.5 + duration_score * 0.3 + title_score * 0.2)
        
        return quality_score
    
    def _calculate_variety_score(self, videos_df: pd.DataFrame) -> pd.Series:
        """Calculate variety score based on uniqueness."""
        variety_scores = []
        
        for idx, row in videos_df.iterrows():
            score = 0.0
            
            # Channel diversity (lower score for channels with many videos)
            channel_count = (videos_df['channel_name'] == row['channel_name']).sum()
            channel_diversity = 1.0 / channel_count if channel_count > 0 else 0.0
            
            # Duration diversity (prefer different durations)
            duration_category_count = (videos_df['duration_category'] == row['duration_category']).sum()
            duration_diversity = 1.0 / duration_category_count if duration_category_count > 0 else 0.0
            
            # Title uniqueness (simple word overlap check)
            title_words = set(row['title'].lower().split())
            title_uniqueness = 1.0  # Start with full uniqueness
            
            for other_idx, other_row in videos_df.iterrows():
                if idx != other_idx:
                    other_words = set(other_row['title'].lower().split())
                    overlap = len(title_words.intersection(other_words))
                    if overlap > 2:  # Significant overlap
                        title_uniqueness *= 0.9
            
            # Combine variety factors
            variety_score = (channel_diversity * 0.4 + duration_diversity * 0.3 + title_uniqueness * 0.3)
            variety_scores.append(variety_score)
        
        return pd.Series(variety_scores, index=videos_df.index)
    
    def _calculate_trending_score(self, videos_df: pd.DataFrame, recency_weight: float) -> pd.Series:
        """Calculate trending score combining views and recency."""
        # Normalize views
        max_views = videos_df['views'].max()
        min_views = videos_df['views'].min()
        
        if max_views == min_views:
            view_score = pd.Series([0.5] * len(videos_df), index=videos_df.index)
        else:
            view_score = (videos_df['views'] - min_views) / (max_views - min_views)
        
        # Recency score (if upload date available)
        recency_score = pd.Series([0.5] * len(videos_df), index=videos_df.index)  # Default neutral
        
        if 'upload_date' in videos_df.columns:
            try:
                videos_df['upload_date'] = pd.to_datetime(videos_df['upload_date'])
                max_date = videos_df['upload_date'].max()
                min_date = videos_df['upload_date'].min()
                
                if max_date != min_date:
                    days_diff = (max_date - min_date).days
                    if days_diff > 0:
                        recency_score = (videos_df['upload_date'] - min_date).dt.days / days_diff
            except Exception as e:
                logger.warning(f"Could not calculate recency score: {e}")
        
        # Combine scores
        trending_score = (1 - recency_weight) * view_score + recency_weight * recency_score
        
        return trending_score
    
    def _apply_diversity_selection(self, videos_df: pd.DataFrame, 
                                 max_videos: int, diversity_factor: float) -> pd.DataFrame:
        """Apply diversity selection to ensure channel variety."""
        if diversity_factor == 0:
            return videos_df.nlargest(max_videos, 'views')
        
        selected_videos = []
        remaining_videos = videos_df.copy()
        channel_counts = defaultdict(int)
        
        # Calculate max videos per channel based on diversity factor
        max_per_channel = max(1, int(max_videos * (1 - diversity_factor)))
        
        while len(selected_videos) < max_videos and not remaining_videos.empty:
            # Sort remaining videos by views
            remaining_videos = remaining_videos.sort_values('views', ascending=False)
            
            # Try to select the best video that doesn't exceed channel limit
            selected = False
            for idx, row in remaining_videos.iterrows():
                channel = row['channel_name']
                if channel_counts[channel] < max_per_channel:
                    selected_videos.append(row)
                    channel_counts[channel] += 1
                    remaining_videos = remaining_videos.drop(idx)
                    selected = True
                    break
            
            # If no video can be selected due to channel limits, relax the constraint
            if not selected:
                if not remaining_videos.empty:
                    row = remaining_videos.iloc[0]
                    selected_videos.append(row)
                    remaining_videos = remaining_videos.drop(remaining_videos.index[0])
                break
        
        return pd.DataFrame(selected_videos)
    
    def _select_discovery_videos(self, candidates_df: pd.DataFrame, max_videos: int) -> pd.DataFrame:
        """Select videos for discovery playlist using balanced approach."""
        # Sort by quality score but ensure diversity
        candidates_df = candidates_df.sort_values('quality_score', ascending=False)
        
        # Apply moderate diversity (30% diversity factor)
        selected = self._apply_diversity_selection(candidates_df, max_videos, 0.3)
        
        # If not enough videos, fill with remaining best quality
        if len(selected) < max_videos:
            remaining_count = max_videos - len(selected)
            remaining_candidates = candidates_df[~candidates_df.index.isin(selected.index)]
            additional = remaining_candidates.head(remaining_count)
            selected = pd.concat([selected, additional])
        
        return selected
    
    def _optimize_playlist_selection(self, videos_df: pd.DataFrame, 
                                   duration_target: int, score_column: str) -> pd.DataFrame:
        """Optimize playlist selection using a greedy approach."""
        # Sort by score per second (efficiency)
        videos_df['efficiency'] = videos_df[score_column] / videos_df['duration_seconds']
        videos_df = videos_df.sort_values('efficiency', ascending=False)
        
        selected_videos = []
        total_duration = 0
        
        for idx, row in videos_df.iterrows():
            if total_duration + row['duration_seconds'] <= duration_target * 1.2:  # Allow 20% overage
                selected_videos.append(row)
                total_duration += row['duration_seconds']
                
                if total_duration >= duration_target * 0.8:  # At least 80% of target
                    break
        
        return pd.DataFrame(selected_videos)
    
    def _analyze_playlist_composition(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the composition of a playlist."""
        analysis = {
            'duration_analysis': {
                'total_duration_hours': round(videos_df['duration_seconds'].sum() / 3600, 2),
                'average_duration_minutes': round(videos_df['duration_seconds'].mean() / 60, 2),
                'duration_distribution': videos_df['duration_category'].value_counts().to_dict()
            },
            'channel_analysis': {
                'unique_channels': videos_df['channel_name'].nunique(),
                'channel_distribution': videos_df['channel_name'].value_counts().head(10).to_dict(),
                'diversity_score': round(videos_df['channel_name'].nunique() / len(videos_df), 2)
            },
            'performance_analysis': {
                'total_views': int(videos_df['views'].sum()),
                'average_views': int(videos_df['views'].mean()),
                'view_range': {
                    'min': int(videos_df['views'].min()),
                    'max': int(videos_df['views'].max())
                },
                'performance_consistency': round(1 - (videos_df['views'].std() / videos_df['views'].mean()), 2) if videos_df['views'].mean() > 0 else 0
            },
            'content_analysis': {
                'title_characteristics': {
                    'average_title_length': round(videos_df['title_length'].mean(), 1),
                    'titles_with_numbers': int(videos_df['has_numbers_in_title'].sum()),
                    'titles_with_caps': int(videos_df['has_caps_in_title'].sum())
                }
            }
        }
        
        return analysis
    
    def _analyze_theme_coverage(self, videos_df: pd.DataFrame, theme_keywords: List[str]) -> Dict[str, Any]:
        """Analyze how well the playlist covers the theme."""
        coverage_analysis = {
            'keyword_coverage': {},
            'theme_strength': 0.0,
            'content_relevance': 0.0
        }
        
        # Check coverage of each keyword
        for keyword in theme_keywords:
            matching_count = videos_df['title'].str.contains(keyword, case=False, na=False).sum()
            coverage_analysis['keyword_coverage'][keyword] = {
                'matches': int(matching_count),
                'percentage': round((matching_count / len(videos_df)) * 100, 1)
            }
        
        # Calculate overall theme strength
        total_matches = sum(data['matches'] for data in coverage_analysis['keyword_coverage'].values())
        coverage_analysis['theme_strength'] = round((total_matches / len(videos_df)) / len(theme_keywords), 2)
        
        # Content relevance (how many videos match at least one keyword)
        videos_with_keywords = videos_df['title'].str.contains('|'.join(theme_keywords), case=False, na=False).sum()
        coverage_analysis['content_relevance'] = round((videos_with_keywords / len(videos_df)) * 100, 1)
        
        return coverage_analysis
    
    def _analyze_discovery_potential(self, selected_df: pd.DataFrame, all_videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the discovery potential of selected videos."""
        # Calculate percentiles of selected videos in the overall dataset
        view_percentiles = []
        for views in selected_df['views']:
            percentile = (all_videos_df['views'] < views).mean() * 100
            view_percentiles.append(percentile)
        
        discovery_analysis = {
            'underrated_factor': {
                'average_view_percentile': round(np.mean(view_percentiles), 1),
                'hidden_gems_count': len([p for p in view_percentiles if p < 50]),
                'potential_breakouts': len([p for p in view_percentiles if 20 < p < 60])
            },
            'diversity_metrics': {
                'channel_diversity': round(selected_df['channel_name'].nunique() / len(selected_df), 2),
                'duration_variety': len(selected_df['duration_category'].unique()),
                'content_spread': selected_df['duration_category'].value_counts().to_dict()
            },
            'quality_indicators': {
                'average_quality_score': round(selected_df.get('quality_score', pd.Series([0.5] * len(selected_df))).mean(), 2),
                'title_quality': {
                    'average_length': round(selected_df['title_length'].mean(), 1),
                    'well_formatted_titles': int((selected_df['title_length'] >= 30).sum())
                }
            }
        }
        
        return discovery_analysis
    
    def _analyze_playlist_balance(self, videos_df: pd.DataFrame, duration_target: int) -> Dict[str, Any]:
        """Analyze how balanced the playlist is."""
        total_duration = videos_df['duration_seconds'].sum()
        
        balance_analysis = {
            'duration_optimization': {
                'target_duration_minutes': duration_target // 60,
                'actual_duration_minutes': round(total_duration / 60, 1),
                'target_achievement': round((total_duration / duration_target) * 100, 1),
                'efficiency_score': round(min(1.0, total_duration / duration_target), 2)
            },
            'content_balance': {
                'duration_variety': videos_df['duration_category'].value_counts().to_dict(),
                'channel_balance': round(videos_df['channel_name'].nunique() / len(videos_df), 2),
                'performance_balance': round(1 - (videos_df['views'].std() / videos_df['views'].mean()), 2) if videos_df['views'].mean() > 0 else 0
            },
            'optimization_metrics': {
                'quality_consistency': round(videos_df.get('quality_score', pd.Series([0.5] * len(videos_df))).std(), 3),
                'variety_score': round(videos_df.get('variety_score', pd.Series([0.5] * len(videos_df))).mean(), 2)
            }
        }
        
        return balance_analysis
    
    def _analyze_trending_composition(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the composition of trending playlist."""
        trending_analysis = {
            'trending_metrics': {
                'average_trending_score': round(videos_df.get('trending_score', pd.Series([0.5] * len(videos_df))).mean(), 2),
                'score_distribution': {
                    'high_trending': len(videos_df[videos_df.get('trending_score', pd.Series([0.5] * len(videos_df))) > 0.8]),
                    'medium_trending': len(videos_df[(videos_df.get('trending_score', pd.Series([0.5] * len(videos_df))) > 0.5) & 
                                                   (videos_df.get('trending_score', pd.Series([0.5] * len(videos_df))) <= 0.8)]),
                    'low_trending': len(videos_df[videos_df.get('trending_score', pd.Series([0.5] * len(videos_df))) <= 0.5])
                }
            },
            'category_analysis': {},
            'performance_indicators': {
                'total_trending_views': int(videos_df['views'].sum()),
                'average_performance': int(videos_df['views'].mean()),
                'top_performer': {
                    'title': videos_df.loc[videos_df['views'].idxmax(), 'title'],
                    'views': int(videos_df['views'].max())
                }
            }
        }
        
        # Add category analysis if available
        if 'trending_category' in videos_df.columns:
            trending_analysis['category_analysis'] = videos_df['trending_category'].value_counts().to_dict()
        
        return trending_analysis
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if pd.isna(seconds):
            return "0:00"
        
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _format_number(self, number: int) -> str:
        """Format large numbers with K, M, B suffixes."""
        if number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.1f}B"
        elif number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.1f}K"
        else:
            return str(number)
    
    def export_playlist_to_format(self, playlist_id: str, format_type: str = "json") -> Dict[str, Any]:
        """
        Export playlist to various formats.
        
        Args:
            playlist_id: ID of the playlist to export
            format_type: Export format ('json', 'csv', 'txt', 'youtube_urls')
            
        Returns:
            Dictionary with export data or error message
        """
        if playlist_id not in self.generated_playlists:
            return {"error": f"Playlist {playlist_id} not found"}
        
        playlist = self.generated_playlists[playlist_id]
        
        if format_type == "json":
            return {"format": "json", "data": playlist}
        
        elif format_type == "csv":
            # Convert to CSV format
            videos = playlist['videos']
            csv_data = "Position,Title,Channel,Duration,Views,URL\n"
            for video in videos:
                csv_data += f"{video['position']},\"{video['title']}\",\"{video['channel']}\",{video['duration']},{video['views']},{video['url']}\n"
            return {"format": "csv", "data": csv_data}
        
        elif format_type == "txt":
            # Convert to plain text format
            txt_data = f"{playlist['title']}\n"
            txt_data += f"{playlist['description']}\n"
            txt_data += f"Total Duration: {playlist['total_duration_formatted']}\n"
            txt_data += f"Videos: {playlist['video_count']}\n\n"
            
            for video in playlist['videos']:
                txt_data += f"{video['position']}. {video['title']} - {video['channel']} ({video['duration']})\n"
            
            return {"format": "txt", "data": txt_data}
        
        elif format_type == "youtube_urls":
            # Extract just the URLs
            urls = [video['url'] for video in playlist['videos'] if video['url']]
            return {"format": "youtube_urls", "data": urls}
        
        else:
            return {"error": f"Unsupported format: {format_type}"}
    
    def get_playlist_recommendations(self, playlist_id: str) -> List[str]:
        """Get recommendations for improving a playlist."""
        if playlist_id not in self.generated_playlists:
            return ["Playlist not found"]
        
        playlist = self.generated_playlists[playlist_id]
        recommendations = []
        
        # Duration recommendations
        total_minutes = playlist['total_duration_seconds'] / 60
        if total_minutes < 30:
            recommendations.append("Consider adding more videos - playlist is quite short for extended viewing")
        elif total_minutes > 180:
            recommendations.append("Playlist might be too long - consider splitting into multiple themed playlists")
        
        # Diversity recommendations
        analysis = playlist.get('analysis', {})
        channel_analysis = analysis.get('channel_analysis', {})
        diversity_score = channel_analysis.get('diversity_score', 0)
        
        if diversity_score < 0.3:
            recommendations.append("Low channel diversity - consider including videos from more creators")
        elif diversity_score > 0.9:
            recommendations.append("Very high diversity - might lack thematic coherence")
        
        # Performance recommendations
        performance_analysis = analysis.get('performance_analysis', {})
        consistency = performance_analysis.get('performance_consistency', 0)
        
        if consistency < 0.3:
            recommendations.append("High performance variation - consider balancing popular and niche content")
        
        # Content recommendations
        duration_dist = analysis.get('duration_analysis', {}).get('duration_distribution', {})
        if len(duration_dist) == 1:
            recommendations.append("All videos have similar duration - consider mixing short and long content")
        
        return recommendations
    
    def get_all_playlists(self) -> Dict[str, Dict[str, Any]]:
        """Get all generated playlists."""
        return self.generated_playlists
    
    def clear_playlists(self):
        """Clear all generated playlists."""
        self.generated_playlists = {}
