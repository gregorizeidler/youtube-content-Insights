"""
Visualization module for creating charts and reports from YouTube analysis data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Create visualizations and reports from YouTube analysis data."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        self.created_visualizations = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_channel_comparison_chart(self, channel_analyses: List[Dict], 
                                      save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive channel comparison chart.
        
        Args:
            channel_analyses: List of channel analysis dictionaries
            save_path: Optional path to save the chart
            
        Returns:
            Path to the saved chart
        """
        if not channel_analyses:
            raise ValueError("No channel analysis data provided")
        
        logger.info(f"Creating channel comparison chart for {len(channel_analyses)} channels")
        
        # Extract data for visualization
        channels_data = []
        for analysis in channel_analyses:
            channel_info = analysis.get('channel_info', {})
            performance = analysis.get('performance_metrics', {})
            engagement = analysis.get('engagement_analysis', {})
            
            channels_data.append({
                'Channel': channel_info.get('name', 'Unknown'),
                'Subscribers': engagement.get('subscriber_count', 0),
                'Avg Views': performance.get('average_views', 0),
                'Total Views': performance.get('total_views', 0),
                'Engagement Rate': engagement.get('engagement_rate', 0),
                'Consistency Score': engagement.get('consistency_score', 0),
                'Videos Analyzed': channel_info.get('total_videos_analyzed', 0)
            })
        
        df = pd.DataFrame(channels_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('YouTube Channel Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Subscriber Count Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df['Channel'], df['Subscribers'], color=sns.color_palette("husl", len(df)))
        ax1.set_title('Subscriber Count', fontweight='bold')
        ax1.set_ylabel('Subscribers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    self._format_number(height), ha='center', va='bottom')
        
        # 2. Average Views Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df['Channel'], df['Avg Views'], color=sns.color_palette("husl", len(df)))
        ax2.set_title('Average Views per Video', fontweight='bold')
        ax2.set_ylabel('Average Views')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    self._format_number(height), ha='center', va='bottom')
        
        # 3. Engagement Rate Comparison
        ax3 = axes[0, 2]
        bars3 = ax3.bar(df['Channel'], df['Engagement Rate'], color=sns.color_palette("husl", len(df)))
        ax3.set_title('Engagement Rate (%)', fontweight='bold')
        ax3.set_ylabel('Engagement Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 4. Consistency Score
        ax4 = axes[1, 0]
        bars4 = ax4.bar(df['Channel'], df['Consistency Score'], color=sns.color_palette("husl", len(df)))
        ax4.set_title('Content Consistency Score', fontweight='bold')
        ax4.set_ylabel('Consistency Score')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 5. Total Views vs Subscribers Scatter
        ax5 = axes[1, 1]
        scatter = ax5.scatter(df['Subscribers'], df['Total Views'], 
                            c=range(len(df)), cmap='husl', s=100, alpha=0.7)
        ax5.set_title('Total Views vs Subscribers', fontweight='bold')
        ax5.set_xlabel('Subscribers')
        ax5.set_ylabel('Total Views')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        
        # Add channel labels
        for i, channel in enumerate(df['Channel']):
            ax5.annotate(channel, (df['Subscribers'].iloc[i], df['Total Views'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Performance Radar Chart (simplified)
        ax6 = axes[1, 2]
        
        # Normalize metrics for radar chart
        metrics = ['Subscribers', 'Avg Views', 'Engagement Rate', 'Consistency Score']
        normalized_data = df[metrics].copy()
        
        for metric in metrics:
            max_val = normalized_data[metric].max()
            if max_val > 0:
                normalized_data[metric] = normalized_data[metric] / max_val
        
        # Create a simple bar chart instead of radar for simplicity
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(df)
        
        for i, channel in enumerate(df['Channel']):
            values = normalized_data.iloc[i].values
            ax6.bar(x_pos + i * width, values, width, label=channel, alpha=0.7)
        
        ax6.set_title('Normalized Performance Comparison', fontweight='bold')
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Normalized Score (0-1)')
        ax6.set_xticks(x_pos + width * (len(df) - 1) / 2)
        ax6.set_xticklabels(metrics, rotation=45)
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"channel_comparison_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_visualizations.append(save_path)
        logger.info(f"Channel comparison chart saved to: {save_path}")
        
        return save_path
    
    def create_sentiment_analysis_chart(self, sentiment_analysis: Dict, 
                                      save_path: Optional[str] = None) -> str:
        """
        Create sentiment analysis visualization.
        
        Args:
            sentiment_analysis: Sentiment analysis results dictionary
            save_path: Optional path to save the chart
            
        Returns:
            Path to the saved chart
        """
        logger.info("Creating sentiment analysis chart")
        
        overview = sentiment_analysis.get('overview', {})
        detailed = sentiment_analysis.get('detailed_analysis', {})
        emotions = sentiment_analysis.get('emotion_analysis', {})
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YouTube Video Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution Pie Chart
        ax1 = axes[0, 0]
        sentiment_dist = overview.get('sentiment_distribution', {})
        
        if sentiment_dist:
            labels = list(sentiment_dist.keys())
            sizes = list(sentiment_dist.values())
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90, explode=(0.05, 0.05, 0.05))
            ax1.set_title('Sentiment Distribution', fontweight='bold')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 2. Sentiment Score Distribution
        ax2 = axes[0, 1]
        score_stats = detailed.get('score_statistics', {})
        
        if score_stats:
            scores = ['Mean', 'Median', 'Min', 'Max']
            values = [
                score_stats.get('mean_compound_score', 0),
                score_stats.get('median_compound_score', 0),
                score_stats.get('min_compound_score', 0),
                score_stats.get('max_compound_score', 0)
            ]
            
            bars = ax2.bar(scores, values, color=['#3498db', '#9b59b6', '#e74c3c', '#2ecc71'])
            ax2.set_title('Sentiment Score Statistics', fontweight='bold')
            ax2.set_ylabel('Compound Score')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2., value,
                        f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
        
        # 3. Emotion Analysis
        ax3 = axes[1, 0]
        emotion_counts = emotions.get('emotion_counts', {})
        
        if emotion_counts:
            emotions_list = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            bars = ax3.bar(emotions_list, counts, color=sns.color_palette("husl", len(emotions_list)))
            ax3.set_title('Emotion Detection in Comments', fontweight='bold')
            ax3.set_ylabel('Number of Comments')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 4. Sentiment Over Time (if temporal data available)
        ax4 = axes[1, 1]
        temporal = sentiment_analysis.get('temporal_analysis', {})
        
        if temporal.get('temporal_data_available', False):
            # This would require actual temporal data
            ax4.text(0.5, 0.5, 'Temporal Analysis\n(Requires timestamp data)', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Sentiment Trend Over Time', fontweight='bold')
        else:
            # Show overall sentiment classification instead
            overall_class = overview.get('overall_classification', 'Neutral')
            overall_score = overview.get('overall_sentiment_score', 0)
            
            # Create a gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # Color based on sentiment
            if overall_score > 0.1:
                color = '#2ecc71'  # Green for positive
            elif overall_score < -0.1:
                color = '#e74c3c'  # Red for negative
            else:
                color = '#95a5a6'  # Gray for neutral
            
            ax4.fill_between(theta, 0, r, color=color, alpha=0.3)
            ax4.plot(theta, r, color=color, linewidth=3)
            
            # Add score indicator
            score_angle = (overall_score + 1) * np.pi / 2  # Map -1,1 to 0,π
            ax4.plot([score_angle, score_angle], [0, 1], 'k-', linewidth=4)
            
            ax4.set_ylim(0, 1.2)
            ax4.set_xlim(0, np.pi)
            ax4.set_title(f'Overall Sentiment: {overall_class}', fontweight='bold')
            ax4.text(np.pi/2, 0.5, f'Score: {overall_score:.3f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            ax4.set_xticks([0, np.pi/2, np.pi])
            ax4.set_xticklabels(['Negative', 'Neutral', 'Positive'])
            ax4.set_yticks([])
        
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"sentiment_analysis_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_visualizations.append(save_path)
        logger.info(f"Sentiment analysis chart saved to: {save_path}")
        
        return save_path
    
    def create_trending_analysis_chart(self, trending_analysis: Dict, 
                                     save_path: Optional[str] = None) -> str:
        """
        Create trending content analysis visualization.
        
        Args:
            trending_analysis: Trending analysis results dictionary
            save_path: Optional path to save the chart
            
        Returns:
            Path to the saved chart
        """
        logger.info("Creating trending analysis chart")
        
        overview = trending_analysis.get('overview', {})
        content_patterns = trending_analysis.get('content_patterns', {})
        title_trends = trending_analysis.get('title_trends', {})
        viral_factors = trending_analysis.get('viral_factors', {})
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YouTube Trending Content Analysis', fontsize=16, fontweight='bold')
        
        # 1. Duration Distribution
        ax1 = axes[0, 0]
        duration_patterns = content_patterns.get('duration_patterns', {})
        duration_dist = duration_patterns.get('duration_distribution', {})
        
        if duration_dist:
            categories = list(duration_dist.keys())
            counts = list(duration_dist.values())
            
            bars = ax1.bar(categories, counts, color=sns.color_palette("viridis", len(categories)))
            ax1.set_title('Video Duration Distribution', fontweight='bold')
            ax1.set_ylabel('Number of Videos')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 2. Top Trending Keywords
        ax2 = axes[0, 1]
        trending_keywords = title_trends.get('trending_keywords', {})
        
        if trending_keywords:
            # Get top 10 keywords
            top_keywords = dict(list(trending_keywords.items())[:10])
            keywords = list(top_keywords.keys())
            frequencies = list(top_keywords.values())
            
            bars = ax2.barh(keywords, frequencies, color=sns.color_palette("plasma", len(keywords)))
            ax2.set_title('Top Trending Keywords', fontweight='bold')
            ax2.set_xlabel('Frequency')
            
            for i, (bar, freq) in enumerate(zip(bars, frequencies)):
                ax2.text(freq, bar.get_y() + bar.get_height()/2.,
                        f'{freq}', ha='left', va='center')
        
        # 3. Channel Dominance
        ax3 = axes[0, 2]
        channel_analysis = trending_analysis.get('channel_analysis', {})
        top_channels = channel_analysis.get('top_trending_channels', [])
        
        if top_channels and len(top_channels) >= 5:
            top_5_channels = top_channels[:5]
            channel_names = [ch.get('channel_name', 'Unknown') for ch in top_5_channels]
            video_counts = [ch.get('video_count', 0) for ch in top_5_channels]
            
            bars = ax3.bar(range(len(channel_names)), video_counts, 
                          color=sns.color_palette("Set2", len(channel_names)))
            ax3.set_title('Top 5 Trending Channels', fontweight='bold')
            ax3.set_ylabel('Videos in Trending')
            ax3.set_xticks(range(len(channel_names)))
            ax3.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                               for name in channel_names], rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 4. Title Characteristics
        ax4 = axes[1, 0]
        title_chars = title_trends.get('title_characteristics', {})
        
        if title_chars:
            characteristics = ['Avg Length', 'With Numbers (%)', 'With Caps (%)']
            values = [
                title_chars.get('average_length', 0),
                title_chars.get('percentage_with_numbers', 0),
                title_chars.get('percentage_with_caps', 0)
            ]
            
            bars = ax4.bar(characteristics, values, color=['#3498db', '#e74c3c', '#f39c12'])
            ax4.set_title('Title Characteristics', fontweight='bold')
            ax4.set_ylabel('Value')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2., value,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # 5. Viral vs Non-Viral Comparison
        ax5 = axes[1, 1]
        viral_chars = viral_factors.get('viral_characteristics', {})
        non_viral_chars = viral_factors.get('non_viral_characteristics', {})
        
        if viral_chars and non_viral_chars:
            metrics = ['Avg Duration (min)', 'Avg Title Length', 'Numbers Rate (%)', 'Caps Rate (%)']
            viral_values = [
                viral_chars.get('avg_duration_minutes', 0),
                viral_chars.get('avg_title_length', 0),
                viral_chars.get('numbers_in_title_rate', 0),
                viral_chars.get('caps_in_title_rate', 0)
            ]
            non_viral_values = [
                non_viral_chars.get('avg_duration_minutes', 0),
                non_viral_chars.get('avg_title_length', 0),
                non_viral_chars.get('numbers_in_title_rate', 0),
                non_viral_chars.get('caps_in_title_rate', 0)
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, viral_values, width, label='Viral', color='#e74c3c', alpha=0.8)
            bars2 = ax5.bar(x + width/2, non_viral_values, width, label='Non-Viral', color='#3498db', alpha=0.8)
            
            ax5.set_title('Viral vs Non-Viral Content', fontweight='bold')
            ax5.set_ylabel('Value')
            ax5.set_xticks(x)
            ax5.set_xticklabels(metrics, rotation=45)
            ax5.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Performance Overview
        ax6 = axes[1, 2]
        
        # Create a summary metrics visualization
        summary_metrics = {
            'Total Videos': overview.get('total_trending_videos', 0),
            'Unique Channels': overview.get('unique_channels', 0),
            'Avg Views (M)': overview.get('average_views', 0) / 1_000_000,
            'Total Views (B)': overview.get('total_views', 0) / 1_000_000_000
        }
        
        metrics = list(summary_metrics.keys())
        values = list(summary_metrics.values())
        
        bars = ax6.bar(metrics, values, color=sns.color_palette("coolwarm", len(metrics)))
        ax6.set_title('Trending Overview', fontweight='bold')
        ax6.set_ylabel('Value')
        ax6.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2., value,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"trending_analysis_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_visualizations.append(save_path)
        logger.info(f"Trending analysis chart saved to: {save_path}")
        
        return save_path
    
    def create_interactive_dashboard(self, analysis_data: Dict, 
                                   save_path: Optional[str] = None) -> str:
        """
        Create an interactive HTML dashboard using Plotly.
        
        Args:
            analysis_data: Combined analysis data from multiple modules
            save_path: Optional path to save the dashboard
            
        Returns:
            Path to the saved HTML dashboard
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Channel Performance', 'Sentiment Distribution', 
                          'Trending Keywords', 'Content Duration Analysis',
                          'Engagement Metrics', 'Performance Timeline'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"secondary_y": True}, {"type": "scatter"}]]
        )
        
        # Extract data from analysis_data
        channels_data = analysis_data.get('channels', [])
        sentiment_data = analysis_data.get('sentiment', {})
        trending_data = analysis_data.get('trending', {})
        
        # 1. Channel Performance (if available)
        if channels_data:
            channel_names = [ch.get('channel_info', {}).get('name', 'Unknown') for ch in channels_data]
            avg_views = [ch.get('performance_metrics', {}).get('average_views', 0) for ch in channels_data]
            
            fig.add_trace(
                go.Bar(x=channel_names, y=avg_views, name="Avg Views", marker_color='lightblue'),
                row=1, col=1
            )
        
        # 2. Sentiment Distribution (if available)
        if sentiment_data:
            overview = sentiment_data.get('overview', {})
            sentiment_dist = overview.get('sentiment_distribution', {})
            
            if sentiment_dist:
                fig.add_trace(
                    go.Pie(labels=list(sentiment_dist.keys()), 
                          values=list(sentiment_dist.values()),
                          name="Sentiment"),
                    row=1, col=2
                )
        
        # 3. Trending Keywords (if available)
        if trending_data:
            title_trends = trending_data.get('title_trends', {})
            keywords = title_trends.get('trending_keywords', {})
            
            if keywords:
                top_10_keywords = dict(list(keywords.items())[:10])
                fig.add_trace(
                    go.Bar(x=list(top_10_keywords.values()), 
                          y=list(top_10_keywords.keys()),
                          orientation='h',
                          name="Keywords",
                          marker_color='lightgreen'),
                    row=2, col=1
                )
        
        # 4. Duration Analysis (if available)
        if trending_data:
            content_patterns = trending_data.get('content_patterns', {})
            duration_dist = content_patterns.get('duration_patterns', {}).get('duration_distribution', {})
            
            if duration_dist:
                fig.add_trace(
                    go.Bar(x=list(duration_dist.keys()), 
                          y=list(duration_dist.values()),
                          name="Duration",
                          marker_color='orange'),
                    row=2, col=2
                )
        
        # 5. Engagement Metrics (if available)
        if channels_data:
            engagement_rates = [ch.get('engagement_analysis', {}).get('engagement_rate', 0) for ch in channels_data]
            consistency_scores = [ch.get('engagement_analysis', {}).get('consistency_score', 0) for ch in channels_data]
            
            fig.add_trace(
                go.Scatter(x=channel_names, y=engagement_rates, 
                          mode='markers+lines', name="Engagement Rate",
                          marker_color='red'),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=channel_names, y=consistency_scores, 
                          mode='markers+lines', name="Consistency Score",
                          marker_color='blue', yaxis='y2'),
                row=3, col=1, secondary_y=True
            )
        
        # 6. Performance Scatter (if available)
        if channels_data:
            subscribers = [ch.get('engagement_analysis', {}).get('subscriber_count', 0) for ch in channels_data]
            total_views = [ch.get('performance_metrics', {}).get('total_views', 0) for ch in channels_data]
            
            fig.add_trace(
                go.Scatter(x=subscribers, y=total_views,
                          mode='markers',
                          text=channel_names,
                          name="Subs vs Views",
                          marker=dict(size=10, color='purple')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="YouTube Content Insights Dashboard",
            title_x=0.5,
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Channels", row=1, col=1)
        fig.update_yaxes(title_text="Average Views", row=1, col=1)
        
        fig.update_xaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Keywords", row=2, col=1)
        
        fig.update_xaxes(title_text="Duration Category", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        fig.update_xaxes(title_text="Channels", row=3, col=1)
        fig.update_yaxes(title_text="Engagement Rate (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Subscribers", row=3, col=2)
        fig.update_yaxes(title_text="Total Views", row=3, col=2)
        
        # Save the dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"interactive_dashboard_{timestamp}.html")
        
        pyo.plot(fig, filename=save_path, auto_open=False)
        
        self.created_visualizations.append(save_path)
        logger.info(f"Interactive dashboard saved to: {save_path}")
        
        return save_path
    
    def create_performance_report(self, analysis_data: Dict, 
                                save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive performance report with multiple visualizations.
        
        Args:
            analysis_data: Combined analysis data
            save_path: Optional path to save the report
            
        Returns:
            Path to the saved report
        """
        logger.info("Creating comprehensive performance report")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('YouTube Content Performance Report', fontsize=20, fontweight='bold', y=0.98)
        
        # Extract data
        channels_data = analysis_data.get('channels', [])
        sentiment_data = analysis_data.get('sentiment', {})
        trending_data = analysis_data.get('trending', {})
        
        # 1. Channel Performance Overview (2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        if channels_data:
            self._create_channel_performance_subplot(ax1, channels_data)
        else:
            ax1.text(0.5, 0.5, 'No Channel Data Available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Channel Performance Overview', fontweight='bold')
        
        # 2. Sentiment Analysis (1x2)
        ax2 = fig.add_subplot(gs[0, 2:])
        if sentiment_data:
            self._create_sentiment_subplot(ax2, sentiment_data)
        else:
            ax2.text(0.5, 0.5, 'No Sentiment Data Available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Sentiment Analysis', fontweight='bold')
        
        # 3. Trending Keywords (1x2)
        ax3 = fig.add_subplot(gs[1, 2:])
        if trending_data:
            self._create_trending_keywords_subplot(ax3, trending_data)
        else:
            ax3.text(0.5, 0.5, 'No Trending Data Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Trending Keywords', fontweight='bold')
        
        # 4. Content Duration Analysis (2x1)
        ax4 = fig.add_subplot(gs[2:, 0])
        if trending_data:
            self._create_duration_analysis_subplot(ax4, trending_data)
        else:
            ax4.text(0.5, 0.5, 'No Duration Data Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Duration Analysis', fontweight='bold')
        
        # 5. Engagement Metrics (2x1)
        ax5 = fig.add_subplot(gs[2:, 1])
        if channels_data:
            self._create_engagement_subplot(ax5, channels_data)
        else:
            ax5.text(0.5, 0.5, 'No Engagement Data Available', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Engagement Metrics', fontweight='bold')
        
        # 6. Performance Correlation (2x2)
        ax6 = fig.add_subplot(gs[2:, 2:])
        if channels_data:
            self._create_correlation_subplot(ax6, channels_data)
        else:
            ax6.text(0.5, 0.5, 'No Correlation Data Available', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Performance Correlations', fontweight='bold')
        
        # Add timestamp and metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.02, 0.02, f'Generated on: {timestamp}', fontsize=10, alpha=0.7)
        
        # Save the report
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"performance_report_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.created_visualizations.append(save_path)
        logger.info(f"Performance report saved to: {save_path}")
        
        return save_path
    
    def _create_channel_performance_subplot(self, ax, channels_data):
        """Create channel performance subplot."""
        channel_names = [ch.get('channel_info', {}).get('name', 'Unknown')[:15] for ch in channels_data]
        avg_views = [ch.get('performance_metrics', {}).get('average_views', 0) for ch in channels_data]
        subscribers = [ch.get('engagement_analysis', {}).get('subscriber_count', 0) for ch in channels_data]
        
        # Create scatter plot with size based on subscribers
        sizes = [max(50, min(500, sub / 1000)) for sub in subscribers]  # Scale for visibility
        
        scatter = ax.scatter(range(len(channel_names)), avg_views, s=sizes, alpha=0.6, c=range(len(channel_names)), cmap='viridis')
        
        ax.set_xlabel('Channels')
        ax.set_ylabel('Average Views')
        ax.set_title('Channel Performance Overview\n(Bubble size = Subscribers)', fontweight='bold')
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Channel Index')
    
    def _create_sentiment_subplot(self, ax, sentiment_data):
        """Create sentiment analysis subplot."""
        overview = sentiment_data.get('overview', {})
        sentiment_dist = overview.get('sentiment_distribution', {})
        
        if sentiment_dist:
            labels = list(sentiment_dist.keys())
            sizes = list(sentiment_dist.values())
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Sentiment Distribution', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
    
    def _create_trending_keywords_subplot(self, ax, trending_data):
        """Create trending keywords subplot."""
        title_trends = trending_data.get('title_trends', {})
        keywords = title_trends.get('trending_keywords', {})
        
        if keywords:
            top_10_keywords = dict(list(keywords.items())[:10])
            keyword_list = list(top_10_keywords.keys())
            frequencies = list(top_10_keywords.values())
            
            bars = ax.barh(keyword_list, frequencies, color=sns.color_palette("plasma", len(keyword_list)))
            ax.set_xlabel('Frequency')
            ax.set_title('Top Trending Keywords', fontweight='bold')
            
            for bar, freq in zip(bars, frequencies):
                ax.text(freq, bar.get_y() + bar.get_height()/2., f'{freq}', ha='left', va='center')
    
    def _create_duration_analysis_subplot(self, ax, trending_data):
        """Create duration analysis subplot."""
        content_patterns = trending_data.get('content_patterns', {})
        duration_dist = content_patterns.get('duration_patterns', {}).get('duration_distribution', {})
        
        if duration_dist:
            categories = list(duration_dist.keys())
            counts = list(duration_dist.values())
            
            bars = ax.bar(categories, counts, color=sns.color_palette("viridis", len(categories)))
            ax.set_ylabel('Count')
            ax.set_title('Video Duration Distribution', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
    
    def _create_engagement_subplot(self, ax, channels_data):
        """Create engagement metrics subplot."""
        channel_names = [ch.get('channel_info', {}).get('name', 'Unknown')[:10] for ch in channels_data]
        engagement_rates = [ch.get('engagement_analysis', {}).get('engagement_rate', 0) for ch in channels_data]
        consistency_scores = [ch.get('engagement_analysis', {}).get('consistency_score', 0) for ch in channels_data]
        
        x = np.arange(len(channel_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, engagement_rates, width, label='Engagement Rate', alpha=0.8)
        bars2 = ax.bar(x + width/2, consistency_scores, width, label='Consistency Score', alpha=0.8)
        
        ax.set_xlabel('Channels')
        ax.set_ylabel('Score')
        ax.set_title('Engagement Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channel_names, rotation=45)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _create_correlation_subplot(self, ax, channels_data):
        """Create correlation analysis subplot."""
        # Extract metrics for correlation
        metrics_data = []
        for ch in channels_data:
            metrics_data.append({
                'Subscribers': ch.get('engagement_analysis', {}).get('subscriber_count', 0),
                'Avg Views': ch.get('performance_metrics', {}).get('average_views', 0),
                'Engagement Rate': ch.get('engagement_analysis', {}).get('engagement_rate', 0),
                'Consistency': ch.get('engagement_analysis', {}).get('consistency_score', 0)
            })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Add labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45)
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black")
            
            ax.set_title('Metrics Correlation Matrix', fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
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
    
    def create_summary_infographic(self, analysis_data: Dict, 
                                 save_path: Optional[str] = None) -> str:
        """
        Create a summary infographic with key insights.
        
        Args:
            analysis_data: Combined analysis data
            save_path: Optional path to save the infographic
            
        Returns:
            Path to the saved infographic
        """
        logger.info("Creating summary infographic")
        
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # Title
        ax.text(5, 19, 'YouTube Content Insights', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='#2c3e50')
        ax.text(5, 18.3, 'Summary Report', ha='center', va='center', 
                fontsize=16, color='#34495e')
        
        # Extract key metrics
        channels_data = analysis_data.get('channels', [])
        sentiment_data = analysis_data.get('sentiment', {})
        trending_data = analysis_data.get('trending', {})
        
        y_pos = 17
        
        # Channels analyzed
        if channels_data:
            ax.text(1, y_pos, f"📊 Channels Analyzed: {len(channels_data)}", 
                   fontsize=14, fontweight='bold', color='#3498db')
            y_pos -= 0.8
            
            total_subscribers = sum(ch.get('engagement_analysis', {}).get('subscriber_count', 0) for ch in channels_data)
            ax.text(1, y_pos, f"👥 Total Subscribers: {self._format_number(total_subscribers)}", 
                   fontsize=12, color='#2980b9')
            y_pos -= 0.6
            
            total_views = sum(ch.get('performance_metrics', {}).get('total_views', 0) for ch in channels_data)
            ax.text(1, y_pos, f"👀 Total Views: {self._format_number(total_views)}", 
                   fontsize=12, color='#2980b9')
            y_pos -= 1.2
        
        # Sentiment analysis
        if sentiment_data:
            overview = sentiment_data.get('overview', {})
            ax.text(1, y_pos, f"💭 Comments Analyzed: {overview.get('total_comments_analyzed', 0)}", 
                   fontsize=14, fontweight='bold', color='#e74c3c')
            y_pos -= 0.8
            
            overall_sentiment = overview.get('overall_classification', 'Unknown')
            sentiment_score = overview.get('overall_sentiment_score', 0)
            ax.text(1, y_pos, f"😊 Overall Sentiment: {overall_sentiment} ({sentiment_score:.3f})", 
                   fontsize=12, color='#c0392b')
            y_pos -= 1.2
        
        # Trending analysis
        if trending_data:
            overview = trending_data.get('overview', {})
            ax.text(1, y_pos, f"🔥 Trending Videos: {overview.get('total_trending_videos', 0)}", 
                   fontsize=14, fontweight='bold', color='#f39c12')
            y_pos -= 0.8
            
            avg_views = overview.get('average_views', 0)
            ax.text(1, y_pos, f"📈 Avg Trending Views: {self._format_number(avg_views)}", 
                   fontsize=12, color='#e67e22')
            y_pos -= 1.2
        
        # Key insights section
        ax.text(5, y_pos, 'Key Insights', ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#2c3e50')
        y_pos -= 1
        
        # Generate insights based on available data
        insights = self._generate_key_insights(analysis_data)
        
        for i, insight in enumerate(insights[:6]):  # Limit to 6 insights
            ax.text(1, y_pos - i*0.8, f"• {insight}", fontsize=11, color='#34495e', wrap=True)
        
        # Footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(5, 1, f'Generated on {timestamp}', ha='center', va='center', 
                fontsize=10, color='#7f8c8d', style='italic')
        
        # Save the infographic
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"summary_infographic_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.created_visualizations.append(save_path)
        logger.info(f"Summary infographic saved to: {save_path}")
        
        return save_path
    
    def _generate_key_insights(self, analysis_data: Dict) -> List[str]:
        """Generate key insights from analysis data."""
        insights = []
        
        channels_data = analysis_data.get('channels', [])
        sentiment_data = analysis_data.get('sentiment', {})
        trending_data = analysis_data.get('trending', {})
        
        # Channel insights
        if channels_data:
            best_channel = max(channels_data, key=lambda x: x.get('engagement_analysis', {}).get('engagement_rate', 0))
            best_channel_name = best_channel.get('channel_info', {}).get('name', 'Unknown')
            insights.append(f"'{best_channel_name}' has the highest engagement rate")
            
            avg_consistency = np.mean([ch.get('engagement_analysis', {}).get('consistency_score', 0) for ch in channels_data])
            if avg_consistency > 70:
                insights.append("Channels show good content consistency overall")
            else:
                insights.append("Content consistency could be improved across channels")
        
        # Sentiment insights
        if sentiment_data:
            overview = sentiment_data.get('overview', {})
            positive_pct = overview.get('sentiment_percentages', {}).get('positive', 0)
            
            if positive_pct > 60:
                insights.append("Audience sentiment is predominantly positive")
            elif positive_pct < 30:
                insights.append("Audience sentiment shows room for improvement")
            else:
                insights.append("Audience sentiment is mixed - focus on engagement")
        
        # Trending insights
        if trending_data:
            title_trends = trending_data.get('title_trends', {})
            title_chars = title_trends.get('title_characteristics', {})
            
            if title_chars.get('percentage_with_numbers', 0) > 50:
                insights.append("Numbers in titles are common in trending content")
            
            duration_patterns = trending_data.get('content_patterns', {}).get('duration_patterns', {})
            optimal_duration = duration_patterns.get('optimal_trending_duration', '')
            if optimal_duration:
                insights.append(f"Optimal trending duration appears to be {optimal_duration}")
        
        # Default insights if no specific data
        if not insights:
            insights = [
                "Consistent content creation is key to success",
                "Audience engagement varies significantly across channels",
                "Title optimization can improve video performance",
                "Content duration affects viewer retention",
                "Trending patterns change over time"
            ]
        
        return insights
    
    def get_created_visualizations(self) -> List[str]:
        """Get list of all created visualization files."""
        return self.created_visualizations
    
    def clear_visualizations(self):
        """Clear the list of created visualizations."""
        self.created_visualizations = []
    
    def cleanup_old_files(self, days_old: int = 7):
        """Clean up visualization files older than specified days."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        cleaned_files = []
        
        try:
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        cleaned_files.append(filename)
                        
            logger.info(f"Cleaned up {len(cleaned_files)} old visualization files")
            return cleaned_files
            
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")
            return []
