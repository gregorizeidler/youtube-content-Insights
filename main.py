"""
YouTube Content Insights - Main Entry Point

A comprehensive web scraping and analysis tool for YouTube content.
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scraper import YouTubeScraper
from src.data_processor import DataProcessor
from src.analyzer_channel import ChannelAnalyzer
from src.analyzer_sentiment import SentimentAnalyzer
from src.analyzer_trends import TrendsAnalyzer
from src.generator_playlist import PlaylistGenerator
from src.generator_thumbnail import ThumbnailAnalyzer
from src.visualizer import DataVisualizer
from src.database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_insights.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class YouTubeContentInsights:
    """Main application class for YouTube Content Insights."""
    
    def __init__(self):
        self.scraper = None
        self.data_processor = DataProcessor()
        self.channel_analyzer = ChannelAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trends_analyzer = TrendsAnalyzer()
        self.playlist_generator = PlaylistGenerator()
        self.thumbnail_analyzer = ThumbnailAnalyzer()
        self.visualizer = DataVisualizer()
        self.database = DatabaseManager()
        
        self.session_data = {
            'scraped_videos': [],
            'scraped_channels': [],
            'scraped_comments': [],
            'analysis_results': {}
        }
    
    def display_welcome(self):
        """Display welcome message and application info."""
        print("\n" + "="*60)
        print("🎥 YOUTUBE CONTENT INSIGHTS")
        print("="*60)
        print("A comprehensive analysis tool for YouTube content")
        print("Features:")
        print("  • Video and Channel Scraping")
        print("  • Sentiment Analysis")
        print("  • Trending Content Analysis")
        print("  • Playlist Generation")
        print("  • Thumbnail Analysis")
        print("  • Data Visualization")
        print("  • Database Storage")
        print("="*60)
    
    def display_main_menu(self):
        """Display the main menu options."""
        print("\n📋 MAIN MENU")
        print("-" * 30)
        print("1.  🔍 Scrape YouTube Data")
        print("2.  📊 Analyze Channel Performance")
        print("3.  💭 Analyze Sentiment")
        print("4.  🔥 Analyze Trending Content")
        print("5.  🎵 Generate Playlists")
        print("6.  🖼️  Analyze Thumbnails")
        print("7.  📈 Create Visualizations")
        print("8.  💾 Database Operations")
        print("9.  ⚙️  Settings & Configuration")
        print("10. 📄 Generate Reports")
        print("0.  ❌ Exit")
        print("-" * 30)
    
    def setup_scraper(self) -> bool:
        """Setup the YouTube scraper with user configuration."""
        print("\n🔧 SCRAPER SETUP")
        print("Please provide the paths to your browser and ChromeDriver:")
        
        # Get browser path
        print("\nCommon browser paths:")
        print("  Windows Chrome: C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
        print("  Windows Brave: C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe")
        print("  macOS Chrome: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        print("  Linux Chrome: /usr/bin/google-chrome")
        
        browser_path = input("\nEnter browser executable path: ").strip()
        if not browser_path or not os.path.exists(browser_path):
            print("❌ Invalid browser path!")
            return False
        
        # Get ChromeDriver path
        print("\nChromeDriver should match your browser version.")
        print("Download from: https://chromedriver.chromium.org/")
        
        driver_path = input("Enter ChromeDriver executable path: ").strip()
        if not driver_path or not os.path.exists(driver_path):
            print("❌ Invalid ChromeDriver path!")
            return False
        
        # Ask about headless mode
        headless = input("Run in headless mode? (y/n): ").strip().lower() == 'y'
        
        try:
            self.scraper = YouTubeScraper(browser_path, driver_path, headless)
            print("✅ Scraper setup successful!")
            return True
        except Exception as e:
            print(f"❌ Error setting up scraper: {e}")
            return False
    
    def scrape_data_menu(self):
        """Handle data scraping operations."""
        if not self.scraper:
            print("\n⚠️  Scraper not configured. Setting up...")
            if not self.setup_scraper():
                return
        
        print("\n🔍 SCRAPING OPTIONS")
        print("1. Search and scrape videos")
        print("2. Scrape channel information")
        print("3. Scrape trending videos")
        print("4. Scrape video comments")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.scrape_videos()
        elif choice == '2':
            self.scrape_channel()
        elif choice == '3':
            self.scrape_trending()
        elif choice == '4':
            self.scrape_comments()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def scrape_videos(self):
        """Scrape videos based on search query."""
        query = input("\nEnter search query: ").strip()
        if not query:
            print("❌ Search query cannot be empty!")
            return
        
        try:
            max_results = int(input("Maximum results (default 50): ").strip() or "50")
        except ValueError:
            max_results = 50
        
        print(f"\n🔍 Searching for videos: '{query}'...")
        
        try:
            videos = self.scraper.search_videos(query, max_results)
            
            if videos:
                self.session_data['scraped_videos'].extend(videos)
                
                # Store in database
                stored_count = self.database.store_videos(videos)
                
                print(f"✅ Successfully scraped {len(videos)} videos!")
                print(f"💾 Stored {stored_count} videos in database")
                
                # Show sample results
                print("\n📋 Sample Results:")
                for i, video in enumerate(videos[:3]):
                    print(f"{i+1}. {video['title'][:60]}... - {video['channel_name']}")
                
                if len(videos) > 3:
                    print(f"... and {len(videos) - 3} more videos")
                    
            else:
                print("❌ No videos found!")
                
        except Exception as e:
            print(f"❌ Error scraping videos: {e}")
    
    def scrape_channel(self):
        """Scrape channel information."""
        channel_url = input("\nEnter YouTube channel URL: ").strip()
        if not channel_url:
            print("❌ Channel URL cannot be empty!")
            return
        
        print(f"\n🔍 Scraping channel: {channel_url}")
        
        try:
            channel_data = self.scraper.scrape_channel_info(channel_url)
            
            if channel_data:
                self.session_data['scraped_channels'].append(channel_data)
                
                # Process channel data for database storage
                processed_channel = {
                    'name': channel_data['name'],
                    'url': channel_data['url'],
                    'subscribers': self.data_processor.clean_subscriber_count(channel_data['subscribers']),
                    'total_videos_analyzed': channel_data['total_videos_scraped'],
                    'scraped_at': channel_data['scraped_at']
                }
                
                # Store in database
                self.database.store_channels([processed_channel])
                
                # Store videos if any
                if channel_data.get('recent_videos'):
                    self.database.store_videos(channel_data['recent_videos'])
                
                print(f"✅ Successfully scraped channel: {channel_data['name']}")
                print(f"👥 Subscribers: {channel_data['subscribers']}")
                print(f"🎥 Recent videos analyzed: {channel_data['total_videos_scraped']}")
                
            else:
                print("❌ Failed to scrape channel!")
                
        except Exception as e:
            print(f"❌ Error scraping channel: {e}")
    
    def scrape_trending(self):
        """Scrape trending videos."""
        print("\n🔥 TRENDING CATEGORIES")
        print("1. All")
        print("2. Music")
        print("3. Gaming")
        print("4. Movies")
        print("5. News")
        
        category_map = {'1': 'all', '2': 'music', '3': 'gaming', '4': 'movies', '5': 'news'}
        choice = input("Select category (default 1): ").strip() or '1'
        category = category_map.get(choice, 'all')
        
        print(f"\n🔍 Scraping trending videos: {category}")
        
        try:
            trending_videos = self.scraper.scrape_trending_videos(category)
            
            if trending_videos:
                self.session_data['scraped_videos'].extend(trending_videos)
                
                # Store in database
                stored_count = self.database.store_videos(trending_videos)
                
                print(f"✅ Successfully scraped {len(trending_videos)} trending videos!")
                print(f"💾 Stored {stored_count} videos in database")
                
                # Show sample results
                print("\n📋 Top Trending Videos:")
                for i, video in enumerate(trending_videos[:5]):
                    views = self.data_processor.clean_view_count(video.get('views', '0'))
                    print(f"{i+1}. {video['title'][:50]}... - {views:,} views")
                    
            else:
                print("❌ No trending videos found!")
                
        except Exception as e:
            print(f"❌ Error scraping trending videos: {e}")
    
    def scrape_comments(self):
        """Scrape comments from a video."""
        video_url = input("\nEnter YouTube video URL: ").strip()
        if not video_url:
            print("❌ Video URL cannot be empty!")
            return
        
        try:
            max_comments = int(input("Maximum comments (default 100): ").strip() or "100")
        except ValueError:
            max_comments = 100
        
        print(f"\n🔍 Scraping comments from video...")
        
        try:
            comments = self.scraper.scrape_video_comments(video_url, max_comments)
            
            if comments:
                self.session_data['scraped_comments'].extend(comments)
                
                # Store in database
                stored_count = self.database.store_comments(comments, video_url)
                
                print(f"✅ Successfully scraped {len(comments)} comments!")
                print(f"💾 Stored {stored_count} comments in database")
                
                # Show sample results
                print("\n📋 Sample Comments:")
                for i, comment in enumerate(comments[:3]):
                    print(f"{i+1}. {comment['author']}: {comment['text'][:80]}...")
                    
            else:
                print("❌ No comments found!")
                
        except Exception as e:
            print(f"❌ Error scraping comments: {e}")
    
    def analyze_channels_menu(self):
        """Handle channel analysis operations."""
        print("\n📊 CHANNEL ANALYSIS OPTIONS")
        print("1. Analyze single channel")
        print("2. Compare multiple channels")
        print("3. Analyze from database")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.analyze_single_channel()
        elif choice == '2':
            self.compare_channels()
        elif choice == '3':
            self.analyze_channels_from_db()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def analyze_single_channel(self):
        """Analyze a single channel."""
        if not self.session_data['scraped_channels']:
            print("❌ No channel data available. Please scrape a channel first.")
            return
        
        print("\n📋 Available Channels:")
        for i, channel in enumerate(self.session_data['scraped_channels']):
            print(f"{i+1}. {channel['name']}")
        
        try:
            choice = int(input("Select channel to analyze: ").strip()) - 1
            if 0 <= choice < len(self.session_data['scraped_channels']):
                channel_data = self.session_data['scraped_channels'][choice]
                videos_data = channel_data.get('recent_videos', [])
                
                print(f"\n🔍 Analyzing channel: {channel_data['name']}")
                
                analysis = self.channel_analyzer.analyze_single_channel(channel_data, videos_data)
                
                # Store analysis result
                self.database.store_analysis_result(
                    'channel', 
                    f"single_channel_{channel_data['name']}", 
                    analysis
                )
                
                self.session_data['analysis_results']['channel_analysis'] = analysis
                
                # Display key results
                self.display_channel_analysis_summary(analysis)
                
            else:
                print("❌ Invalid selection!")
                
        except (ValueError, IndexError):
            print("❌ Invalid input!")
    
    def compare_channels(self):
        """Compare multiple channels."""
        if len(self.session_data['scraped_channels']) < 2:
            print("❌ Need at least 2 channels for comparison. Please scrape more channels.")
            return
        
        print("\n📋 Available Channels:")
        for i, channel in enumerate(self.session_data['scraped_channels']):
            print(f"{i+1}. {channel['name']}")
        
        print("\nSelect channels to compare (comma-separated numbers):")
        try:
            selections = input("Enter selections: ").strip().split(',')
            selected_indices = [int(s.strip()) - 1 for s in selections]
            
            if len(selected_indices) < 2:
                print("❌ Please select at least 2 channels!")
                return
            
            channels_to_compare = []
            for idx in selected_indices:
                if 0 <= idx < len(self.session_data['scraped_channels']):
                    channel_data = self.session_data['scraped_channels'][idx]
                    channels_to_compare.append({
                        'channel_data': channel_data,
                        'videos_data': channel_data.get('recent_videos', [])
                    })
            
            if len(channels_to_compare) >= 2:
                print(f"\n🔍 Comparing {len(channels_to_compare)} channels...")
                
                comparison = self.channel_analyzer.compare_channels(channels_to_compare)
                
                # Store analysis result
                channel_names = [ch['channel_data']['name'] for ch in channels_to_compare]
                self.database.store_analysis_result(
                    'channel_comparison', 
                    f"comparison_{'_vs_'.join(channel_names[:2])}", 
                    comparison
                )
                
                self.session_data['analysis_results']['channel_comparison'] = comparison
                
                # Display key results
                self.display_channel_comparison_summary(comparison)
                
            else:
                print("❌ Invalid channel selections!")
                
        except (ValueError, IndexError):
            print("❌ Invalid input!")
    
    def analyze_channels_from_db(self):
        """Analyze channels from database."""
        channels = self.database.get_channels(limit=10)
        
        if not channels:
            print("❌ No channels found in database!")
            return
        
        print(f"\n📋 Found {len(channels)} channels in database:")
        for i, channel in enumerate(channels):
            print(f"{i+1}. {channel['name']} - {channel['subscribers']:,} subscribers")
        
        print("\nNote: This will analyze channels based on stored data.")
        print("For complete analysis, scrape fresh channel data.")
        
        # This would require implementing database-based analysis
        print("🚧 Database-based analysis coming soon!")
    
    def analyze_sentiment_menu(self):
        """Handle sentiment analysis operations."""
        if not self.session_data['scraped_comments']:
            print("❌ No comment data available. Please scrape video comments first.")
            return
        
        print("\n💭 SENTIMENT ANALYSIS")
        print(f"Available comments: {len(self.session_data['scraped_comments'])}")
        
        video_info = {
            'title': input("Enter video title (optional): ").strip() or "Unknown Video"
        }
        
        print(f"\n🔍 Analyzing sentiment for {len(self.session_data['scraped_comments'])} comments...")
        
        try:
            analysis = self.sentiment_analyzer.analyze_video_sentiment(
                self.session_data['scraped_comments'], 
                video_info
            )
            
            # Store analysis result
            self.database.store_analysis_result(
                'sentiment', 
                f"sentiment_{video_info['title']}", 
                analysis
            )
            
            self.session_data['analysis_results']['sentiment_analysis'] = analysis
            
            # Display key results
            self.display_sentiment_analysis_summary(analysis)
            
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {e}")
    
    def analyze_trending_menu(self):
        """Handle trending analysis operations."""
        # Check for trending videos in session data
        trending_videos = [v for v in self.session_data['scraped_videos'] 
                          if v.get('trending_category')]
        
        if not trending_videos:
            print("❌ No trending video data available. Please scrape trending videos first.")
            return
        
        print(f"\n🔥 TRENDING ANALYSIS")
        print(f"Available trending videos: {len(trending_videos)}")
        
        category = trending_videos[0].get('trending_category', 'all')
        
        print(f"\n🔍 Analyzing trending content for category: {category}")
        
        try:
            analysis = self.trends_analyzer.analyze_trending_content(trending_videos, category)
            
            # Store analysis result
            self.database.store_analysis_result(
                'trending', 
                f"trending_{category}", 
                analysis
            )
            
            # Store trending snapshot
            self.database.store_trending_snapshot(analysis, category)
            
            self.session_data['analysis_results']['trending_analysis'] = analysis
            
            # Display key results
            self.display_trending_analysis_summary(analysis)
            
        except Exception as e:
            print(f"❌ Error analyzing trending content: {e}")
    
    def generate_playlists_menu(self):
        """Handle playlist generation operations."""
        if not self.session_data['scraped_videos']:
            print("❌ No video data available. Please scrape videos first.")
            return
        
        print("\n🎵 PLAYLIST GENERATION OPTIONS")
        print("1. Best performing videos")
        print("2. Themed playlist")
        print("3. Discovery playlist (hidden gems)")
        print("4. Balanced playlist")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.generate_best_performing_playlist()
        elif choice == '2':
            self.generate_themed_playlist()
        elif choice == '3':
            self.generate_discovery_playlist()
        elif choice == '4':
            self.generate_balanced_playlist()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def generate_best_performing_playlist(self):
        """Generate best performing playlist."""
        print("\n📊 BEST PERFORMING PLAYLIST")
        print("Criteria options:")
        print("1. Views")
        print("2. Engagement")
        print("3. Recent")
        print("4. Duration")
        
        criteria_map = {'1': 'views', '2': 'engagement', '3': 'recent', '4': 'duration'}
        choice = input("Select criteria (default 1): ").strip() or '1'
        criteria = criteria_map.get(choice, 'views')
        
        try:
            max_videos = int(input("Maximum videos in playlist (default 20): ").strip() or "20")
        except ValueError:
            max_videos = 20
        
        print(f"\n🎵 Generating playlist with {criteria} criteria...")
        
        try:
            playlist = self.playlist_generator.generate_best_performing_playlist(
                self.session_data['scraped_videos'], 
                criteria, 
                max_videos
            )
            
            if 'error' not in playlist:
                # Store playlist
                playlist_id = self.database.store_playlist(playlist)
                
                print(f"✅ Generated playlist: {playlist['title']}")
                print(f"🎥 Videos: {playlist['video_count']}")
                print(f"⏱️  Total duration: {playlist['total_duration_formatted']}")
                print(f"💾 Stored with ID: {playlist_id}")
                
                # Show top videos
                print("\n📋 Top Videos in Playlist:")
                for video in playlist['videos'][:5]:
                    print(f"{video['position']}. {video['title'][:50]}... - {video['views_formatted']} views")
                    
            else:
                print(f"❌ Error: {playlist['error']}")
                
        except Exception as e:
            print(f"❌ Error generating playlist: {e}")
    
    def generate_themed_playlist(self):
        """Generate themed playlist."""
        print("\n🎯 THEMED PLAYLIST")
        
        keywords_input = input("Enter theme keywords (comma-separated): ").strip()
        if not keywords_input:
            print("❌ Keywords cannot be empty!")
            return
        
        keywords = [k.strip() for k in keywords_input.split(',')]
        
        try:
            max_videos = int(input("Maximum videos in playlist (default 15): ").strip() or "15")
        except ValueError:
            max_videos = 15
        
        print(f"\n🎵 Generating themed playlist for: {', '.join(keywords)}")
        
        try:
            playlist = self.playlist_generator.generate_themed_playlist(
                self.session_data['scraped_videos'], 
                keywords, 
                max_videos
            )
            
            if 'error' not in playlist:
                # Store playlist
                playlist_id = self.database.store_playlist(playlist)
                
                print(f"✅ Generated playlist: {playlist['title']}")
                print(f"🎥 Videos: {playlist['video_count']}")
                print(f"⏱️  Total duration: {playlist['total_duration_formatted']}")
                print(f"💾 Stored with ID: {playlist_id}")
                
                # Show theme coverage
                theme_analysis = playlist.get('theme_analysis', {})
                relevance = theme_analysis.get('content_relevance', 0)
                print(f"🎯 Theme relevance: {relevance}%")
                
            else:
                print(f"❌ Error: {playlist['error']}")
                
        except Exception as e:
            print(f"❌ Error generating themed playlist: {e}")
    
    def generate_discovery_playlist(self):
        """Generate discovery playlist."""
        print("\n💎 DISCOVERY PLAYLIST (Hidden Gems)")
        
        exclude_popular = input("Exclude highly popular videos? (y/n, default y): ").strip().lower() != 'n'
        
        try:
            max_videos = int(input("Maximum videos in playlist (default 25): ").strip() or "25")
        except ValueError:
            max_videos = 25
        
        print(f"\n🔍 Generating discovery playlist...")
        
        try:
            playlist = self.playlist_generator.generate_discovery_playlist(
                self.session_data['scraped_videos'], 
                exclude_popular, 
                max_videos
            )
            
            if 'error' not in playlist:
                # Store playlist
                playlist_id = self.database.store_playlist(playlist)
                
                print(f"✅ Generated playlist: {playlist['title']}")
                print(f"🎥 Videos: {playlist['video_count']}")
                print(f"⏱️  Total duration: {playlist['total_duration_formatted']}")
                print(f"💾 Stored with ID: {playlist_id}")
                
                # Show discovery metrics
                discovery_analysis = playlist.get('discovery_analysis', {})
                underrated_factor = discovery_analysis.get('underrated_factor', {})
                hidden_gems = underrated_factor.get('hidden_gems_count', 0)
                print(f"💎 Hidden gems found: {hidden_gems}")
                
            else:
                print(f"❌ Error: {playlist['error']}")
                
        except Exception as e:
            print(f"❌ Error generating discovery playlist: {e}")
    
    def generate_balanced_playlist(self):
        """Generate balanced playlist."""
        print("\n⚖️  BALANCED PLAYLIST")
        
        try:
            duration_minutes = int(input("Target duration in minutes (default 60): ").strip() or "60")
            duration_target = duration_minutes * 60
        except ValueError:
            duration_target = 3600
        
        print(f"\n🎵 Generating balanced playlist for {duration_minutes} minutes...")
        
        try:
            playlist = self.playlist_generator.generate_balanced_playlist(
                self.session_data['scraped_videos'], 
                duration_target
            )
            
            if 'error' not in playlist:
                # Store playlist
                playlist_id = self.database.store_playlist(playlist)
                
                print(f"✅ Generated playlist: {playlist['title']}")
                print(f"🎥 Videos: {playlist['video_count']}")
                print(f"⏱️  Total duration: {playlist['total_duration_formatted']}")
                print(f"💾 Stored with ID: {playlist_id}")
                
                # Show balance metrics
                balance_analysis = playlist.get('balance_analysis', {})
                duration_opt = balance_analysis.get('duration_optimization', {})
                achievement = duration_opt.get('target_achievement', 0)
                print(f"🎯 Target achievement: {achievement}%")
                
            else:
                print(f"❌ Error: {playlist['error']}")
                
        except Exception as e:
            print(f"❌ Error generating balanced playlist: {e}")
    
    def analyze_thumbnails_menu(self):
        """Handle thumbnail analysis operations."""
        if not self.session_data['scraped_videos']:
            print("❌ No video data available. Please scrape videos first.")
            return
        
        print("\n🖼️  THUMBNAIL ANALYSIS")
        print(f"Available videos: {len(self.session_data['scraped_videos'])}")
        
        download_thumbnails = input("Download and analyze actual thumbnails? (y/n, default n): ").strip().lower() == 'y'
        
        if download_thumbnails:
            print("⚠️  Note: Downloading thumbnails may take time and storage space.")
        
        try:
            max_thumbnails = int(input("Maximum thumbnails to analyze (default 20): ").strip() or "20")
        except ValueError:
            max_thumbnails = 20
        
        print(f"\n🔍 Analyzing thumbnails...")
        
        try:
            analysis = self.thumbnail_analyzer.analyze_thumbnails(
                self.session_data['scraped_videos'], 
                download_thumbnails, 
                max_thumbnails
            )
            
            # Store analysis result
            self.database.store_analysis_result(
                'thumbnail', 
                f"thumbnail_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                analysis
            )
            
            self.session_data['analysis_results']['thumbnail_analysis'] = analysis
            
            # Display key results
            self.display_thumbnail_analysis_summary(analysis)
            
        except Exception as e:
            print(f"❌ Error analyzing thumbnails: {e}")
    
    def create_visualizations_menu(self):
        """Handle visualization creation."""
        if not self.session_data['analysis_results']:
            print("❌ No analysis results available. Please run some analysis first.")
            return
        
        print("\n📈 VISUALIZATION OPTIONS")
        print("1. Channel comparison chart")
        print("2. Sentiment analysis chart")
        print("3. Trending analysis chart")
        print("4. Interactive dashboard")
        print("5. Performance report")
        print("6. Summary infographic")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.create_channel_comparison_chart()
        elif choice == '2':
            self.create_sentiment_chart()
        elif choice == '3':
            self.create_trending_chart()
        elif choice == '4':
            self.create_interactive_dashboard()
        elif choice == '5':
            self.create_performance_report()
        elif choice == '6':
            self.create_summary_infographic()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def create_channel_comparison_chart(self):
        """Create channel comparison chart."""
        if 'channel_comparison' not in self.session_data['analysis_results']:
            print("❌ No channel comparison data available.")
            return
        
        print("\n📊 Creating channel comparison chart...")
        
        try:
            # Convert comparison data to format expected by visualizer
            comparison = self.session_data['analysis_results']['channel_comparison']
            
            # This would need to be adapted based on the actual data structure
            # For now, use scraped channels data
            channel_analyses = []
            for channel in self.session_data['scraped_channels']:
                # Create mock analysis structure for visualization
                analysis = {
                    'channel_info': {'name': channel['name']},
                    'performance_metrics': {'average_views': 100000, 'total_views': 1000000},
                    'engagement_analysis': {'subscriber_count': 50000, 'engagement_rate': 5.0, 'consistency_score': 75}
                }
                channel_analyses.append(analysis)
            
            chart_path = self.visualizer.create_channel_comparison_chart(channel_analyses)
            print(f"✅ Chart saved to: {chart_path}")
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
    
    def create_sentiment_chart(self):
        """Create sentiment analysis chart."""
        if 'sentiment_analysis' not in self.session_data['analysis_results']:
            print("❌ No sentiment analysis data available.")
            return
        
        print("\n💭 Creating sentiment analysis chart...")
        
        try:
            analysis = self.session_data['analysis_results']['sentiment_analysis']
            chart_path = self.visualizer.create_sentiment_analysis_chart(analysis)
            print(f"✅ Chart saved to: {chart_path}")
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
    
    def create_trending_chart(self):
        """Create trending analysis chart."""
        if 'trending_analysis' not in self.session_data['analysis_results']:
            print("❌ No trending analysis data available.")
            return
        
        print("\n🔥 Creating trending analysis chart...")
        
        try:
            analysis = self.session_data['analysis_results']['trending_analysis']
            chart_path = self.visualizer.create_trending_analysis_chart(analysis)
            print(f"✅ Chart saved to: {chart_path}")
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
    
    def create_interactive_dashboard(self):
        """Create interactive dashboard."""
        print("\n🌐 Creating interactive dashboard...")
        
        try:
            # Combine all analysis data
            combined_data = {
                'channels': self.session_data.get('scraped_channels', []),
                'sentiment': self.session_data['analysis_results'].get('sentiment_analysis', {}),
                'trending': self.session_data['analysis_results'].get('trending_analysis', {})
            }
            
            dashboard_path = self.visualizer.create_interactive_dashboard(combined_data)
            print(f"✅ Interactive dashboard saved to: {dashboard_path}")
            print("🌐 Open the HTML file in your browser to view the dashboard")
            
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
    
    def create_performance_report(self):
        """Create comprehensive performance report."""
        print("\n📄 Creating performance report...")
        
        try:
            # Combine all analysis data
            combined_data = {
                'channels': self.session_data.get('scraped_channels', []),
                'sentiment': self.session_data['analysis_results'].get('sentiment_analysis', {}),
                'trending': self.session_data['analysis_results'].get('trending_analysis', {})
            }
            
            report_path = self.visualizer.create_performance_report(combined_data)
            print(f"✅ Performance report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Error creating report: {e}")
    
    def create_summary_infographic(self):
        """Create summary infographic."""
        print("\n📋 Creating summary infographic...")
        
        try:
            # Combine all analysis data
            combined_data = {
                'channels': self.session_data.get('scraped_channels', []),
                'sentiment': self.session_data['analysis_results'].get('sentiment_analysis', {}),
                'trending': self.session_data['analysis_results'].get('trending_analysis', {})
            }
            
            infographic_path = self.visualizer.create_summary_infographic(combined_data)
            print(f"✅ Summary infographic saved to: {infographic_path}")
            
        except Exception as e:
            print(f"❌ Error creating infographic: {e}")
    
    def database_operations_menu(self):
        """Handle database operations."""
        print("\n💾 DATABASE OPERATIONS")
        print("1. View database statistics")
        print("2. Export data to CSV")
        print("3. Backup database")
        print("4. Clean up old data")
        print("5. View recent analysis results")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.view_database_stats()
        elif choice == '2':
            self.export_data_to_csv()
        elif choice == '3':
            self.backup_database()
        elif choice == '4':
            self.cleanup_old_data()
        elif choice == '5':
            self.view_recent_analysis()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def view_database_stats(self):
        """View database statistics."""
        print("\n📊 DATABASE STATISTICS")
        
        try:
            stats = self.database.get_database_stats()
            
            print(f"Videos: {stats.get('videos_count', 0):,}")
            print(f"Channels: {stats.get('channels_count', 0):,}")
            print(f"Comments: {stats.get('comments_count', 0):,}")
            print(f"Analysis Results: {stats.get('analysis_results_count', 0):,}")
            print(f"Playlists: {stats.get('playlists_count', 0):,}")
            print(f"Trending Snapshots: {stats.get('trending_snapshots_count', 0):,}")
            print(f"\nTotal Video Views: {stats.get('total_video_views', 0):,}")
            print(f"Total Channel Subscribers: {stats.get('total_channel_subscribers', 0):,}")
            print(f"Database Size: {stats.get('database_size_mb', 0)} MB")
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
    
    def export_data_to_csv(self):
        """Export database tables to CSV."""
        print("\n📤 EXPORT TO CSV")
        print("1. Videos")
        print("2. Channels")
        print("3. Comments")
        print("4. Analysis Results")
        print("5. Playlists")
        
        table_map = {
            '1': 'videos',
            '2': 'channels', 
            '3': 'comments',
            '4': 'analysis_results',
            '5': 'playlists'
        }
        
        choice = input("Select table to export: ").strip()
        table_name = table_map.get(choice)
        
        if not table_name:
            print("❌ Invalid selection!")
            return
        
        output_path = f"{table_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            success = self.database.export_to_csv(table_name, output_path)
            if success:
                print(f"✅ Data exported to: {output_path}")
            else:
                print("❌ Export failed!")
                
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def backup_database(self):
        """Create database backup."""
        backup_path = f"youtube_insights_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        try:
            success = self.database.backup_database(backup_path)
            if success:
                print(f"✅ Database backed up to: {backup_path}")
            else:
                print("❌ Backup failed!")
                
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
    
    def cleanup_old_data(self):
        """Clean up old database data."""
        try:
            days_old = int(input("Remove data older than how many days? (default 30): ").strip() or "30")
        except ValueError:
            days_old = 30
        
        print(f"\n🧹 Cleaning up data older than {days_old} days...")
        
        try:
            deleted_counts = self.database.cleanup_old_data(days_old)
            
            if deleted_counts:
                print("✅ Cleanup completed:")
                for table, count in deleted_counts.items():
                    print(f"  {table}: {count} records deleted")
            else:
                print("❌ Cleanup failed!")
                
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def view_recent_analysis(self):
        """View recent analysis results."""
        print("\n📋 RECENT ANALYSIS RESULTS")
        
        try:
            results = self.database.get_analysis_results(limit=10)
            
            if results:
                for result in results:
                    print(f"\n{result['analysis_type'].upper()}: {result['analysis_name']}")
                    print(f"Created: {result['created_at']}")
                    print(f"ID: {result['id']}")
            else:
                print("No analysis results found.")
                
        except Exception as e:
            print(f"❌ Error retrieving analysis results: {e}")
    
    def settings_menu(self):
        """Handle settings and configuration."""
        print("\n⚙️  SETTINGS & CONFIGURATION")
        print("1. Reconfigure scraper")
        print("2. Clear session data")
        print("3. View created visualizations")
        print("4. Clear visualization cache")
        print("5. Clear thumbnail cache")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.setup_scraper()
        elif choice == '2':
            self.clear_session_data()
        elif choice == '3':
            self.view_created_visualizations()
        elif choice == '4':
            self.clear_visualization_cache()
        elif choice == '5':
            self.clear_thumbnail_cache()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def clear_session_data(self):
        """Clear current session data."""
        confirm = input("Are you sure you want to clear all session data? (y/n): ").strip().lower()
        
        if confirm == 'y':
            self.session_data = {
                'scraped_videos': [],
                'scraped_channels': [],
                'scraped_comments': [],
                'analysis_results': {}
            }
            print("✅ Session data cleared!")
        else:
            print("❌ Operation cancelled.")
    
    def view_created_visualizations(self):
        """View list of created visualizations."""
        visualizations = self.visualizer.get_created_visualizations()
        
        if visualizations:
            print(f"\n📈 CREATED VISUALIZATIONS ({len(visualizations)})")
            for i, viz_path in enumerate(visualizations, 1):
                print(f"{i}. {os.path.basename(viz_path)}")
        else:
            print("No visualizations created yet.")
    
    def clear_visualization_cache(self):
        """Clear visualization cache."""
        try:
            cleaned_files = self.visualizer.cleanup_old_files(days_old=0)  # Clear all
            print(f"✅ Cleared {len(cleaned_files)} visualization files")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    
    def clear_thumbnail_cache(self):
        """Clear thumbnail cache."""
        try:
            self.thumbnail_analyzer.clear_cache()
            print("✅ Thumbnail cache cleared!")
        except Exception as e:
            print(f"❌ Error clearing thumbnail cache: {e}")
    
    def generate_reports_menu(self):
        """Handle report generation."""
        print("\n📄 REPORT GENERATION")
        print("1. Session summary report")
        print("2. Database summary report")
        print("3. Export session data to JSON")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.generate_session_summary()
        elif choice == '2':
            self.generate_database_summary()
        elif choice == '3':
            self.export_session_to_json()
        elif choice == '0':
            return
        else:
            print("❌ Invalid option!")
    
    def generate_session_summary(self):
        """Generate session summary report."""
        print("\n📋 SESSION SUMMARY REPORT")
        print("=" * 50)
        
        # Data summary
        print(f"Videos scraped: {len(self.session_data['scraped_videos'])}")
        print(f"Channels scraped: {len(self.session_data['scraped_channels'])}")
        print(f"Comments scraped: {len(self.session_data['scraped_comments'])}")
        print(f"Analysis results: {len(self.session_data['analysis_results'])}")
        
        # Analysis summary
        if self.session_data['analysis_results']:
            print("\nCompleted Analyses:")
            for analysis_type in self.session_data['analysis_results']:
                print(f"  • {analysis_type.replace('_', ' ').title()}")
        
        # Video statistics
        if self.session_data['scraped_videos']:
            total_views = sum(self.data_processor.clean_view_count(v.get('views', '0')) 
                            for v in self.session_data['scraped_videos'])
            print(f"\nTotal views across scraped videos: {total_views:,}")
        
        # Channel statistics
        if self.session_data['scraped_channels']:
            total_subscribers = sum(self.data_processor.clean_subscriber_count(c.get('subscribers', '0'))
                                  for c in self.session_data['scraped_channels'])
            print(f"Total subscribers across scraped channels: {total_subscribers:,}")
        
        print("=" * 50)
    
    def generate_database_summary(self):
        """Generate database summary report."""
        print("\n💾 DATABASE SUMMARY REPORT")
        print("=" * 50)
        
        try:
            stats = self.database.get_database_stats()
            
            print("Database Contents:")
            print(f"  Videos: {stats.get('videos_count', 0):,}")
            print(f"  Channels: {stats.get('channels_count', 0):,}")
            print(f"  Comments: {stats.get('comments_count', 0):,}")
            print(f"  Analysis Results: {stats.get('analysis_results_count', 0):,}")
            print(f"  Playlists: {stats.get('playlists_count', 0):,}")
            print(f"  Trending Snapshots: {stats.get('trending_snapshots_count', 0):,}")
            
            print(f"\nAggregate Statistics:")
            print(f"  Total Video Views: {stats.get('total_video_views', 0):,}")
            print(f"  Total Channel Subscribers: {stats.get('total_channel_subscribers', 0):,}")
            print(f"  Unique Channels: {stats.get('unique_channels_in_videos', 0):,}")
            
            print(f"\nDatabase File:")
            print(f"  Size: {stats.get('database_size_mb', 0)} MB")
            print(f"  Location: {self.database.db_path}")
            
        except Exception as e:
            print(f"❌ Error generating database summary: {e}")
        
        print("=" * 50)
    
    def export_session_to_json(self):
        """Export session data to JSON file."""
        output_path = f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Session data exported to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error exporting session data: {e}")
    
    # Display helper methods
    def display_channel_analysis_summary(self, analysis: Dict):
        """Display channel analysis summary."""
        print("\n📊 CHANNEL ANALYSIS SUMMARY")
        print("=" * 40)
        
        channel_info = analysis.get('channel_info', {})
        performance = analysis.get('performance_metrics', {})
        engagement = analysis.get('engagement_analysis', {})
        
        print(f"Channel: {channel_info.get('name', 'Unknown')}")
        print(f"Subscribers: {engagement.get('subscriber_count', 0):,}")
        print(f"Videos Analyzed: {channel_info.get('total_videos_analyzed', 0)}")
        print(f"Average Views: {performance.get('average_views', 0):,}")
        print(f"Engagement Rate: {engagement.get('engagement_rate', 0):.2f}%")
        print(f"Consistency Score: {engagement.get('consistency_score', 0):.1f}")
        
        # Top performing video
        top_video = performance.get('top_performing_video', {})
        if top_video:
            print(f"\nTop Video: {top_video.get('title', 'Unknown')[:50]}...")
            print(f"Views: {top_video.get('views', 0):,}")
        
        print("=" * 40)
    
    def display_channel_comparison_summary(self, comparison: Dict):
        """Display channel comparison summary."""
        print("\n📊 CHANNEL COMPARISON SUMMARY")
        print("=" * 40)
        
        overview = comparison.get('comparison_overview', {})
        performance = comparison.get('performance_comparison', {})
        
        print(f"Channels Compared: {overview.get('channels_compared', 0)}")
        print(f"Comparison Date: {overview.get('comparison_date', 'Unknown')}")
        
        # Performance rankings
        subscriber_ranking = performance.get('subscriber_ranking', [])
        if subscriber_ranking:
            print("\nSubscriber Rankings:")
            for i, channel in enumerate(subscriber_ranking[:3], 1):
                print(f"{i}. {channel.get('channel', 'Unknown')}: {channel.get('subscribers', 0):,}")
        
        view_performance = performance.get('view_performance', [])
        if view_performance:
            print("\nView Performance Rankings:")
            for i, channel in enumerate(view_performance[:3], 1):
                print(f"{i}. {channel.get('channel', 'Unknown')}: {channel.get('avg_views', 0):,} avg views")
        
        print("=" * 40)
    
    def display_sentiment_analysis_summary(self, analysis: Dict):
        """Display sentiment analysis summary."""
        print("\n💭 SENTIMENT ANALYSIS SUMMARY")
        print("=" * 40)
        
        overview = analysis.get('overview', {})
        
        print(f"Comments Analyzed: {overview.get('total_comments_analyzed', 0)}")
        print(f"Overall Sentiment: {overview.get('overall_classification', 'Unknown')}")
        print(f"Sentiment Score: {overview.get('overall_sentiment_score', 0):.3f}")
        
        # Sentiment distribution
        sentiment_dist = overview.get('sentiment_distribution', {})
        if sentiment_dist:
            print("\nSentiment Distribution:")
            for sentiment, count in sentiment_dist.items():
                percentage = overview.get('sentiment_percentages', {}).get(sentiment, 0)
                print(f"  {sentiment}: {count} ({percentage}%)")
        
        # Emotions
        emotions = analysis.get('emotion_analysis', {})
        dominant_emotion = emotions.get('dominant_emotion', 'None')
        if dominant_emotion != 'None':
            print(f"\nDominant Emotion: {dominant_emotion}")
        
        print("=" * 40)
    
    def display_trending_analysis_summary(self, analysis: Dict):
        """Display trending analysis summary."""
        print("\n🔥 TRENDING ANALYSIS SUMMARY")
        print("=" * 40)
        
        overview = analysis.get('overview', {})
        content_patterns = analysis.get('content_patterns', {})
        
        print(f"Trending Videos: {overview.get('total_trending_videos', 0)}")
        print(f"Unique Channels: {overview.get('unique_channels', 0)}")
        print(f"Average Views: {overview.get('average_views', 0):,}")
        print(f"Total Views: {overview.get('total_views', 0):,}")
        
        # Duration patterns
        duration_patterns = content_patterns.get('duration_patterns', {})
        avg_duration = duration_patterns.get('average_duration_minutes', 0)
        optimal_duration = duration_patterns.get('optimal_trending_duration', 'Unknown')
        
        print(f"\nAverage Duration: {avg_duration:.1f} minutes")
        print(f"Optimal Duration Category: {optimal_duration}")
        
        # Top performing video
        top_video = overview.get('top_performing_video', {})
        if top_video:
            print(f"\nTop Trending Video:")
            print(f"  {top_video.get('title', 'Unknown')[:50]}...")
            print(f"  Channel: {top_video.get('channel', 'Unknown')}")
            print(f"  Views: {top_video.get('views', 0):,}")
        
        print("=" * 40)
    
    def display_thumbnail_analysis_summary(self, analysis: Dict):
        """Display thumbnail analysis summary."""
        print("\n🖼️  THUMBNAIL ANALYSIS SUMMARY")
        print("=" * 40)
        
        overview = analysis.get('overview', {})
        
        print(f"Videos Analyzed: {overview.get('total_videos', 0)}")
        print(f"Thumbnails Processed: {overview.get('thumbnails_analyzed', 0)}")
        
        # Color analysis
        color_analysis = analysis.get('color_analysis', {})
        if color_analysis:
            temp_dist = color_analysis.get('color_temperature_distribution', {})
            if temp_dist:
                most_common_temp = max(temp_dist, key=temp_dist.get)
                print(f"Most Common Color Temperature: {most_common_temp}")
            
            brightness_dist = color_analysis.get('brightness_distribution', {})
            if brightness_dist:
                most_common_brightness = max(brightness_dist, key=brightness_dist.get)
                print(f"Most Common Brightness: {most_common_brightness}")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print("\nKey Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec}")
        
        print("=" * 40)
    
    def run(self):
        """Run the main application loop."""
        self.display_welcome()
        
        while True:
            try:
                self.display_main_menu()
                choice = input("\nSelect option: ").strip()
                
                if choice == '1':
                    self.scrape_data_menu()
                elif choice == '2':
                    self.analyze_channels_menu()
                elif choice == '3':
                    self.analyze_sentiment_menu()
                elif choice == '4':
                    self.analyze_trending_menu()
                elif choice == '5':
                    self.generate_playlists_menu()
                elif choice == '6':
                    self.analyze_thumbnails_menu()
                elif choice == '7':
                    self.create_visualizations_menu()
                elif choice == '8':
                    self.database_operations_menu()
                elif choice == '9':
                    self.settings_menu()
                elif choice == '10':
                    self.generate_reports_menu()
                elif choice == '0':
                    print("\n👋 Thank you for using YouTube Content Insights!")
                    break
                else:
                    print("❌ Invalid option! Please try again.")
                
                # Pause before showing menu again
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                logger.error(f"Application error: {e}", exc_info=True)
                input("Press Enter to continue...")
        
        # Cleanup
        if self.scraper:
            self.scraper.close()
        
        self.database.close()


def main():
    """Main entry point."""
    try:
        app = YouTubeContentInsights()
        app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal application error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
