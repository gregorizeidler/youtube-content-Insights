"""
YouTube Content Insights - Streamlit Web Interface

A modern web interface for YouTube content analysis and insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
from datetime import datetime
import time
import base64
from io import BytesIO

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

# Page configuration
st.set_page_config(
    page_title="YouTube Content Insights",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'scraper' not in st.session_state:
        st.session_state.scraper = None
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = {
            'videos': [],
            'channels': [],
            'comments': [],
            'trending': []
        }
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'database' not in st.session_state:
        st.session_state.database = DatabaseManager()

def main_header():
    """Display main header."""
    st.markdown('<h1 class="main-header">🎥 YouTube Content Insights</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Comprehensive YouTube content analysis and optimization platform
        </p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.title("🎯 Navigation")
    
    pages = {
        "🏠 Home": "home",
        "🔍 Data Scraping": "scraping",
        "📊 Channel Analysis": "channel_analysis",
        "💭 Sentiment Analysis": "sentiment_analysis",
        "🔥 Trending Analysis": "trending_analysis",
        "🎵 Playlist Generator": "playlist_generator",
        "🖼️ Thumbnail Analysis": "thumbnail_analysis",
        "📈 Visualizations": "visualizations",
        "💾 Database": "database",
        "⚙️ Settings": "settings"
    }
    
    selected_page = st.sidebar.selectbox("Choose a feature:", list(pages.keys()))
    
    # Quick stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    
    if st.session_state.scraped_data['videos']:
        st.sidebar.metric("Videos Scraped", len(st.session_state.scraped_data['videos']))
    if st.session_state.scraped_data['channels']:
        st.sidebar.metric("Channels Analyzed", len(st.session_state.scraped_data['channels']))
    if st.session_state.scraped_data['comments']:
        st.sidebar.metric("Comments Analyzed", len(st.session_state.scraped_data['comments']))
    
    return pages[selected_page]

def home_page():
    """Display home page."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Data Scraping
        - Search and extract video data
        - Channel information analysis
        - Trending content monitoring
        - Comment collection
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Advanced Analytics
        - Performance metrics
        - Sentiment analysis
        - Trend identification
        - Competitive analysis
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Smart Features
        - Playlist generation
        - Thumbnail analysis
        - Data visualization
        - Report generation
        """)
    
    st.markdown("---")
    
    # Quick start section
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Start with Video Search", use_container_width=True):
            st.session_state.current_page = "scraping"
            st.rerun()
    
    with col2:
        if st.button("📊 View Database Stats", use_container_width=True):
            st.session_state.current_page = "database"
            st.rerun()
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    
    try:
        db_stats = st.session_state.database.get_database_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Videos", f"{db_stats.get('videos_count', 0):,}")
        with col2:
            st.metric("Total Channels", f"{db_stats.get('channels_count', 0):,}")
        with col3:
            st.metric("Total Comments", f"{db_stats.get('comments_count', 0):,}")
        with col4:
            st.metric("Database Size", f"{db_stats.get('database_size_mb', 0)} MB")
            
    except Exception as e:
        st.warning(f"Could not load database stats: {e}")

def scraping_page():
    """Data scraping interface."""
    st.header("🔍 Data Scraping")
    
    # Scraper setup
    if not st.session_state.scraper:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Scraper Setup Required</strong><br>
            Please configure your browser and ChromeDriver paths to start scraping.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔧 Configure Scraper", expanded=True):
            browser_path = st.text_input(
                "Browser Path",
                placeholder="e.g., /Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                help="Path to your browser executable"
            )
            
            driver_path = st.text_input(
                "ChromeDriver Path",
                placeholder="e.g., /usr/local/bin/chromedriver",
                help="Path to ChromeDriver executable"
            )
            
            headless = st.checkbox("Run in headless mode", value=True)
            
            if st.button("Setup Scraper"):
                try:
                    # Use provided paths or None for automatic detection/download
                    browser = browser_path if browser_path.strip() else None
                    driver = driver_path if driver_path.strip() else None
                    
                    st.session_state.scraper = YouTubeScraper(browser, driver, headless)
                    st.success("✅ Scraper configured successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error setting up scraper: {e}")
    
    else:
        # Scraping options
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Video Search", "📺 Channel Analysis", "🔥 Trending Videos", "💬 Comments"])
        
        with tab1:
            video_search_interface()
        
        with tab2:
            channel_analysis_interface()
        
        with tab3:
            trending_videos_interface()
        
        with tab4:
            comments_interface()

def video_search_interface():
    """Video search interface."""
    st.subheader("🔍 Search Videos")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., python programming")
    
    with col2:
        max_results = st.number_input("Max Results", min_value=1, max_value=100, value=20)
    
    if st.button("🚀 Start Scraping", disabled=not query):
        with st.spinner("Scraping videos..."):
            try:
                videos = st.session_state.scraper.search_videos(query, max_results)
                
                if videos:
                    st.session_state.scraped_data['videos'].extend(videos)
                    
                    # Store in database
                    stored_count = st.session_state.database.store_videos(videos)
                    
                    st.success(f"✅ Successfully scraped {len(videos)} videos!")
                    st.info(f"💾 Stored {stored_count} videos in database")
                    
                    # Display results
                    display_scraped_videos(videos[:10])  # Show first 10
                    
                else:
                    st.warning("No videos found for this query")
                    
            except Exception as e:
                st.error(f"❌ Error scraping videos: {e}")

def channel_analysis_interface():
    """Channel analysis interface."""
    st.subheader("📺 Channel Analysis")
    
    channel_url = st.text_input(
        "YouTube Channel URL",
        placeholder="https://www.youtube.com/@channelname"
    )
    
    if st.button("🔍 Analyze Channel", disabled=not channel_url):
        with st.spinner("Analyzing channel..."):
            try:
                channel_data = st.session_state.scraper.scrape_channel_info(channel_url)
                
                if channel_data:
                    st.session_state.scraped_data['channels'].append(channel_data)
                    
                    # Store in database
                    processed_channel = {
                        'name': channel_data['name'],
                        'url': channel_data['url'],
                        'subscribers': DataProcessor().clean_subscriber_count(channel_data['subscribers']),
                        'total_videos_analyzed': channel_data['total_videos_scraped'],
                        'scraped_at': channel_data['scraped_at']
                    }
                    
                    st.session_state.database.store_channels([processed_channel])
                    
                    if channel_data.get('recent_videos'):
                        st.session_state.database.store_videos(channel_data['recent_videos'])
                    
                    st.success(f"✅ Successfully analyzed channel: {channel_data['name']}")
                    
                    # Display channel info
                    display_channel_info(channel_data)
                    
                else:
                    st.error("Failed to analyze channel")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing channel: {e}")

def trending_videos_interface():
    """Trending videos interface."""
    st.subheader("🔥 Trending Videos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category = st.selectbox(
            "Category",
            ["all", "music", "gaming", "movies", "news"]
        )
    
    with col2:
        max_videos = st.number_input("Max Videos", min_value=10, max_value=50, value=30)
    
    if st.button("🔥 Get Trending Videos"):
        with st.spinner("Fetching trending videos..."):
            try:
                trending_videos = st.session_state.scraper.scrape_trending_videos(category)
                
                if trending_videos:
                    st.session_state.scraped_data['trending'].extend(trending_videos)
                    st.session_state.scraped_data['videos'].extend(trending_videos)
                    
                    # Store in database
                    stored_count = st.session_state.database.store_videos(trending_videos)
                    
                    st.success(f"✅ Successfully scraped {len(trending_videos)} trending videos!")
                    st.info(f"💾 Stored {stored_count} videos in database")
                    
                    # Display trending videos
                    display_trending_videos(trending_videos[:10])
                    
                else:
                    st.warning("No trending videos found")
                    
            except Exception as e:
                st.error(f"❌ Error fetching trending videos: {e}")

def comments_interface():
    """Comments scraping interface."""
    st.subheader("💬 Scrape Comments")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
    
    with col2:
        max_comments = st.number_input("Max Comments", min_value=10, max_value=500, value=100)
    
    if st.button("💬 Scrape Comments", disabled=not video_url):
        with st.spinner("Scraping comments..."):
            try:
                comments = st.session_state.scraper.scrape_video_comments(video_url, max_comments)
                
                if comments:
                    st.session_state.scraped_data['comments'].extend(comments)
                    
                    # Store in database
                    stored_count = st.session_state.database.store_comments(comments, video_url)
                    
                    st.success(f"✅ Successfully scraped {len(comments)} comments!")
                    st.info(f"💾 Stored {stored_count} comments in database")
                    
                    # Display sample comments
                    display_comments_sample(comments[:5])
                    
                else:
                    st.warning("No comments found")
                    
            except Exception as e:
                st.error(f"❌ Error scraping comments: {e}")

def channel_analysis_page():
    """Channel analysis page."""
    st.header("📊 Channel Analysis")
    
    if not st.session_state.scraped_data['channels']:
        st.warning("No channel data available. Please scrape channel data first.")
        return
    
    # Channel selection
    channel_names = [ch['name'] for ch in st.session_state.scraped_data['channels']]
    
    tab1, tab2 = st.tabs(["📊 Single Channel", "⚖️ Compare Channels"])
    
    with tab1:
        single_channel_analysis()
    
    with tab2:
        channel_comparison_analysis()

def single_channel_analysis():
    """Single channel analysis interface."""
    st.subheader("📊 Single Channel Analysis")
    
    channel_names = [ch['name'] for ch in st.session_state.scraped_data['channels']]
    selected_channel = st.selectbox("Select Channel", channel_names)
    
    if st.button("🔍 Analyze Channel"):
        channel_data = next(ch for ch in st.session_state.scraped_data['channels'] if ch['name'] == selected_channel)
        videos_data = channel_data.get('recent_videos', [])
        
        with st.spinner("Analyzing channel..."):
            analyzer = ChannelAnalyzer()
            analysis = analyzer.analyze_single_channel(channel_data, videos_data)
            
            st.session_state.analysis_results['channel_analysis'] = analysis
            
            # Display results
            display_channel_analysis_results(analysis)

def channel_comparison_analysis():
    """Channel comparison analysis interface."""
    st.subheader("⚖️ Compare Channels")
    
    channel_names = [ch['name'] for ch in st.session_state.scraped_data['channels']]
    
    if len(channel_names) < 2:
        st.warning("Need at least 2 channels for comparison.")
        return
    
    selected_channels = st.multiselect("Select Channels to Compare", channel_names, default=channel_names[:2])
    
    if len(selected_channels) >= 2 and st.button("⚖️ Compare Channels"):
        channels_to_compare = []
        for name in selected_channels:
            channel_data = next(ch for ch in st.session_state.scraped_data['channels'] if ch['name'] == name)
            channels_to_compare.append({
                'channel_data': channel_data,
                'videos_data': channel_data.get('recent_videos', [])
            })
        
        with st.spinner("Comparing channels..."):
            analyzer = ChannelAnalyzer()
            comparison = analyzer.compare_channels(channels_to_compare)
            
            st.session_state.analysis_results['channel_comparison'] = comparison
            
            # Display comparison results
            display_channel_comparison_results(comparison)

def sentiment_analysis_page():
    """Sentiment analysis page."""
    st.header("💭 Sentiment Analysis")
    
    if not st.session_state.scraped_data['comments']:
        st.warning("No comment data available. Please scrape video comments first.")
        return
    
    st.info(f"📊 Available comments: {len(st.session_state.scraped_data['comments'])}")
    
    video_title = st.text_input("Video Title (optional)", placeholder="Enter video title for context")
    
    if st.button("🔍 Analyze Sentiment"):
        video_info = {'title': video_title or "Unknown Video"}
        
        with st.spinner("Analyzing sentiment..."):
            analyzer = SentimentAnalyzer()
            analysis = analyzer.analyze_video_sentiment(st.session_state.scraped_data['comments'], video_info)
            
            st.session_state.analysis_results['sentiment_analysis'] = analysis
            
            # Display results
            display_sentiment_analysis_results(analysis)

def trending_analysis_page():
    """Trending analysis page."""
    st.header("🔥 Trending Analysis")
    
    trending_videos = [v for v in st.session_state.scraped_data['videos'] if v.get('trending_category')]
    
    if not trending_videos:
        st.warning("No trending video data available. Please scrape trending videos first.")
        return
    
    st.info(f"📊 Available trending videos: {len(trending_videos)}")
    
    category = trending_videos[0].get('trending_category', 'all')
    
    if st.button("🔍 Analyze Trending Content"):
        with st.spinner("Analyzing trending content..."):
            analyzer = TrendsAnalyzer()
            analysis = analyzer.analyze_trending_content(trending_videos, category)
            
            st.session_state.analysis_results['trending_analysis'] = analysis
            
            # Display results
            display_trending_analysis_results(analysis)

def playlist_generator_page():
    """Playlist generator page."""
    st.header("🎵 Playlist Generator")
    
    if not st.session_state.scraped_data['videos']:
        st.warning("No video data available. Please scrape videos first.")
        return
    
    st.info(f"📊 Available videos: {len(st.session_state.scraped_data['videos'])}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Best Performing", "🎯 Themed", "💎 Discovery", "⚖️ Balanced"])
    
    with tab1:
        best_performing_playlist_interface()
    
    with tab2:
        themed_playlist_interface()
    
    with tab3:
        discovery_playlist_interface()
    
    with tab4:
        balanced_playlist_interface()

def best_performing_playlist_interface():
    """Best performing playlist interface."""
    st.subheader("⭐ Best Performing Playlist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        criteria = st.selectbox("Criteria", ["views", "engagement", "recent", "duration"])
    
    with col2:
        max_videos = st.number_input("Max Videos", min_value=5, max_value=50, value=20)
    
    if st.button("🎵 Generate Playlist"):
        with st.spinner("Generating playlist..."):
            generator = PlaylistGenerator()
            playlist = generator.generate_best_performing_playlist(
                st.session_state.scraped_data['videos'], 
                criteria, 
                max_videos
            )
            
            if 'error' not in playlist:
                # Store playlist
                st.session_state.database.store_playlist(playlist)
                
                # Display playlist
                display_playlist_results(playlist)
            else:
                st.error(f"Error: {playlist['error']}")

def themed_playlist_interface():
    """Themed playlist interface."""
    st.subheader("🎯 Themed Playlist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        keywords_input = st.text_input("Theme Keywords (comma-separated)", placeholder="python, programming, tutorial")
    
    with col2:
        max_videos = st.number_input("Max Videos", min_value=5, max_value=30, value=15)
    
    if st.button("🎵 Generate Themed Playlist") and keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',')]
        
        with st.spinner("Generating themed playlist..."):
            generator = PlaylistGenerator()
            playlist = generator.generate_themed_playlist(
                st.session_state.scraped_data['videos'], 
                keywords, 
                max_videos
            )
            
            if 'error' not in playlist:
                # Store playlist
                st.session_state.database.store_playlist(playlist)
                
                # Display playlist
                display_playlist_results(playlist)
            else:
                st.error(f"Error: {playlist['error']}")

def discovery_playlist_interface():
    """Discovery playlist interface."""
    st.subheader("💎 Discovery Playlist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exclude_popular = st.checkbox("Exclude popular videos", value=True)
    
    with col2:
        max_videos = st.number_input("Max Videos", min_value=10, max_value=50, value=25)
    
    if st.button("🎵 Generate Discovery Playlist"):
        with st.spinner("Generating discovery playlist..."):
            generator = PlaylistGenerator()
            playlist = generator.generate_discovery_playlist(
                st.session_state.scraped_data['videos'], 
                exclude_popular, 
                max_videos
            )
            
            if 'error' not in playlist:
                # Store playlist
                st.session_state.database.store_playlist(playlist)
                
                # Display playlist
                display_playlist_results(playlist)
            else:
                st.error(f"Error: {playlist['error']}")

def balanced_playlist_interface():
    """Balanced playlist interface."""
    st.subheader("⚖️ Balanced Playlist")
    
    duration_minutes = st.number_input("Target Duration (minutes)", min_value=30, max_value=180, value=60)
    
    if st.button("🎵 Generate Balanced Playlist"):
        with st.spinner("Generating balanced playlist..."):
            generator = PlaylistGenerator()
            playlist = generator.generate_balanced_playlist(
                st.session_state.scraped_data['videos'], 
                duration_minutes * 60
            )
            
            if 'error' not in playlist:
                # Store playlist
                st.session_state.database.store_playlist(playlist)
                
                # Display playlist
                display_playlist_results(playlist)
            else:
                st.error(f"Error: {playlist['error']}")

def thumbnail_analysis_page():
    """Thumbnail analysis page."""
    st.header("🖼️ Thumbnail Analysis")
    
    if not st.session_state.scraped_data['videos']:
        st.warning("No video data available. Please scrape videos first.")
        return
    
    st.info(f"📊 Available videos: {len(st.session_state.scraped_data['videos'])}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        download_thumbnails = st.checkbox("Download and analyze thumbnails", value=False)
        st.caption("⚠️ This may take time and storage space")
    
    with col2:
        max_thumbnails = st.number_input("Max Thumbnails", min_value=5, max_value=50, value=20)
    
    if st.button("🔍 Analyze Thumbnails"):
        with st.spinner("Analyzing thumbnails..."):
            analyzer = ThumbnailAnalyzer()
            analysis = analyzer.analyze_thumbnails(
                st.session_state.scraped_data['videos'], 
                download_thumbnails, 
                max_thumbnails
            )
            
            st.session_state.analysis_results['thumbnail_analysis'] = analysis
            
            # Display results
            display_thumbnail_analysis_results(analysis)

def visualizations_page():
    """Visualizations page."""
    st.header("📈 Data Visualizations")
    
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please run some analysis first.")
        return
    
    # Available visualizations
    viz_options = []
    
    if 'channel_analysis' in st.session_state.analysis_results or 'channel_comparison' in st.session_state.analysis_results:
        viz_options.append("📊 Channel Performance")
    
    if 'sentiment_analysis' in st.session_state.analysis_results:
        viz_options.append("💭 Sentiment Analysis")
    
    if 'trending_analysis' in st.session_state.analysis_results:
        viz_options.append("🔥 Trending Analysis")
    
    if not viz_options:
        st.warning("No visualizations available. Run some analysis first.")
        return
    
    selected_viz = st.selectbox("Select Visualization", viz_options)
    
    if selected_viz == "📊 Channel Performance":
        create_channel_performance_viz()
    elif selected_viz == "💭 Sentiment Analysis":
        create_sentiment_analysis_viz()
    elif selected_viz == "🔥 Trending Analysis":
        create_trending_analysis_viz()

def database_page():
    """Database management page."""
    st.header("💾 Database Management")
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📤 Export", "🧹 Maintenance"])
    
    with tab1:
        database_statistics()
    
    with tab2:
        database_export()
    
    with tab3:
        database_maintenance()

def settings_page():
    """Settings and configuration page."""
    st.header("⚙️ Settings & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Scraper", "🎨 Interface", "📊 Data"])
    
    with tab1:
        scraper_settings()
    
    with tab2:
        interface_settings()
    
    with tab3:
        data_settings()

# Display helper functions
def display_scraped_videos(videos):
    """Display scraped videos in a table."""
    st.subheader("📋 Scraped Videos")
    
    df = pd.DataFrame(videos)
    if not df.empty:
        # Select relevant columns
        display_columns = ['title', 'channel_name', 'views', 'duration', 'upload_time']
        available_columns = [col for col in display_columns if col in df.columns]
        
        st.dataframe(df[available_columns], use_container_width=True)

def display_channel_info(channel_data):
    """Display channel information."""
    st.subheader("📺 Channel Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Channel Name", channel_data['name'])
    
    with col2:
        st.metric("Subscribers", channel_data['subscribers'])
    
    with col3:
        st.metric("Videos Analyzed", channel_data['total_videos_scraped'])

def display_trending_videos(videos):
    """Display trending videos."""
    st.subheader("🔥 Trending Videos")
    
    for i, video in enumerate(videos, 1):
        with st.expander(f"{i}. {video['title'][:60]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Channel:** {video['channel_name']}")
                st.write(f"**Views:** {video['views']}")
            
            with col2:
                st.write(f"**Duration:** {video['duration']}")
                st.write(f"**Upload Time:** {video['upload_time']}")

def display_comments_sample(comments):
    """Display sample comments."""
    st.subheader("💬 Sample Comments")
    
    for i, comment in enumerate(comments, 1):
        with st.expander(f"Comment {i} - {comment['author']}"):
            st.write(comment['text'])
            st.caption(f"Likes: {comment['likes']} | Time: {comment['time']}")

def display_channel_analysis_results(analysis):
    """Display channel analysis results."""
    st.subheader("📊 Channel Analysis Results")
    
    # Channel info
    channel_info = analysis['channel_info']
    st.markdown(f"**Channel:** {channel_info['name']}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    performance = analysis['performance_metrics']
    engagement = analysis['engagement_analysis']
    
    with col1:
        st.metric("Avg Views", f"{performance['average_views']:,}")
    
    with col2:
        st.metric("Total Views", f"{performance['total_views']:,}")
    
    with col3:
        st.metric("Engagement Rate", f"{engagement['engagement_rate']:.2f}%")
    
    with col4:
        st.metric("Consistency Score", f"{engagement['consistency_score']:.1f}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = analysis.get('recommendations', [])
    for rec in recommendations:
        st.write(f"• {rec}")

def display_channel_comparison_results(comparison):
    """Display channel comparison results."""
    st.subheader("⚖️ Channel Comparison Results")
    
    # Performance comparison
    performance = comparison.get('performance_comparison', {})
    
    if 'subscriber_ranking' in performance:
        st.subheader("👥 Subscriber Rankings")
        ranking_df = pd.DataFrame(performance['subscriber_ranking'])
        st.dataframe(ranking_df, use_container_width=True)
    
    if 'view_performance' in performance:
        st.subheader("👀 View Performance")
        view_df = pd.DataFrame(performance['view_performance'])
        st.dataframe(view_df, use_container_width=True)

def display_sentiment_analysis_results(analysis):
    """Display sentiment analysis results."""
    st.subheader("💭 Sentiment Analysis Results")
    
    overview = analysis['overview']
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Comments Analyzed", overview['total_comments_analyzed'])
    
    with col2:
        st.metric("Overall Sentiment", overview['overall_classification'])
    
    with col3:
        st.metric("Sentiment Score", f"{overview['overall_sentiment_score']:.3f}")
    
    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_dist = overview['sentiment_distribution']
    
    fig = px.pie(
        values=list(sentiment_dist.values()),
        names=list(sentiment_dist.keys()),
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_trending_analysis_results(analysis):
    """Display trending analysis results."""
    st.subheader("🔥 Trending Analysis Results")
    
    overview = analysis['overview']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Trending Videos", overview['total_trending_videos'])
    
    with col2:
        st.metric("Unique Channels", overview['unique_channels'])
    
    with col3:
        st.metric("Avg Views", f"{overview['average_views']:,}")
    
    with col4:
        st.metric("Total Views", f"{overview['total_views']:,}")
    
    # Duration analysis
    content_patterns = analysis['content_patterns']
    duration_patterns = content_patterns['duration_patterns']
    
    st.subheader("⏱️ Duration Analysis")
    duration_dist = duration_patterns['duration_distribution']
    
    fig = px.bar(
        x=list(duration_dist.keys()),
        y=list(duration_dist.values()),
        title="Video Duration Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_playlist_results(playlist):
    """Display playlist generation results."""
    st.subheader("🎵 Generated Playlist")
    
    # Playlist info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Title", playlist['title'])
    
    with col2:
        st.metric("Videos", playlist['video_count'])
    
    with col3:
        st.metric("Total Duration", playlist['total_duration_formatted'])
    
    # Videos in playlist
    st.subheader("📋 Playlist Videos")
    videos_df = pd.DataFrame(playlist['videos'])
    
    if not videos_df.empty:
        display_columns = ['position', 'title', 'channel', 'duration', 'views_formatted']
        available_columns = [col for col in display_columns if col in videos_df.columns]
        st.dataframe(videos_df[available_columns], use_container_width=True)

def display_thumbnail_analysis_results(analysis):
    """Display thumbnail analysis results."""
    st.subheader("🖼️ Thumbnail Analysis Results")
    
    overview = analysis['overview']
    
    # Key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Videos Analyzed", overview['total_videos'])
    
    with col2:
        st.metric("Thumbnails Processed", overview['thumbnails_analyzed'])
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")

def create_channel_performance_viz():
    """Create channel performance visualization."""
    st.subheader("📊 Channel Performance Visualization")
    
    # This would create interactive Plotly charts
    st.info("Channel performance charts would be displayed here")

def create_sentiment_analysis_viz():
    """Create sentiment analysis visualization."""
    st.subheader("💭 Sentiment Analysis Visualization")
    
    if 'sentiment_analysis' in st.session_state.analysis_results:
        analysis = st.session_state.analysis_results['sentiment_analysis']
        overview = analysis['overview']
        
        # Sentiment pie chart
        sentiment_dist = overview['sentiment_distribution']
        
        fig = px.pie(
            values=list(sentiment_dist.values()),
            names=list(sentiment_dist.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#95a5a6'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def create_trending_analysis_viz():
    """Create trending analysis visualization."""
    st.subheader("🔥 Trending Analysis Visualization")
    
    if 'trending_analysis' in st.session_state.analysis_results:
        analysis = st.session_state.analysis_results['trending_analysis']
        
        # Duration distribution
        content_patterns = analysis['content_patterns']
        duration_patterns = content_patterns['duration_patterns']
        duration_dist = duration_patterns['duration_distribution']
        
        fig = px.bar(
            x=list(duration_dist.keys()),
            y=list(duration_dist.values()),
            title="Trending Video Duration Distribution",
            color=list(duration_dist.values()),
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def database_statistics():
    """Display database statistics."""
    st.subheader("📊 Database Statistics")
    
    try:
        stats = st.session_state.database.get_database_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Videos", f"{stats.get('videos_count', 0):,}")
        
        with col2:
            st.metric("Channels", f"{stats.get('channels_count', 0):,}")
        
        with col3:
            st.metric("Comments", f"{stats.get('comments_count', 0):,}")
        
        with col4:
            st.metric("Database Size", f"{stats.get('database_size_mb', 0)} MB")
        
        # Additional stats
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Video Views", f"{stats.get('total_video_views', 0):,}")
        
        with col2:
            st.metric("Total Subscribers", f"{stats.get('total_channel_subscribers', 0):,}")
            
    except Exception as e:
        st.error(f"Error loading database stats: {e}")

def database_export():
    """Database export interface."""
    st.subheader("📤 Export Data")
    
    export_options = ["videos", "channels", "comments", "analysis_results", "playlists"]
    selected_table = st.selectbox("Select table to export", export_options)
    
    if st.button("📤 Export to CSV"):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{selected_table}_export_{timestamp}.csv"
            
            success = st.session_state.database.export_to_csv(selected_table, output_path)
            
            if success:
                st.success(f"✅ Data exported to: {output_path}")
                
                # Provide download link
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download CSV",
                        data=f.read(),
                        file_name=output_path,
                        mime='text/csv'
                    )
            else:
                st.error("❌ Export failed!")
                
        except Exception as e:
            st.error(f"❌ Error exporting data: {e}")

def database_maintenance():
    """Database maintenance interface."""
    st.subheader("🧹 Database Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Backup Database"):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"youtube_insights_backup_{timestamp}.db"
                
                success = st.session_state.database.backup_database(backup_path)
                
                if success:
                    st.success(f"✅ Database backed up to: {backup_path}")
                else:
                    st.error("❌ Backup failed!")
                    
            except Exception as e:
                st.error(f"❌ Error creating backup: {e}")
    
    with col2:
        days_old = st.number_input("Clean data older than (days)", min_value=1, value=30)
        
        if st.button("🧹 Clean Old Data"):
            try:
                deleted_counts = st.session_state.database.cleanup_old_data(days_old)
                
                if deleted_counts:
                    st.success("✅ Cleanup completed:")
                    for table, count in deleted_counts.items():
                        st.write(f"  {table}: {count} records deleted")
                else:
                    st.warning("No old data found to clean")
                    
            except Exception as e:
                st.error(f"❌ Error during cleanup: {e}")

def scraper_settings():
    """Scraper configuration settings."""
    st.subheader("🔧 Scraper Configuration")
    
    if st.session_state.scraper:
        st.success("✅ Scraper is configured and ready")
        
        if st.button("🔄 Reconfigure Scraper"):
            st.session_state.scraper = None
            st.rerun()
    else:
        st.warning("⚠️ Scraper not configured")

def interface_settings():
    """Interface settings."""
    st.subheader("🎨 Interface Settings")
    
    # Theme selection
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh data", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)

def data_settings():
    """Data management settings."""
    st.subheader("📊 Data Settings")
    
    # Clear session data
    if st.button("🗑️ Clear Session Data", type="secondary"):
        st.session_state.scraped_data = {
            'videos': [],
            'channels': [],
            'comments': [],
            'trending': []
        }
        st.session_state.analysis_results = {}
        st.success("✅ Session data cleared!")

# ========== PAGE FUNCTIONS ==========

def sidebar_navigation():
    """Sidebar navigation."""
    with st.sidebar:
        st.title("🎥 Navigation")
        
        pages = {
            "🏠 Dashboard": "home",
            "🔍 Data Scraping": "scraping", 
            "📊 Channel Analysis": "channel_analysis",
            "😊 Sentiment Analysis": "sentiment_analysis",
            "🔥 Trending Analysis": "trending_analysis",
            "🎵 Playlist Generator": "playlist_generator",
            "🖼️ Thumbnail Analysis": "thumbnail_analysis",
            "📈 Visualizations": "visualizations",
            "💾 Database": "database",
            "⚙️ Settings": "settings"
        }
        
        selected_page = st.selectbox("Select Page", list(pages.keys()))
        return pages[selected_page]

def home_page():
    """Home/Dashboard page."""
    st.title("🎥 YouTube Content Insights")
    st.markdown("### Comprehensive YouTube content analysis and optimization platform")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        stats = st.session_state.database.get_database_stats()
        
        with col1:
            st.metric("📹 Videos", stats.get('videos_count', 0))
        with col2:
            st.metric("📺 Channels", stats.get('channels_count', 0))
        with col3:
            st.metric("💬 Comments", stats.get('comments_count', 0))
        with col4:
            st.metric("📊 Analyses", stats.get('analysis_results_count', 0))
    except Exception as e:
        st.error(f"Could not load database stats: {e}")
    
    # Quick actions
    st.markdown("### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Videos", use_container_width=True):
            st.session_state.current_page = "scraping"
            st.rerun()
    
    with col2:
        if st.button("📊 Analyze Channel", use_container_width=True):
            st.session_state.current_page = "channel_analysis"
            st.rerun()
    
    with col3:
        if st.button("📈 View Trends", use_container_width=True):
            st.session_state.current_page = "trending_analysis"
            st.rerun()
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    
    try:
        recent_videos = st.session_state.database.get_videos(limit=5)
        if recent_videos:
            df = pd.DataFrame(recent_videos)
            st.dataframe(df[['title', 'channel_name', 'views', 'scraped_at']], use_container_width=True)
        else:
            st.info("No recent data. Start by scraping some videos!")
    except Exception as e:
        st.error(f"Could not load recent activity: {e}")

def channel_analysis_page():
    """Channel analysis page."""
    st.title("📊 Channel Analysis")
    
    if not st.session_state.scraper:
        st.warning("⚠️ Please configure the scraper first in Data Scraping section")
        return
    
    # Channel input
    channel_url = st.text_input("Channel URL", placeholder="https://www.youtube.com/@channelname")
    
    if st.button("🔍 Analyze Channel"):
        if channel_url:
            with st.spinner("Analyzing channel..."):
                try:
                    # Scrape channel data
                    channel_data = st.session_state.scraper.get_channel_info(channel_url)
                    
                    if channel_data:
                        # Store in database
                        st.session_state.database.store_channels([channel_data])
                        
                        # Analyze with ChannelAnalyzer
                        analyzer = ChannelAnalyzer()
                        analysis = analyzer.analyze_channel_performance([channel_data])
                        
                        # Store analysis results
                        st.session_state.database.store_analysis_result(
                            "channel_analysis", channel_url, analysis
                        )
                        
                        # Display results
                        st.success("✅ Channel analysis completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Channel Info")
                            st.write(f"**Name:** {channel_data.get('name', 'N/A')}")
                            st.write(f"**Subscribers:** {channel_data.get('subscribers', 0):,}")
                            st.write(f"**Total Videos:** {channel_data.get('total_videos', 0):,}")
                        
                        with col2:
                            st.subheader("📈 Performance Metrics")
                            if analysis:
                                st.json(analysis)
                    
                    else:
                        st.error("❌ Could not retrieve channel data")
                        
                except Exception as e:
                    st.error(f"❌ Error analyzing channel: {e}")
        else:
            st.error("Please enter a channel URL")

def sentiment_analysis_page():
    """Sentiment analysis page."""
    st.title("😊 Sentiment Analysis")
    
    if not st.session_state.scraper:
        st.warning("⚠️ Please configure the scraper first in Data Scraping section")
        return
    
    video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("🔍 Analyze Sentiment"):
        if video_url:
            with st.spinner("Analyzing comments sentiment..."):
                try:
                    # Scrape comments
                    comments = st.session_state.scraper.get_video_comments(video_url, max_comments=100)
                    
                    if comments:
                        # Store comments
                        st.session_state.database.store_comments(comments)
                        
                        # Analyze sentiment
                        analyzer = SentimentAnalyzer()
                        sentiment_results = analyzer.analyze_video_sentiment(comments)
                        
                        # Store results
                        st.session_state.database.store_analysis_result(
                            "sentiment_analysis", video_url, sentiment_results
                        )
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(comments)} comments")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Sentiment Distribution")
                            if sentiment_results:
                                sentiment_data = sentiment_results.get('sentiment_distribution', {})
                                if sentiment_data:
                                    fig = px.pie(
                                        values=list(sentiment_data.values()),
                                        names=list(sentiment_data.keys()),
                                        title="Comment Sentiment"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("📈 Metrics")
                            if sentiment_results:
                                st.metric("Overall Score", f"{sentiment_results.get('overall_sentiment_score', 0):.2f}")
                                st.metric("Positive %", f"{sentiment_results.get('positive_percentage', 0):.1f}%")
                                st.metric("Negative %", f"{sentiment_results.get('negative_percentage', 0):.1f}%")
                    
                    else:
                        st.error("❌ Could not retrieve comments")
                        
                except Exception as e:
                    st.error(f"❌ Error analyzing sentiment: {e}")
        else:
            st.error("Please enter a video URL")

def trending_analysis_page():
    """Trending analysis page."""
    st.title("🔥 Trending Analysis")
    
    if not st.session_state.scraper:
        st.warning("⚠️ Please configure the scraper first in Data Scraping section")
        return
    
    if st.button("🔍 Analyze Trending Videos"):
        with st.spinner("Analyzing trending content..."):
            try:
                # Scrape trending videos
                trending_videos = st.session_state.scraper.get_trending_videos(max_results=50)
                
                if trending_videos:
                    # Store videos
                    st.session_state.database.store_videos(trending_videos)
                    
                    # Analyze trends
                    analyzer = TrendsAnalyzer()
                    trend_analysis = analyzer.analyze_trending_patterns(trending_videos)
                    
                    # Store results
                    st.session_state.database.store_analysis_result(
                        "trending_analysis", "trending", trend_analysis
                    )
                    
                    # Display results
                    st.success(f"✅ Analyzed {len(trending_videos)} trending videos")
                    
                    # Show trending videos
                    st.subheader("📊 Trending Videos")
                    df = pd.DataFrame(trending_videos)
                    if not df.empty:
                        st.dataframe(df[['title', 'channel_name', 'views', 'duration']], use_container_width=True)
                    
                    # Show analysis
                    if trend_analysis:
                        st.subheader("📈 Trend Analysis")
                        st.json(trend_analysis)
                
                else:
                    st.error("❌ Could not retrieve trending videos")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing trends: {e}")

def playlist_generator_page():
    """Playlist generator page."""
    st.title("🎵 Playlist Generator")
    
    st.markdown("### Generate Smart Playlists")
    
    # Playlist options
    col1, col2 = st.columns(2)
    
    with col1:
        playlist_type = st.selectbox("Playlist Type", [
            "Top Performing Videos",
            "Recent Uploads",
            "Most Engaging",
            "Trending Topic"
        ])
    
    with col2:
        max_videos = st.number_input("Max Videos", min_value=5, max_value=50, value=10)
    
    if playlist_type == "Trending Topic":
        topic = st.text_input("Topic/Keyword", placeholder="e.g., AI, music, gaming")
    
    if st.button("🎵 Generate Playlist"):
        try:
            generator = PlaylistGenerator()
            
            # Get videos from database
            videos = st.session_state.database.get_videos(limit=100)
            
            if videos:
                if playlist_type == "Top Performing Videos":
                    playlist = generator.create_top_performing_playlist(videos, max_videos)
                elif playlist_type == "Recent Uploads":
                    playlist = generator.create_recent_playlist(videos, max_videos)
                elif playlist_type == "Most Engaging":
                    playlist = generator.create_engagement_playlist(videos, max_videos)
                elif playlist_type == "Trending Topic":
                    if 'topic' in locals():
                        playlist = generator.create_topic_playlist(videos, topic, max_videos)
                    else:
                        st.error("Please enter a topic")
                        return
                
                if playlist:
                    st.success(f"✅ Generated playlist with {len(playlist)} videos")
                    
                    # Display playlist
                    st.subheader("🎵 Generated Playlist")
                    for i, video in enumerate(playlist, 1):
                        with st.expander(f"{i}. {video.get('title', 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Channel:** {video.get('channel_name', 'N/A')}")
                                st.write(f"**Views:** {video.get('views', 0):,}")
                            with col2:
                                st.write(f"**Duration:** {video.get('duration', 'N/A')}")
                                if video.get('url'):
                                    st.link_button("🔗 Watch", video['url'])
                else:
                    st.error("❌ Could not generate playlist")
            else:
                st.warning("No videos available. Please scrape some videos first.")
                
        except Exception as e:
            st.error(f"❌ Error generating playlist: {e}")

def thumbnail_analysis_page():
    """Thumbnail analysis page."""
    st.title("🖼️ Thumbnail Analysis")
    
    if not st.session_state.scraper:
        st.warning("⚠️ Please configure the scraper first in Data Scraping section")
        return
    
    # Analysis options
    analysis_type = st.selectbox("Analysis Type", [
        "Single Video Thumbnail",
        "Channel Thumbnail Analysis",
        "Topic Thumbnail Trends"
    ])
    
    if analysis_type == "Single Video Thumbnail":
        video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
        
        if st.button("🔍 Analyze Thumbnail"):
            if video_url:
                with st.spinner("Analyzing thumbnail..."):
                    try:
                        # Get video data
                        video_data = st.session_state.scraper.search_videos("", max_results=1)
                        
                        if video_data:
                            analyzer = ThumbnailAnalyzer()
                            thumbnail_analysis = analyzer.analyze_thumbnail_effectiveness([video_data[0]])
                            
                            # Store results
                            st.session_state.database.store_analysis_result(
                                "thumbnail_analysis", video_url, thumbnail_analysis
                            )
                            
                            st.success("✅ Thumbnail analysis completed!")
                            st.json(thumbnail_analysis)
                        else:
                            st.error("❌ Could not retrieve video data")
                            
                    except Exception as e:
                        st.error(f"❌ Error analyzing thumbnail: {e}")
            else:
                st.error("Please enter a video URL")
    
    elif analysis_type == "Topic Thumbnail Trends":
        topic = st.text_input("Topic/Keyword", placeholder="e.g., AI, music, gaming")
        
        if st.button("🔍 Analyze Topic Thumbnails"):
            if topic:
                with st.spinner("Analyzing topic thumbnails..."):
                    try:
                        # Search videos by topic
                        videos = st.session_state.scraper.search_videos(topic, max_results=20)
                        
                        if videos:
                            analyzer = ThumbnailAnalyzer()
                            thumbnail_trends = analyzer.analyze_thumbnail_trends(videos, topic)
                            
                            # Store results
                            st.session_state.database.store_analysis_result(
                                "thumbnail_trends", topic, thumbnail_trends
                            )
                            
                            st.success(f"✅ Analyzed thumbnails for '{topic}'")
                            st.json(thumbnail_trends)
                        else:
                            st.error("❌ Could not find videos for this topic")
                            
                    except Exception as e:
                        st.error(f"❌ Error analyzing thumbnails: {e}")
            else:
                st.error("Please enter a topic")

def visualizations_page():
    """Data visualizations page."""
    st.title("📈 Visualizations")
    
    # Get data from database
    try:
        videos = st.session_state.database.get_videos(limit=100)
        channels = st.session_state.database.get_channels(limit=50)
        
        if not videos and not channels:
            st.warning("No data available for visualization. Please scrape some data first.")
            return
        
        # Visualization options
        viz_type = st.selectbox("Visualization Type", [
            "Video Performance",
            "Channel Comparison", 
            "View Distribution",
            "Upload Trends",
            "Engagement Analysis"
        ])
        
        if viz_type == "Video Performance" and videos:
            st.subheader("📊 Video Performance")
            
            df = pd.DataFrame(videos)
            if 'views' in df.columns and 'title' in df.columns:
                # Top videos by views
                top_videos = df.nlargest(10, 'views')
                
                fig = px.bar(
                    top_videos,
                    x='views',
                    y='title',
                    orientation='h',
                    title="Top 10 Videos by Views"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Channel Comparison" and channels:
            st.subheader("📺 Channel Comparison")
            
            df = pd.DataFrame(channels)
            if 'subscribers' in df.columns and 'name' in df.columns:
                fig = px.bar(
                    df,
                    x='name',
                    y='subscribers',
                    title="Channel Subscribers Comparison"
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "View Distribution" and videos:
            st.subheader("👁️ View Distribution")
            
            df = pd.DataFrame(videos)
            if 'views' in df.columns:
                fig = px.histogram(
                    df,
                    x='views',
                    title="Video Views Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Custom visualization
        st.subheader("🎨 Custom Visualization")
        
        visualizer = DataVisualizer()
        
        if st.button("📊 Generate Performance Report"):
            try:
                report_data = visualizer.create_performance_dashboard(videos, channels)
                st.success("✅ Performance report generated!")
                
                if report_data:
                    st.json(report_data)
                    
            except Exception as e:
                st.error(f"❌ Error generating report: {e}")
                
    except Exception as e:
        st.error(f"❌ Error loading data for visualization: {e}")

def database_page():
    """Database management page."""
    st.title("💾 Database Management")
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📤 Export", "🧹 Maintenance"])
    
    with tab1:
        database_statistics()
    
    with tab2:
        database_export()
    
    with tab3:
        database_maintenance()

def settings_page():
    """Settings page."""
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Scraper", "🎨 Interface", "📊 Data"])
    
    with tab1:
        scraper_settings()
    
    with tab2:
        interface_settings()
    
    with tab3:
        data_settings()

def main():
    """Main application function."""
    initialize_session_state()
    main_header()
    
    # Navigation
    current_page = sidebar_navigation()
    
    # Page routing
    if current_page == "home":
        home_page()
    elif current_page == "scraping":
        scraping_page()
    elif current_page == "channel_analysis":
        channel_analysis_page()
    elif current_page == "sentiment_analysis":
        sentiment_analysis_page()
    elif current_page == "trending_analysis":
        trending_analysis_page()
    elif current_page == "playlist_generator":
        playlist_generator_page()
    elif current_page == "thumbnail_analysis":
        thumbnail_analysis_page()
    elif current_page == "visualizations":
        visualizations_page()
    elif current_page == "database":
        database_page()
    elif current_page == "settings":
        settings_page()

if __name__ == "__main__":
    main()
