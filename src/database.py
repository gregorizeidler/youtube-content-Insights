"""
Database module for storing and managing YouTube analysis data.
"""

import sqlite3
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os
import threading

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager for YouTube analysis data."""
    
    def __init__(self, db_path: str = "youtube_insights.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize_database()
    
    def _get_connection(self):
        """Get a thread-safe database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _initialize_database(self):
        """Initialize database with required tables."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    self._create_tables(conn)
            
            logger.info(f"Database initialized at: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_tables(self, conn):
        """Create all required tables."""
        cursor = conn.cursor()
        
        # Videos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT UNIQUE,
                channel_name TEXT,
                channel_url TEXT,
                views INTEGER DEFAULT 0,
                duration TEXT,
                upload_date TEXT,
                description TEXT,
                tags TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Channels table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT UNIQUE,
                subscribers INTEGER DEFAULT 0,
                total_videos INTEGER DEFAULT 0,
                description TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Comments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url TEXT,
                author TEXT,
                text TEXT,
                likes INTEGER DEFAULT 0,
                timestamp TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_url) REFERENCES videos (url)
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT NOT NULL,
                target_url TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trending snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trending_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url TEXT,
                position INTEGER,
                snapshot_date DATE,
                views INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_videos_views ON videos(views)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(analysis_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trending_date ON trending_snapshots(snapshot_date)')
        
        conn.commit()
        logger.info("Database tables created successfully")
    
    def store_videos(self, videos_data: List[Dict]) -> int:
        """Store video data in the database."""
        if not videos_data:
            return 0
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                stored_count = 0
                
                for video in videos_data:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO videos 
                            (title, url, channel_name, channel_url, views, duration, upload_date, description, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            video.get('title', ''),
                            video.get('url', ''),
                            video.get('channel_name', ''),
                            video.get('channel_url', ''),
                            video.get('views', 0),
                            video.get('duration', ''),
                            video.get('upload_date', ''),
                            video.get('description', ''),
                            json.dumps(video.get('tags', []))
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Error storing video {video.get('title', 'Unknown')}: {e}")
                
                conn.commit()
                logger.info(f"Stored {stored_count} videos in database")
                return stored_count
    
    def store_channels(self, channels_data: List[Dict]) -> int:
        """Store channel data in the database."""
        if not channels_data:
            return 0
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                stored_count = 0
                
                for channel in channels_data:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO channels 
                            (name, url, subscribers, total_videos, description)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            channel.get('name', ''),
                            channel.get('url', ''),
                            channel.get('subscribers', 0),
                            channel.get('total_videos', 0),
                            channel.get('description', '')
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Error storing channel {channel.get('name', 'Unknown')}: {e}")
                
                conn.commit()
                logger.info(f"Stored {stored_count} channels in database")
                return stored_count
    
    def store_comments(self, comments_data: List[Dict]) -> int:
        """Store comment data in the database."""
        if not comments_data:
            return 0
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                stored_count = 0
                
                for comment in comments_data:
                    try:
                        cursor.execute('''
                            INSERT INTO comments 
                            (video_url, author, text, likes, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            comment.get('video_url', ''),
                            comment.get('author', ''),
                            comment.get('text', ''),
                            comment.get('likes', 0),
                            comment.get('timestamp', '')
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Error storing comment: {e}")
                
                conn.commit()
                logger.info(f"Stored {stored_count} comments in database")
                return stored_count
    
    def store_analysis_result(self, analysis_type: str, target_url: str, results: Dict) -> int:
        """Store analysis results in the database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute('''
                        INSERT INTO analysis_results 
                        (analysis_type, target_url, results)
                        VALUES (?, ?, ?)
                    ''', (analysis_type, target_url, json.dumps(results)))
                    
                    conn.commit()
                    logger.info(f"Stored {analysis_type} analysis result for {target_url}")
                    return cursor.lastrowid
                    
                except Exception as e:
                    logger.error(f"Error storing analysis result: {e}")
                    return 0
    
    def get_videos(self, limit: int = 100) -> List[Dict]:
        """Retrieve videos from database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM videos ORDER BY scraped_at DESC LIMIT ?', (limit,))
                
                videos = []
                for row in cursor.fetchall():
                    video = dict(row)
                    if video['tags']:
                        try:
                            video['tags'] = json.loads(video['tags'])
                        except:
                            video['tags'] = []
                    videos.append(video)
                
                return videos
    
    def get_channels(self, limit: int = 100) -> List[Dict]:
        """Retrieve channels from database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM channels ORDER BY scraped_at DESC LIMIT ?', (limit,))
                return [dict(row) for row in cursor.fetchall()]
    
    def get_comments(self, video_url: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve comments from database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if video_url:
                    cursor.execute('SELECT * FROM comments WHERE video_url = ? ORDER BY scraped_at DESC LIMIT ?', 
                                 (video_url, limit))
                else:
                    cursor.execute('SELECT * FROM comments ORDER BY scraped_at DESC LIMIT ?', (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
    
    def get_analysis_results(self, analysis_type: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve analysis results from database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if analysis_type:
                    cursor.execute('SELECT * FROM analysis_results WHERE analysis_type = ? ORDER BY created_at DESC LIMIT ?', 
                                 (analysis_type, limit))
                else:
                    cursor.execute('SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT ?', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['results']:
                        try:
                            result['results'] = json.loads(result['results'])
                        except:
                            result['results'] = {}
                    results.append(result)
                
                return results
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    stats = {}
                    
                    # Count records in each table
                    tables = ['videos', 'channels', 'comments', 'analysis_results', 'trending_snapshots']
                    for table in tables:
                        cursor.execute(f'SELECT COUNT(*) FROM {table}')
                        stats[f'{table}_count'] = cursor.fetchone()[0]
                    
                    return stats
                    
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def export_to_csv(self, table_name: str, output_path: str) -> bool:
        """Export table data to CSV."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    df.to_csv(output_path, index=False)
                    logger.info(f"Exported {table_name} to {output_path}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error exporting {table_name}: {e}")
            return False
    
    def clear_table(self, table_name: str) -> bool:
        """Clear all data from a table."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'DELETE FROM {table_name}')
                    conn.commit()
                    logger.info(f"Cleared table {table_name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error clearing table {table_name}: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        # With our thread-safe approach, connections are closed automatically
        logger.info("Database manager closed")