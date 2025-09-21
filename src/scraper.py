"""
Core YouTube scraper module with enhanced functionality.
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeScraper:
    """Enhanced YouTube scraper with multiple data extraction capabilities."""
    
    def __init__(self, browser_path: str = None, driver_path: str = None, headless: bool = True):
        """
        Initialize the YouTube scraper.
        
        Args:
            browser_path: Path to browser executable (optional)
            driver_path: Path to ChromeDriver executable (optional, will auto-download)
            headless: Run browser in headless mode
        """
        self.browser_path = browser_path
        self.driver_path = driver_path
        self.headless = headless
        self.driver = self._setup_driver()
        self.scraped_data = {
            'videos': [],
            'channels': [],
            'comments': [],
            'trending': []
        }

    def _setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome driver with options."""
        options = Options()
        
        # Set browser path if provided
        if self.browser_path:
            options.binary_location = self.browser_path
        
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Use webdriver-manager to automatically download and manage ChromeDriver
        if self.driver_path:
            service = Service(executable_path=self.driver_path)
        else:
            service = Service(ChromeDriverManager().install())
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver

    def search_videos(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search for videos and extract detailed information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to scrape
            
        Returns:
            List of video dictionaries with metadata
        """
        logger.info(f"Searching for videos: {query}")
        
        try:
            self.driver.get('https://www.youtube.com/')
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, 'search_query'))
            )
            
            search_box = self.driver.find_element(By.NAME, 'search_query')
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, 'video-title'))
            )
            
            videos = []
            scroll_count = 0
            max_scrolls = max_results // 20 + 1
            
            while len(videos) < max_results and scroll_count < max_scrolls:
                video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ytd-video-renderer')
                
                for element in video_elements:
                    if len(videos) >= max_results:
                        break
                        
                    try:
                        video_data = self._extract_video_data(element)
                        if video_data and video_data not in videos:
                            videos.append(video_data)
                    except Exception as e:
                        logger.warning(f"Error extracting video data: {e}")
                        continue
                
                # Scroll down to load more videos
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
                scroll_count += 1
            
            self.scraped_data['videos'].extend(videos)
            logger.info(f"Successfully scraped {len(videos)} videos")
            return videos
            
        except TimeoutException:
            logger.error("Timeout waiting for page elements")
            return []
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []

    def _extract_video_data(self, element) -> Optional[Dict]:
        """Extract video metadata from a video element."""
        try:
            # Extract title and URL
            title_element = element.find_element(By.ID, 'video-title')
            title = title_element.get_attribute('title') or title_element.text
            url = title_element.get_attribute('href')
            
            # Extract channel name with multiple fallback selectors
            channel_name = "Unknown Channel"
            channel_url = ""
            
            channel_selectors = [
                'a.yt-simple-endpoint.style-scope.yt-formatted-string',
                'a[href*="/channel/"]',
                'a[href*="/@"]',
                '.ytd-channel-name a',
                '#channel-name a',
                '#text a'
            ]
            
            for selector in channel_selectors:
                try:
                    channel_element = element.find_element(By.CSS_SELECTOR, selector)
                    if channel_element.text.strip():
                        channel_name = channel_element.text.strip()
                        channel_url = channel_element.get_attribute('href') or ""
                        break
                except NoSuchElementException:
                    continue
            
            # Extract view count and upload time with fallbacks
            views = "0"
            upload_time = "Unknown"
            
            try:
                metadata_elements = element.find_elements(By.CSS_SELECTOR, 'span.style-scope.ytd-video-meta-block')
                if len(metadata_elements) > 0:
                    views = metadata_elements[0].text
                if len(metadata_elements) > 1:
                    upload_time = metadata_elements[1].text
            except:
                # Fallback for views
                try:
                    views_element = element.find_element(By.CSS_SELECTOR, '#metadata-line span')
                    views = views_element.text
                except:
                    pass
            
            # Extract duration with fallbacks
            duration = "N/A"
            duration_selectors = [
                'span.style-scope.ytd-thumbnail-overlay-time-status-renderer',
                '.ytd-thumbnail-overlay-time-status-renderer span',
                '#overlays .ytd-thumbnail-overlay-time-status-renderer'
            ]
            
            for selector in duration_selectors:
                try:
                    duration_element = element.find_element(By.CSS_SELECTOR, selector)
                    if duration_element.text:
                        duration = duration_element.text
                        break
                except NoSuchElementException:
                    continue
            
            # Clean up views (remove "views" text and convert to number)
            if views and views != "0":
                views_clean = views.replace(" views", "").replace(" visualizações", "").replace(",", "")
                try:
                    # Convert abbreviated numbers (1.2M, 500K, etc.)
                    if 'M' in views_clean:
                        views_num = float(views_clean.replace('M', '')) * 1000000
                    elif 'K' in views_clean:
                        views_num = float(views_clean.replace('K', '')) * 1000
                    else:
                        views_num = int(views_clean)
                    views = int(views_num)
                except:
                    views = 0
            else:
                views = 0
            
            return {
                'title': title,
                'url': url,
                'channel_name': channel_name,
                'channel_url': channel_url,
                'views': views,
                'upload_date': upload_time,
                'duration': duration,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.warning(f"Error extracting video data: {e}")
            return None

    def scrape_channel_info(self, channel_url: str) -> Dict:
        """
        Scrape detailed information about a YouTube channel.
        
        Args:
            channel_url: URL of the YouTube channel
            
        Returns:
            Dictionary with channel information
        """
        logger.info(f"Scraping channel info: {channel_url}")
        
        try:
            self.driver.get(channel_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'yt-formatted-string#text'))
            )
            
            # Extract channel name
            channel_name = self.driver.find_element(By.CSS_SELECTOR, 'yt-formatted-string#text.style-scope.ytd-channel-name').text
            
            # Extract subscriber count
            try:
                subscriber_element = self.driver.find_element(By.CSS_SELECTOR, 'yt-formatted-string#subscriber-count')
                subscribers = subscriber_element.text
            except NoSuchElementException:
                subscribers = "N/A"
            
            # Navigate to videos tab
            videos_tab = self.driver.find_element(By.CSS_SELECTOR, 'tp-yt-paper-tab[tab-title="Videos"]')
            videos_tab.click()
            time.sleep(3)
            
            # Get recent videos
            recent_videos = self.search_videos("", max_results=20)
            
            channel_data = {
                'name': channel_name,
                'url': channel_url,
                'subscribers': subscribers,
                'recent_videos': recent_videos,
                'total_videos_scraped': len(recent_videos),
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.scraped_data['channels'].append(channel_data)
            return channel_data
            
        except Exception as e:
            logger.error(f"Error scraping channel info: {e}")
            return {}

    def scrape_trending_videos(self, category: str = "all") -> List[Dict]:
        """
        Scrape trending videos from YouTube.
        
        Args:
            category: Trending category (all, music, gaming, movies, news)
            
        Returns:
            List of trending video dictionaries
        """
        logger.info(f"Scraping trending videos: {category}")
        
        try:
            trending_url = "https://www.youtube.com/feed/trending"
            if category != "all":
                trending_url += f"?bp=6gQJRkVleHBsb3Jl"  # Base parameter for categories
            
            self.driver.get(trending_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'ytd-video-renderer'))
            )
            
            trending_videos = []
            video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ytd-video-renderer')
            
            for element in video_elements[:50]:  # Limit to top 50 trending
                try:
                    video_data = self._extract_video_data(element)
                    if video_data:
                        video_data['trending_category'] = category
                        trending_videos.append(video_data)
                except Exception as e:
                    logger.warning(f"Error extracting trending video: {e}")
                    continue
            
            self.scraped_data['trending'].extend(trending_videos)
            logger.info(f"Successfully scraped {len(trending_videos)} trending videos")
            return trending_videos
            
        except Exception as e:
            logger.error(f"Error scraping trending videos: {e}")
            return []

    def scrape_video_comments(self, video_url: str, max_comments: int = 100) -> List[Dict]:
        """
        Scrape comments from a specific video.
        
        Args:
            video_url: URL of the YouTube video
            max_comments: Maximum number of comments to scrape
            
        Returns:
            List of comment dictionaries
        """
        logger.info(f"Scraping comments from: {video_url}")
        
        try:
            self.driver.get(video_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'ytd-comments'))
            )
            
            # Scroll to comments section
            self.driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(3)
            
            comments = []
            scroll_count = 0
            max_scrolls = max_comments // 20 + 1
            
            while len(comments) < max_comments and scroll_count < max_scrolls:
                comment_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ytd-comment-thread-renderer')
                
                for element in comment_elements:
                    if len(comments) >= max_comments:
                        break
                        
                    try:
                        comment_data = self._extract_comment_data(element)
                        if comment_data and comment_data not in comments:
                            comments.append(comment_data)
                    except Exception as e:
                        logger.warning(f"Error extracting comment: {e}")
                        continue
                
                # Scroll down to load more comments
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
                scroll_count += 1
            
            self.scraped_data['comments'].extend(comments)
            logger.info(f"Successfully scraped {len(comments)} comments")
            return comments
            
        except Exception as e:
            logger.error(f"Error scraping comments: {e}")
            return []

    def _extract_comment_data(self, element) -> Optional[Dict]:
        """Extract comment data from a comment element."""
        try:
            author_element = element.find_element(By.CSS_SELECTOR, 'a#author-text')
            author = author_element.text.strip()
            
            comment_element = element.find_element(By.CSS_SELECTOR, 'yt-formatted-string#content-text')
            comment_text = comment_element.text
            
            # Extract likes (if available)
            try:
                likes_element = element.find_element(By.CSS_SELECTOR, 'span#vote-count-middle')
                likes = likes_element.text
            except NoSuchElementException:
                likes = "0"
            
            # Extract time
            try:
                time_element = element.find_element(By.CSS_SELECTOR, 'yt-formatted-string.published-time-text')
                comment_time = time_element.text
            except NoSuchElementException:
                comment_time = "N/A"
            
            return {
                'author': author,
                'text': comment_text,
                'likes': likes,
                'time': comment_time,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except NoSuchElementException:
            return None

    def get_scraped_data(self) -> Dict:
        """Return all scraped data."""
        return self.scraped_data

    def clear_data(self):
        """Clear all scraped data."""
        self.scraped_data = {
            'videos': [],
            'channels': [],
            'comments': [],
            'trending': []
        }

    def close(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed successfully")
    
    # Alias methods for Streamlit compatibility
    def get_channel_info(self, channel_url: str) -> Dict:
        """Alias for scrape_channel_info."""
        return self.scrape_channel_info(channel_url)
    
    def get_trending_videos(self, max_results: int = 50) -> List[Dict]:
        """Alias for scrape_trending_videos."""
        return self.scrape_trending_videos("all")[:max_results]
    
    def get_video_comments(self, video_url: str, max_comments: int = 100) -> List[Dict]:
        """Alias for scrape_video_comments."""
        return self.scrape_video_comments(video_url, max_comments)
