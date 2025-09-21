"""
Thumbnail analyzer module for analyzing YouTube video thumbnails and their effectiveness.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import requests
from PIL import Image, ImageStat
import io
import os
from collections import Counter, defaultdict
import colorsys
import cv2

# OCR and Face Detection
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  pytesseract not installed. OCR features will be limited.")
    print("Install with: pip install pytesseract")

try:
    import face_recognition
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("⚠️  face_recognition not installed. Face detection features will be limited.")
    print("Install with: pip install face_recognition")

logger = logging.getLogger(__name__)


class ThumbnailAnalyzer:
    """Analyze YouTube video thumbnails for patterns and effectiveness."""
    
    def __init__(self, cache_dir: str = "thumbnail_cache"):
        self.cache_dir = cache_dir
        self.analysis_results = {}
        self.thumbnail_cache = {}
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def analyze_thumbnails(self, videos_data: List[Dict], 
                          download_thumbnails: bool = True,
                          max_thumbnails: int = 50) -> Dict[str, Any]:
        """
        Analyze thumbnails from a collection of videos.
        
        Args:
            videos_data: List of video dictionaries with thumbnail URLs
            download_thumbnails: Whether to download and analyze actual thumbnails
            max_thumbnails: Maximum number of thumbnails to analyze
            
        Returns:
            Dictionary with thumbnail analysis results
        """
        if not videos_data:
            return {"error": "No video data provided"}
        
        logger.info(f"Analyzing thumbnails for {min(len(videos_data), max_thumbnails)} videos")
        
        # Limit the number of thumbnails to analyze
        videos_to_analyze = videos_data[:max_thumbnails]
        
        analysis = {
            'overview': {
                'total_videos': len(videos_to_analyze),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'thumbnails_analyzed': 0
            },
            'color_analysis': {},
            'composition_analysis': {},
            'text_analysis': {},
            'face_analysis': {},
            'performance_correlation': {},
            'recommendations': []
        }
        
        if download_thumbnails:
            # Download and analyze thumbnails
            thumbnail_data = self._download_and_analyze_thumbnails(videos_to_analyze)
            analysis['overview']['thumbnails_analyzed'] = len(thumbnail_data)
            
            if thumbnail_data:
                analysis['color_analysis'] = self._analyze_colors(thumbnail_data)
                analysis['composition_analysis'] = self._analyze_composition(thumbnail_data)
                analysis['text_analysis'] = self._analyze_text_elements(thumbnail_data)
                analysis['face_analysis'] = self._analyze_faces(thumbnail_data)
                analysis['performance_correlation'] = self._correlate_with_performance(thumbnail_data, videos_to_analyze)
        else:
            # Basic analysis without downloading
            analysis = self._analyze_without_download(videos_to_analyze)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_thumbnail_recommendations(analysis)
        
        # Store results
        analysis_id = f"thumbnail_analysis_{len(self.analysis_results) + 1}"
        self.analysis_results[analysis_id] = analysis
        
        return analysis
    
    def compare_thumbnail_strategies(self, channels_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Compare thumbnail strategies across different channels.
        
        Args:
            channels_data: Dictionary with channel names as keys and video lists as values
            
        Returns:
            Dictionary with comparison results
        """
        if len(channels_data) < 2:
            return {"error": "At least 2 channels required for comparison"}
        
        logger.info(f"Comparing thumbnail strategies across {len(channels_data)} channels")
        
        # Analyze each channel's thumbnails
        channel_analyses = {}
        for channel_name, videos in channels_data.items():
            channel_analyses[channel_name] = self.analyze_thumbnails(videos, max_thumbnails=20)
        
        # Perform comparison
        comparison = {
            'comparison_overview': {
                'channels_compared': list(channels_data.keys()),
                'comparison_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'color_strategy_comparison': self._compare_color_strategies(channel_analyses),
            'composition_comparison': self._compare_compositions(channel_analyses),
            'performance_comparison': self._compare_thumbnail_performance(channel_analyses),
            'best_practices': self._identify_best_practices(channel_analyses),
            'recommendations': self._generate_comparison_recommendations(channel_analyses)
        }
        
        return comparison
    
    def analyze_trending_thumbnails(self, trending_videos: List[Dict]) -> Dict[str, Any]:
        """
        Analyze thumbnails of trending videos to identify successful patterns.
        
        Args:
            trending_videos: List of trending video dictionaries
            
        Returns:
            Dictionary with trending thumbnail analysis
        """
        logger.info(f"Analyzing thumbnails of {len(trending_videos)} trending videos")
        
        # Analyze thumbnails
        base_analysis = self.analyze_thumbnails(trending_videos, max_thumbnails=30)
        
        # Add trending-specific analysis
        trending_analysis = {
            **base_analysis,
            'trending_patterns': self._identify_trending_patterns(base_analysis),
            'viral_indicators': self._identify_viral_indicators(base_analysis, trending_videos),
            'category_patterns': self._analyze_category_patterns(trending_videos),
            'trending_recommendations': self._generate_trending_recommendations(base_analysis)
        }
        
        return trending_analysis
    
    def _download_and_analyze_thumbnails(self, videos_data: List[Dict]) -> List[Dict]:
        """Download thumbnails and extract visual features."""
        thumbnail_data = []
        
        for i, video in enumerate(videos_data):
            try:
                # Extract thumbnail URL (this would need to be adapted based on your data structure)
                thumbnail_url = self._extract_thumbnail_url(video)
                
                if not thumbnail_url:
                    continue
                
                # Download thumbnail
                image = self._download_thumbnail(thumbnail_url, video.get('title', f'video_{i}'))
                
                if image is None:
                    continue
                
                # Analyze the image
                features = self._extract_image_features(image)
                
                thumbnail_info = {
                    'video_title': video.get('title', ''),
                    'video_views': video.get('views', 0),
                    'channel_name': video.get('channel_name', ''),
                    'thumbnail_url': thumbnail_url,
                    **features
                }
                
                thumbnail_data.append(thumbnail_info)
                
            except Exception as e:
                logger.warning(f"Error analyzing thumbnail for video {i}: {e}")
                continue
        
        return thumbnail_data
    
    def _extract_thumbnail_url(self, video: Dict) -> Optional[str]:
        """Extract thumbnail URL from video data."""
        # This is a placeholder - you'd need to adapt this based on your data structure
        # YouTube thumbnails typically follow patterns like:
        # https://img.youtube.com/vi/{video_id}/maxresdefault.jpg
        
        video_url = video.get('url', '')
        if 'youtube.com/watch?v=' in video_url:
            video_id = video_url.split('v=')[1].split('&')[0]
            return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]
            return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        return None
    
    def _download_thumbnail(self, url: str, filename: str) -> Optional[Image.Image]:
        """Download thumbnail image."""
        try:
            # Check cache first
            cache_path = os.path.join(self.cache_dir, f"{filename.replace('/', '_')[:50]}.jpg")
            
            if os.path.exists(cache_path):
                return Image.open(cache_path)
            
            # Download image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Open image
            image = Image.open(io.BytesIO(response.content))
            
            # Save to cache
            image.save(cache_path, 'JPEG')
            
            return image
            
        except Exception as e:
            logger.warning(f"Error downloading thumbnail from {url}: {e}")
            return None
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features from thumbnail image."""
        features = {}
        
        try:
            # Basic image properties
            features['width'] = image.width
            features['height'] = image.height
            features['aspect_ratio'] = round(image.width / image.height, 2)
            
            # Color analysis
            features.update(self._analyze_image_colors(image))
            
            # Brightness and contrast
            features.update(self._analyze_brightness_contrast(image))
            
            # Composition analysis
            features.update(self._analyze_image_composition(image))
            
            # Text detection (basic)
            features.update(self._detect_text_regions(image))
            
        except Exception as e:
            logger.warning(f"Error extracting image features: {e}")
        
        return features
    
    def _analyze_image_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color properties of the image."""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get dominant colors
        colors = image.getcolors(maxcolors=256*256*256)
        if colors:
            # Sort by frequency
            colors.sort(key=lambda x: x[0], reverse=True)
            
            # Get top 5 colors
            dominant_colors = []
            for count, color in colors[:5]:
                percentage = (count / (image.width * image.height)) * 100
                dominant_colors.append({
                    'color_rgb': color,
                    'color_hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    'percentage': round(percentage, 2)
                })
        else:
            dominant_colors = []
        
        # Color temperature (warm vs cool)
        avg_color = ImageStat.Stat(image).mean
        color_temp = self._calculate_color_temperature(avg_color)
        
        # Color saturation
        hsv_image = image.convert('HSV')
        saturation_values = list(hsv_image.getdata())
        avg_saturation = np.mean([s[1] for s in saturation_values]) / 255.0
        
        return {
            'dominant_colors': dominant_colors,
            'color_temperature': color_temp,
            'average_saturation': round(avg_saturation, 3),
            'color_diversity': len(colors) if colors else 0
        }
    
    def _analyze_brightness_contrast(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze brightness and contrast of the image."""
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        
        # Calculate brightness (average pixel value)
        brightness = ImageStat.Stat(gray_image).mean[0] / 255.0
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = ImageStat.Stat(gray_image).stddev[0] / 255.0
        
        # Histogram analysis
        histogram = gray_image.histogram()
        
        # Calculate dynamic range
        non_zero_bins = [i for i, count in enumerate(histogram) if count > 0]
        dynamic_range = (max(non_zero_bins) - min(non_zero_bins)) / 255.0 if non_zero_bins else 0
        
        return {
            'brightness': round(brightness, 3),
            'contrast': round(contrast, 3),
            'dynamic_range': round(dynamic_range, 3),
            'brightness_category': self._categorize_brightness(brightness),
            'contrast_category': self._categorize_contrast(contrast)
        }
    
    def _analyze_image_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze composition elements of the image."""
        # Convert to numpy array for OpenCV processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        composition = {}
        
        try:
            # Edge detection for complexity analysis
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            composition['edge_density'] = round(edge_density, 4)
            composition['complexity'] = 'High' if edge_density > 0.1 else 'Medium' if edge_density > 0.05 else 'Low'
            
            # Center focus analysis (check if center region is brighter/more saturated)
            h, w = gray.shape
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            border_region = np.concatenate([
                gray[:h//4, :].flatten(),
                gray[3*h//4:, :].flatten(),
                gray[:, :w//4].flatten(),
                gray[:, 3*w//4:].flatten()
            ])
            
            center_brightness = np.mean(center_region) / 255.0
            border_brightness = np.mean(border_region) / 255.0
            
            composition['center_focus'] = round(center_brightness - border_brightness, 3)
            composition['has_center_focus'] = composition['center_focus'] > 0.1
            
        except Exception as e:
            logger.warning(f"Error in composition analysis: {e}")
            composition = {
                'edge_density': 0,
                'complexity': 'Unknown',
                'center_focus': 0,
                'has_center_focus': False
            }
        
        return composition
    
    def _detect_text_regions(self, image: Image.Image) -> Dict[str, Any]:
        """Detect text regions in the image using OCR."""
        text_analysis = {
            'has_text_overlay': False,
            'text_coverage': 0.0,
            'text_position': 'unknown',
            'detected_text': '',
            'text_confidence': 0.0,
            'text_regions': [],
            'word_count': 0
        }
        
        if not OCR_AVAILABLE:
            # Fallback to basic analysis
            return self._basic_text_detection(image)
        
        try:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use pytesseract to detect text with bounding boxes
            data = pytesseract.image_to_data(cv_image, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate metrics
            detected_words = []
            text_regions = []
            confidences = []
            
            n_boxes = len(data['text'])
            image_area = image.width * image.height
            text_area = 0
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                # Filter out low confidence and empty text
                if confidence > 30 and text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    detected_words.append(text)
                    confidences.append(confidence)
                    text_area += w * h
                    
                    # Determine text position
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Normalize coordinates
                    norm_x = center_x / image.width
                    norm_y = center_y / image.height
                    
                    position = self._determine_text_position(norm_x, norm_y)
                    
                    text_regions.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'position': position,
                        'area': w * h
                    })
            
            # Compile results
            full_text = ' '.join(detected_words)
            
            text_analysis.update({
                'has_text_overlay': len(detected_words) > 0,
                'detected_text': full_text,
                'word_count': len(detected_words),
                'text_coverage': (text_area / image_area) * 100 if image_area > 0 else 0,
                'text_confidence': np.mean(confidences) if confidences else 0,
                'text_regions': text_regions,
                'dominant_text_position': self._get_dominant_text_position(text_regions)
            })
            
            # Overall text position
            if text_regions:
                positions = [region['position'] for region in text_regions]
                text_analysis['text_position'] = max(set(positions), key=positions.count)
            
        except Exception as e:
            logger.warning(f"OCR failed, using basic text detection: {e}")
            return self._basic_text_detection(image)
        
        return text_analysis
    
    def _basic_text_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Basic text detection fallback when OCR is not available."""
        # Convert to grayscale
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)
        
        # Simple edge detection to find potential text regions
        edges = cv2.Canny(gray_array, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be text (rectangular, appropriate size)
        text_like_contours = []
        image_area = image.width * image.height
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Heuristics for text-like regions
            if (0.1 < aspect_ratio < 10 and  # Not too wide or tall
                0.0001 < area / image_area < 0.3 and  # Reasonable size
                w > 10 and h > 5):  # Minimum dimensions
                text_like_contours.append({'x': x, 'y': y, 'width': w, 'height': h, 'area': area})
        
        total_text_area = sum(region['area'] for region in text_like_contours)
        
        return {
            'has_text_overlay': len(text_like_contours) > 0,
            'text_coverage': (total_text_area / image_area) * 100 if image_area > 0 else 0,
            'text_position': 'center' if text_like_contours else 'none',
            'detected_text': f"[{len(text_like_contours)} text regions detected]",
            'text_confidence': 50.0 if text_like_contours else 0.0,
            'text_regions': text_like_contours,
            'word_count': len(text_like_contours)  # Approximate
        }
    
    def _determine_text_position(self, norm_x: float, norm_y: float) -> str:
        """Determine text position based on normalized coordinates."""
        if norm_y < 0.33:
            if norm_x < 0.33:
                return 'top-left'
            elif norm_x > 0.67:
                return 'top-right'
            else:
                return 'top-center'
        elif norm_y > 0.67:
            if norm_x < 0.33:
                return 'bottom-left'
            elif norm_x > 0.67:
                return 'bottom-right'
            else:
                return 'bottom-center'
        else:
            if norm_x < 0.33:
                return 'middle-left'
            elif norm_x > 0.67:
                return 'middle-right'
            else:
                return 'center'
    
    def _get_dominant_text_position(self, text_regions: List[Dict]) -> str:
        """Get the dominant text position based on text area."""
        if not text_regions:
            return 'none'
        
        position_areas = {}
        for region in text_regions:
            position = region['position']
            area = region['area']
            position_areas[position] = position_areas.get(position, 0) + area
        
        return max(position_areas.items(), key=lambda x: x[1])[0]
    
    def _calculate_color_temperature(self, rgb_color: Tuple[float, float, float]) -> str:
        """Calculate color temperature (warm vs cool)."""
        r, g, b = rgb_color
        
        # Simple heuristic: more red/yellow = warm, more blue = cool
        warmth_score = (r + g/2) - b
        
        if warmth_score > 50:
            return "Warm"
        elif warmth_score < -50:
            return "Cool"
        else:
            return "Neutral"
    
    def _categorize_brightness(self, brightness: float) -> str:
        """Categorize brightness level."""
        if brightness > 0.7:
            return "Bright"
        elif brightness > 0.3:
            return "Medium"
        else:
            return "Dark"
    
    def _categorize_contrast(self, contrast: float) -> str:
        """Categorize contrast level."""
        if contrast > 0.3:
            return "High"
        elif contrast > 0.15:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_colors(self, thumbnail_data: List[Dict]) -> Dict[str, Any]:
        """Analyze color patterns across thumbnails."""
        if not thumbnail_data:
            return {}
        
        color_analysis = {
            'color_temperature_distribution': Counter(),
            'brightness_distribution': Counter(),
            'contrast_distribution': Counter(),
            'saturation_analysis': {
                'average_saturation': 0,
                'high_saturation_count': 0
            },
            'dominant_color_patterns': []
        }
        
        saturations = []
        all_colors = []
        
        for thumb in thumbnail_data:
            # Color temperature
            color_analysis['color_temperature_distribution'][thumb.get('color_temperature', 'Unknown')] += 1
            
            # Brightness
            color_analysis['brightness_distribution'][thumb.get('brightness_category', 'Unknown')] += 1
            
            # Contrast
            color_analysis['contrast_distribution'][thumb.get('contrast_category', 'Unknown')] += 1
            
            # Saturation
            saturation = thumb.get('average_saturation', 0)
            saturations.append(saturation)
            if saturation > 0.6:
                color_analysis['saturation_analysis']['high_saturation_count'] += 1
            
            # Collect dominant colors
            dominant_colors = thumb.get('dominant_colors', [])
            for color_info in dominant_colors[:3]:  # Top 3 colors
                all_colors.append(color_info.get('color_hex', '#000000'))
        
        # Calculate average saturation
        if saturations:
            color_analysis['saturation_analysis']['average_saturation'] = round(np.mean(saturations), 3)
        
        # Find most common colors
        color_counter = Counter(all_colors)
        color_analysis['most_common_colors'] = dict(color_counter.most_common(10))
        
        return color_analysis
    
    def _analyze_composition(self, thumbnail_data: List[Dict]) -> Dict[str, Any]:
        """Analyze composition patterns across thumbnails."""
        if not thumbnail_data:
            return {}
        
        composition_analysis = {
            'complexity_distribution': Counter(),
            'center_focus_analysis': {
                'thumbnails_with_center_focus': 0,
                'average_center_focus_strength': 0
            },
            'aspect_ratio_analysis': {
                'ratios': [],
                'most_common_ratio': None
            },
            'composition_effectiveness': {}
        }
        
        center_focus_values = []
        aspect_ratios = []
        
        for thumb in thumbnail_data:
            # Complexity
            complexity = thumb.get('complexity', 'Unknown')
            composition_analysis['complexity_distribution'][complexity] += 1
            
            # Center focus
            center_focus = thumb.get('center_focus', 0)
            center_focus_values.append(center_focus)
            if thumb.get('has_center_focus', False):
                composition_analysis['center_focus_analysis']['thumbnails_with_center_focus'] += 1
            
            # Aspect ratio
            aspect_ratio = thumb.get('aspect_ratio', 0)
            if aspect_ratio > 0:
                aspect_ratios.append(aspect_ratio)
        
        # Calculate averages
        if center_focus_values:
            composition_analysis['center_focus_analysis']['average_center_focus_strength'] = round(np.mean(center_focus_values), 3)
        
        if aspect_ratios:
            composition_analysis['aspect_ratio_analysis']['ratios'] = aspect_ratios
            # Find most common aspect ratio (rounded)
            rounded_ratios = [round(ratio, 1) for ratio in aspect_ratios]
            ratio_counter = Counter(rounded_ratios)
            composition_analysis['aspect_ratio_analysis']['most_common_ratio'] = ratio_counter.most_common(1)[0][0]
        
        return composition_analysis
    
    def _analyze_text_elements(self, thumbnail_data: List[Dict]) -> Dict[str, Any]:
        """Analyze text elements in thumbnails using OCR."""
        if not thumbnail_data:
            return {}
        
        text_analysis = {
            'thumbnails_with_text': 0,
            'total_text_regions': 0,
            'text_coverage_analysis': {
                'average_coverage': 0,
                'high_coverage_count': 0,
                'max_coverage': 0
            },
            'text_positioning': Counter(),
            'text_confidence_analysis': {
                'average_confidence': 0,
                'high_confidence_count': 0
            },
            'word_count_analysis': {
                'average_words_per_thumbnail': 0,
                'max_words': 0,
                'total_words': 0
            },
            'common_text_positions': [],
            'text_effectiveness': {
                'text_vs_no_text_performance': 'unknown'
            },
            'detected_text_samples': []
        }
        
        text_coverages = []
        text_confidences = []
        word_counts = []
        all_positions = []
        
        for thumb in thumbnail_data:
            # Check if text was detected
            if thumb.get('has_text_overlay', False):
                text_analysis['thumbnails_with_text'] += 1
            
            # Coverage analysis
            coverage = thumb.get('text_coverage', 0)
            text_coverages.append(coverage)
            if coverage > 20:  # More than 20% coverage
                text_analysis['text_coverage_analysis']['high_coverage_count'] += 1
            
            # Confidence analysis
            confidence = thumb.get('text_confidence', 0)
            if confidence > 0:
                text_confidences.append(confidence)
                if confidence > 70:  # High confidence threshold
                    text_analysis['text_confidence_analysis']['high_confidence_count'] += 1
            
            # Word count analysis
            word_count = thumb.get('word_count', 0)
            word_counts.append(word_count)
            text_analysis['word_count_analysis']['total_words'] += word_count
            
            # Position analysis
            position = thumb.get('text_position', 'unknown')
            text_analysis['text_positioning'][position] += 1
            if position != 'unknown':
                all_positions.append(position)
            
            # Count text regions
            text_regions = thumb.get('text_regions', [])
            text_analysis['total_text_regions'] += len(text_regions)
            
            # Collect text samples for analysis
            detected_text = thumb.get('detected_text', '')
            if detected_text and len(detected_text.strip()) > 0:
                text_analysis['detected_text_samples'].append({
                    'text': detected_text[:100] + '...' if len(detected_text) > 100 else detected_text,
                    'confidence': confidence,
                    'word_count': word_count,
                    'coverage': coverage
                })
        
        # Calculate averages and statistics
        if text_coverages:
            text_analysis['text_coverage_analysis']['average_coverage'] = round(np.mean(text_coverages), 2)
            text_analysis['text_coverage_analysis']['max_coverage'] = round(max(text_coverages), 2)
        
        if text_confidences:
            text_analysis['text_confidence_analysis']['average_confidence'] = round(np.mean(text_confidences), 2)
        
        if word_counts:
            text_analysis['word_count_analysis']['average_words_per_thumbnail'] = round(np.mean(word_counts), 1)
            text_analysis['word_count_analysis']['max_words'] = max(word_counts)
        
        # Find most common text positions
        if all_positions:
            position_counts = Counter(all_positions)
            text_analysis['common_text_positions'] = [
                {'position': pos, 'count': count, 'percentage': round((count / len(all_positions)) * 100, 1)}
                for pos, count in position_counts.most_common(5)
            ]
        
        # Text effectiveness analysis (if performance data available)
        text_analysis['text_effectiveness'] = self._analyze_text_effectiveness(thumbnail_data)
        
        # Additional insights
        text_analysis['insights'] = self._generate_text_insights(text_analysis)
        
        return text_analysis
    
    def _analyze_text_effectiveness(self, thumbnail_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how text presence correlates with performance."""
        effectiveness = {
            'analysis_available': False,
            'avg_views_with_text': 0,
            'avg_views_without_text': 0,
            'text_advantage': 'unknown'
        }
        
        # Check if we have performance data (views)
        thumbnails_with_views = [t for t in thumbnail_data if t.get('video_views', 0) > 0]
        
        if len(thumbnails_with_views) > 5:  # Need reasonable sample size
            with_text = [t for t in thumbnails_with_views if t.get('has_text_overlay', False)]
            without_text = [t for t in thumbnails_with_views if not t.get('has_text_overlay', False)]
            
            if len(with_text) > 0 and len(without_text) > 0:
                avg_views_with = np.mean([t['video_views'] for t in with_text])
                avg_views_without = np.mean([t['video_views'] for t in without_text])
                
                effectiveness.update({
                    'analysis_available': True,
                    'avg_views_with_text': int(avg_views_with),
                    'avg_views_without_text': int(avg_views_without),
                    'text_advantage': 'positive' if avg_views_with > avg_views_without else 'negative',
                    'performance_difference_percentage': round(((avg_views_with - avg_views_without) / avg_views_without) * 100, 1) if avg_views_without > 0 else 0
                })
        
        return effectiveness
    
    def _generate_text_insights(self, text_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about text usage in thumbnails."""
        insights = []
        
        # Text presence insights
        text_percentage = (text_analysis['thumbnails_with_text'] / max(1, len(text_analysis.get('detected_text_samples', [])))) * 100
        if text_percentage > 70:
            insights.append("High text usage - most thumbnails contain text overlays")
        elif text_percentage < 30:
            insights.append("Low text usage - consider adding text overlays for better engagement")
        
        # Coverage insights
        avg_coverage = text_analysis['text_coverage_analysis'].get('average_coverage', 0)
        if avg_coverage > 25:
            insights.append("Text coverage is high - ensure readability isn't compromised")
        elif avg_coverage < 5:
            insights.append("Text coverage is low - text might be too small to read")
        
        # Position insights
        common_positions = text_analysis.get('common_text_positions', [])
        if common_positions:
            most_common = common_positions[0]['position']
            insights.append(f"Most common text position: {most_common}")
        
        # Confidence insights
        avg_confidence = text_analysis['text_confidence_analysis'].get('average_confidence', 0)
        if avg_confidence < 50:
            insights.append("Low text detection confidence - text might be stylized or hard to read")
        
        # Word count insights
        avg_words = text_analysis['word_count_analysis'].get('average_words_per_thumbnail', 0)
        if avg_words > 10:
            insights.append("High word count - consider simplifying text for better impact")
        elif avg_words > 0 and avg_words < 3:
            insights.append("Low word count - concise text approach")
        
        return insights
    
    def _analyze_faces(self, thumbnail_data: List[Dict]) -> Dict[str, Any]:
        """Analyze face presence in thumbnails using face recognition."""
        face_analysis = {
            'thumbnails_with_faces': 0,
            'total_faces_detected': 0,
            'average_face_size': 0,
            'face_positioning': Counter(),
            'face_details': [],
            'dominant_face_position': 'none',
            'multiple_faces_count': 0,
            'face_size_distribution': {
                'small': 0,    # < 5% of image
                'medium': 0,   # 5-15% of image  
                'large': 0     # > 15% of image
            }
        }
        
        if not FACE_DETECTION_AVAILABLE:
            # Fallback to basic face detection using OpenCV Haar Cascades
            return self._basic_face_detection(thumbnail_data)
        
        face_sizes = []
        all_positions = []
        
        for thumbnail in thumbnail_data:
            try:
                # Get image path or download if needed
                image_path = self._get_image_for_face_detection(thumbnail)
                if not image_path:
                    continue
                
                # Load image for face_recognition
                image = face_recognition.load_image_file(image_path)
                
                # Find face locations
                face_locations = face_recognition.face_locations(image, model="hog")  # "hog" is faster, "cnn" is more accurate
                
                if face_locations:
                    face_analysis['thumbnails_with_faces'] += 1
                    face_analysis['total_faces_detected'] += len(face_locations)
                    
                    if len(face_locations) > 1:
                        face_analysis['multiple_faces_count'] += 1
                    
                    # Analyze each face
                    image_height, image_width = image.shape[:2]
                    image_area = image_height * image_width
                    
                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        # Calculate face metrics
                        face_width = right - left
                        face_height = bottom - top
                        face_area = face_width * face_height
                        face_size_percentage = (face_area / image_area) * 100
                        
                        face_sizes.append(face_size_percentage)
                        
                        # Determine face position
                        center_x = (left + right) / 2
                        center_y = (top + bottom) / 2
                        
                        # Normalize coordinates
                        norm_x = center_x / image_width
                        norm_y = center_y / image_height
                        
                        position = self._determine_face_position(norm_x, norm_y)
                        face_analysis['face_positioning'][position] += 1
                        all_positions.append(position)
                        
                        # Categorize face size
                        if face_size_percentage < 5:
                            size_category = 'small'
                        elif face_size_percentage < 15:
                            size_category = 'medium'
                        else:
                            size_category = 'large'
                        
                        face_analysis['face_size_distribution'][size_category] += 1
                        
                        # Store detailed face info
                        face_detail = {
                            'thumbnail_index': len(face_analysis['face_details']),
                            'face_index': i,
                            'bbox': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                            'size_percentage': round(face_size_percentage, 2),
                            'size_category': size_category,
                            'position': position,
                            'center_coordinates': {'x': norm_x, 'y': norm_y}
                        }
                        
                        # Try to get face encodings for additional analysis
                        try:
                            face_encodings = face_recognition.face_encodings(image, [face_locations[i]])
                            if face_encodings:
                                face_detail['has_encoding'] = True
                                # Could be used for face similarity analysis
                            else:
                                face_detail['has_encoding'] = False
                        except:
                            face_detail['has_encoding'] = False
                        
                        face_analysis['face_details'].append(face_detail)
                
            except Exception as e:
                logger.warning(f"Error in face detection for thumbnail: {e}")
                continue
        
        # Calculate averages and dominant patterns
        if face_sizes:
            face_analysis['average_face_size'] = round(np.mean(face_sizes), 2)
        
        if all_positions:
            face_analysis['dominant_face_position'] = max(set(all_positions), key=all_positions.count)
        
        # Calculate face presence percentage
        total_thumbnails = len(thumbnail_data)
        if total_thumbnails > 0:
            face_analysis['face_presence_percentage'] = round(
                (face_analysis['thumbnails_with_faces'] / total_thumbnails) * 100, 1
            )
        
        return face_analysis
    
    def _basic_face_detection(self, thumbnail_data: List[Dict]) -> Dict[str, Any]:
        """Basic face detection fallback using OpenCV Haar Cascades."""
        face_analysis = {
            'thumbnails_with_faces': 0,
            'total_faces_detected': 0,
            'average_face_size': 0,
            'face_positioning': Counter(),
            'face_details': [],
            'detection_method': 'opencv_haar_cascade'
        }
        
        # Load OpenCV face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            logger.error("Could not load OpenCV face cascade")
            return face_analysis
        
        face_sizes = []
        all_positions = []
        
        for thumbnail in thumbnail_data:
            try:
                # Get image for detection
                image_path = self._get_image_for_face_detection(thumbnail)
                if not image_path:
                    continue
                
                # Load image with OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                image_area = height * width
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    face_analysis['thumbnails_with_faces'] += 1
                    face_analysis['total_faces_detected'] += len(faces)
                    
                    for (x, y, w, h) in faces:
                        face_area = w * h
                        face_size_percentage = (face_area / image_area) * 100
                        face_sizes.append(face_size_percentage)
                        
                        # Determine position
                        center_x = x + w/2
                        center_y = y + h/2
                        norm_x = center_x / width
                        norm_y = center_y / height
                        
                        position = self._determine_face_position(norm_x, norm_y)
                        face_analysis['face_positioning'][position] += 1
                        all_positions.append(position)
                        
                        # Store face details
                        face_analysis['face_details'].append({
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'size_percentage': round(face_size_percentage, 2),
                            'position': position,
                            'center_coordinates': {'x': norm_x, 'y': norm_y}
                        })
                        
            except Exception as e:
                logger.warning(f"Error in basic face detection: {e}")
                continue
        
        # Calculate averages
        if face_sizes:
            face_analysis['average_face_size'] = round(np.mean(face_sizes), 2)
        
        if all_positions:
            face_analysis['dominant_face_position'] = max(set(all_positions), key=all_positions.count)
        
        return face_analysis
    
    def _get_image_for_face_detection(self, thumbnail: Dict) -> Optional[str]:
        """Get image path for face detection, download if necessary."""
        # Check if we have a cached image
        video_title = thumbnail.get('video_title', 'unknown')
        safe_filename = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        cache_path = os.path.join(self.cache_dir, f"{safe_filename[:50]}.jpg")
        
        if os.path.exists(cache_path):
            return cache_path
        
        # Try to download if we have URL
        thumbnail_url = thumbnail.get('thumbnail_url')
        if thumbnail_url:
            try:
                image = self._download_thumbnail(thumbnail_url, safe_filename)
                if image:
                    image.save(cache_path, 'JPEG')
                    return cache_path
            except Exception as e:
                logger.warning(f"Could not download thumbnail for face detection: {e}")
        
        return None
    
    def _determine_face_position(self, norm_x: float, norm_y: float) -> str:
        """Determine face position based on normalized coordinates."""
        # Similar to text position but optimized for faces
        if norm_y < 0.4:  # Faces are often in upper portion
            if norm_x < 0.33:
                return 'top-left'
            elif norm_x > 0.67:
                return 'top-right'
            else:
                return 'top-center'
        elif norm_y > 0.6:
            if norm_x < 0.33:
                return 'bottom-left'
            elif norm_x > 0.67:
                return 'bottom-right'
            else:
                return 'bottom-center'
        else:
            if norm_x < 0.33:
                return 'middle-left'
            elif norm_x > 0.67:
                return 'middle-right'
            else:
                return 'center'
    
    def _correlate_with_performance(self, thumbnail_data: List[Dict], videos_data: List[Dict]) -> Dict[str, Any]:
        """Correlate thumbnail features with video performance."""
        if not thumbnail_data or not videos_data:
            return {}
        
        # Create DataFrame for analysis
        df_data = []
        for i, thumb in enumerate(thumbnail_data):
            if i < len(videos_data):
                video = videos_data[i]
                df_data.append({
                    'views': video.get('views', 0),
                    'brightness': thumb.get('brightness', 0),
                    'contrast': thumb.get('contrast', 0),
                    'saturation': thumb.get('average_saturation', 0),
                    'center_focus': thumb.get('center_focus', 0),
                    'complexity': thumb.get('complexity', 'Unknown'),
                    'color_temperature': thumb.get('color_temperature', 'Unknown')
                })
        
        if not df_data:
            return {}
        
        df = pd.DataFrame(df_data)
        
        correlations = {}
        
        # Calculate correlations for numerical features
        numerical_features = ['brightness', 'contrast', 'saturation', 'center_focus']
        for feature in numerical_features:
            if feature in df.columns and df[feature].std() > 0:
                corr = df['views'].corr(df[feature])
                correlations[f'{feature}_correlation'] = round(corr, 3) if not pd.isna(corr) else 0
        
        # Analyze categorical features
        categorical_analysis = {}
        
        # Complexity vs performance
        if 'complexity' in df.columns:
            complexity_performance = df.groupby('complexity')['views'].mean().to_dict()
            categorical_analysis['complexity_performance'] = complexity_performance
        
        # Color temperature vs performance
        if 'color_temperature' in df.columns:
            temp_performance = df.groupby('color_temperature')['views'].mean().to_dict()
            categorical_analysis['color_temperature_performance'] = temp_performance
        
        return {
            'feature_correlations': correlations,
            'categorical_analysis': categorical_analysis,
            'sample_size': len(df_data)
        }
    
    def _analyze_without_download(self, videos_data: List[Dict]) -> Dict[str, Any]:
        """Perform basic analysis without downloading thumbnails."""
        return {
            'overview': {
                'total_videos': len(videos_data),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'thumbnails_analyzed': 0,
                'note': 'Analysis performed without downloading thumbnails'
            },
            'basic_analysis': {
                'videos_with_thumbnail_urls': sum(1 for v in videos_data if self._extract_thumbnail_url(v)),
                'thumbnail_url_patterns': self._analyze_url_patterns(videos_data)
            },
            'recommendations': [
                'Enable thumbnail downloading for detailed visual analysis',
                'Ensure video URLs contain valid YouTube video IDs for thumbnail extraction'
            ]
        }
    
    def _analyze_url_patterns(self, videos_data: List[Dict]) -> Dict[str, Any]:
        """Analyze thumbnail URL patterns."""
        patterns = {
            'youtube_standard': 0,
            'youtube_maxres': 0,
            'other': 0
        }
        
        for video in videos_data:
            url = self._extract_thumbnail_url(video)
            if url:
                if 'maxresdefault.jpg' in url:
                    patterns['youtube_maxres'] += 1
                elif 'img.youtube.com' in url:
                    patterns['youtube_standard'] += 1
                else:
                    patterns['other'] += 1
        
        return patterns
    
    def _identify_trending_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns specific to trending thumbnails."""
        trending_patterns = {
            'color_trends': {},
            'composition_trends': {},
            'effectiveness_indicators': []
        }
        
        # Analyze color trends in trending content
        color_analysis = analysis.get('color_analysis', {})
        if color_analysis:
            temp_dist = color_analysis.get('color_temperature_distribution', {})
            if temp_dist:
                most_common_temp = max(temp_dist, key=temp_dist.get)
                trending_patterns['color_trends']['dominant_temperature'] = most_common_temp
            
            brightness_dist = color_analysis.get('brightness_distribution', {})
            if brightness_dist:
                most_common_brightness = max(brightness_dist, key=brightness_dist.get)
                trending_patterns['color_trends']['dominant_brightness'] = most_common_brightness
        
        # Analyze composition trends
        composition_analysis = analysis.get('composition_analysis', {})
        if composition_analysis:
            complexity_dist = composition_analysis.get('complexity_distribution', {})
            if complexity_dist:
                most_common_complexity = max(complexity_dist, key=complexity_dist.get)
                trending_patterns['composition_trends']['dominant_complexity'] = most_common_complexity
        
        return trending_patterns
    
    def _identify_viral_indicators(self, analysis: Dict[str, Any], trending_videos: List[Dict]) -> List[str]:
        """Identify visual indicators that correlate with viral success."""
        indicators = []
        
        # High contrast tends to grab attention
        composition = analysis.get('composition_analysis', {})
        contrast_dist = composition.get('complexity_distribution', {})
        if contrast_dist.get('High', 0) > contrast_dist.get('Low', 0):
            indicators.append("High visual complexity/contrast appears in most trending thumbnails")
        
        # Bright thumbnails often perform well
        color_analysis = analysis.get('color_analysis', {})
        brightness_dist = color_analysis.get('brightness_distribution', {})
        if brightness_dist.get('Bright', 0) > brightness_dist.get('Dark', 0):
            indicators.append("Bright thumbnails dominate trending content")
        
        # Warm colors often attract clicks
        temp_dist = color_analysis.get('color_temperature_distribution', {})
        if temp_dist.get('Warm', 0) > temp_dist.get('Cool', 0):
            indicators.append("Warm color schemes are prevalent in trending thumbnails")
        
        return indicators
    
    def _analyze_category_patterns(self, trending_videos: List[Dict]) -> Dict[str, Any]:
        """Analyze thumbnail patterns by video category."""
        # This would require category information in the video data
        # For now, return placeholder
        return {
            'note': 'Category-specific analysis requires video category data',
            'available_categories': []
        }
    
    def _compare_color_strategies(self, channel_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare color strategies across channels."""
        comparison = {
            'color_temperature_preferences': {},
            'brightness_strategies': {},
            'saturation_approaches': {}
        }
        
        for channel, analysis in channel_analyses.items():
            color_analysis = analysis.get('color_analysis', {})
            
            # Color temperature
            temp_dist = color_analysis.get('color_temperature_distribution', {})
            if temp_dist:
                dominant_temp = max(temp_dist, key=temp_dist.get)
                comparison['color_temperature_preferences'][channel] = dominant_temp
            
            # Brightness
            brightness_dist = color_analysis.get('brightness_distribution', {})
            if brightness_dist:
                dominant_brightness = max(brightness_dist, key=brightness_dist.get)
                comparison['brightness_strategies'][channel] = dominant_brightness
            
            # Saturation
            sat_analysis = color_analysis.get('saturation_analysis', {})
            avg_sat = sat_analysis.get('average_saturation', 0)
            comparison['saturation_approaches'][channel] = 'High' if avg_sat > 0.6 else 'Medium' if avg_sat > 0.3 else 'Low'
        
        return comparison
    
    def _compare_compositions(self, channel_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare composition strategies across channels."""
        comparison = {
            'complexity_preferences': {},
            'center_focus_usage': {},
            'aspect_ratio_consistency': {}
        }
        
        for channel, analysis in channel_analyses.items():
            composition_analysis = analysis.get('composition_analysis', {})
            
            # Complexity
            complexity_dist = composition_analysis.get('complexity_distribution', {})
            if complexity_dist:
                dominant_complexity = max(complexity_dist, key=complexity_dist.get)
                comparison['complexity_preferences'][channel] = dominant_complexity
            
            # Center focus
            center_analysis = composition_analysis.get('center_focus_analysis', {})
            center_count = center_analysis.get('thumbnails_with_center_focus', 0)
            total_thumbs = analysis.get('overview', {}).get('thumbnails_analyzed', 1)
            center_percentage = (center_count / total_thumbs) * 100 if total_thumbs > 0 else 0
            comparison['center_focus_usage'][channel] = f"{center_percentage:.1f}%"
        
        return comparison
    
    def _compare_thumbnail_performance(self, channel_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare thumbnail performance across channels."""
        performance_comparison = {}
        
        for channel, analysis in channel_analyses.items():
            perf_corr = analysis.get('performance_correlation', {})
            correlations = perf_corr.get('feature_correlations', {})
            
            performance_comparison[channel] = {
                'brightness_correlation': correlations.get('brightness_correlation', 0),
                'contrast_correlation': correlations.get('contrast_correlation', 0),
                'saturation_correlation': correlations.get('saturation_correlation', 0),
                'sample_size': perf_corr.get('sample_size', 0)
            }
        
        return performance_comparison
    
    def _identify_best_practices(self, channel_analyses: Dict[str, Dict]) -> List[str]:
        """Identify best practices from successful channels."""
        best_practices = []
        
        # Find channels with strong correlations
        strong_correlations = {}
        for channel, analysis in channel_analyses.items():
            perf_corr = analysis.get('performance_correlation', {})
            correlations = perf_corr.get('feature_correlations', {})
            
            for feature, corr in correlations.items():
                if abs(corr) > 0.3:  # Strong correlation threshold
                    if feature not in strong_correlations:
                        strong_correlations[feature] = []
                    strong_correlations[feature].append((channel, corr))
        
        # Generate best practices based on correlations
        for feature, channel_corrs in strong_correlations.items():
            if len(channel_corrs) >= 2:  # Multiple channels show this pattern
                avg_corr = np.mean([corr for _, corr in channel_corrs])
                if avg_corr > 0.3:
                    best_practices.append(f"Higher {feature.replace('_correlation', '')} correlates with better performance")
                elif avg_corr < -0.3:
                    best_practices.append(f"Lower {feature.replace('_correlation', '')} correlates with better performance")
        
        return best_practices
    
    def _generate_thumbnail_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on thumbnail analysis."""
        recommendations = []
        
        # Color recommendations
        color_analysis = analysis.get('color_analysis', {})
        if color_analysis:
            temp_dist = color_analysis.get('color_temperature_distribution', {})
            if temp_dist and temp_dist.get('Warm', 0) > temp_dist.get('Cool', 0):
                recommendations.append("Consider using warm color schemes - they appear more frequently in successful thumbnails")
            
            brightness_dist = color_analysis.get('brightness_distribution', {})
            if brightness_dist and brightness_dist.get('Bright', 0) > brightness_dist.get('Dark', 0):
                recommendations.append("Bright thumbnails tend to perform better - avoid overly dark images")
            
            sat_analysis = color_analysis.get('saturation_analysis', {})
            avg_sat = sat_analysis.get('average_saturation', 0)
            if avg_sat > 0.5:
                recommendations.append("High saturation colors are common in successful thumbnails")
        
        # Composition recommendations
        composition_analysis = analysis.get('composition_analysis', {})
        if composition_analysis:
            complexity_dist = composition_analysis.get('complexity_distribution', {})
            if complexity_dist and complexity_dist.get('High', 0) > complexity_dist.get('Low', 0):
                recommendations.append("Complex, detailed thumbnails often outperform simple ones")
            
            center_analysis = composition_analysis.get('center_focus_analysis', {})
            center_count = center_analysis.get('thumbnails_with_center_focus', 0)
            total_analyzed = analysis.get('overview', {}).get('thumbnails_analyzed', 1)
            if center_count / total_analyzed > 0.6:
                recommendations.append("Focus attention on the center of your thumbnail for better engagement")
        
        # Performance correlation recommendations
        perf_corr = analysis.get('performance_correlation', {})
        if perf_corr:
            correlations = perf_corr.get('feature_correlations', {})
            
            for feature, corr in correlations.items():
                if corr > 0.3:
                    feature_name = feature.replace('_correlation', '')
                    recommendations.append(f"Higher {feature_name} shows positive correlation with performance")
                elif corr < -0.3:
                    feature_name = feature.replace('_correlation', '')
                    recommendations.append(f"Lower {feature_name} shows positive correlation with performance")
        
        return recommendations
    
    def _generate_trending_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations specific to trending content."""
        recommendations = self._generate_thumbnail_recommendations(analysis)
        
        # Add trending-specific recommendations
        trending_patterns = analysis.get('trending_patterns', {})
        if trending_patterns:
            color_trends = trending_patterns.get('color_trends', {})
            if color_trends.get('dominant_temperature'):
                recommendations.append(f"Trending content favors {color_trends['dominant_temperature'].lower()} color schemes")
            
            if color_trends.get('dominant_brightness'):
                recommendations.append(f"Most trending thumbnails use {color_trends['dominant_brightness'].lower()} brightness levels")
        
        viral_indicators = analysis.get('viral_indicators', [])
        recommendations.extend(viral_indicators)
        
        return recommendations
    
    def _generate_comparison_recommendations(self, channel_analyses: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on channel comparison."""
        recommendations = []
        
        # Find most successful patterns across channels
        all_recommendations = []
        for analysis in channel_analyses.values():
            all_recommendations.extend(analysis.get('recommendations', []))
        
        # Count common recommendations
        recommendation_counts = Counter(all_recommendations)
        common_recommendations = [rec for rec, count in recommendation_counts.items() if count >= 2]
        
        recommendations.extend(common_recommendations)
        
        # Add comparison-specific insights
        recommendations.append("Study successful channels' thumbnail strategies and adapt elements that work")
        recommendations.append("Maintain consistency in your thumbnail style while testing new approaches")
        
        return recommendations
    
    def get_analysis_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all analysis results."""
        return self.analysis_results
    
    def clear_cache(self):
        """Clear thumbnail cache."""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Thumbnail cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about thumbnail cache."""
        try:
            cache_files = os.listdir(self.cache_dir)
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
            
            return {
                'cache_directory': self.cache_dir,
                'cached_thumbnails': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_files': cache_files[:10]  # Show first 10 files
            }
        except Exception as e:
            return {'error': f"Could not get cache info: {e}"}
