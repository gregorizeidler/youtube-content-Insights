"""
Sentiment analysis module for analyzing YouTube comments and video reception.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import re
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment of YouTube comments and video reception."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_results = {}
    
    def analyze_video_sentiment(self, comments_data: List[Dict], video_info: Dict = None) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on video comments.
        
        Args:
            comments_data: List of comment dictionaries
            video_info: Optional video information dictionary
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not comments_data:
            return {"error": "No comments data provided"}
        
        logger.info(f"Analyzing sentiment for {len(comments_data)} comments")
        
        # Convert to DataFrame for easier processing
        comments_df = pd.DataFrame(comments_data)
        
        # Perform sentiment analysis
        sentiment_scores = self._calculate_sentiment_scores(comments_df)
        comments_df = pd.concat([comments_df, sentiment_scores], axis=1)
        
        analysis = {
            'video_info': video_info or {'title': 'Unknown Video'},
            'overview': self._generate_sentiment_overview(comments_df),
            'detailed_analysis': self._perform_detailed_analysis(comments_df),
            'emotion_analysis': self._analyze_emotions(comments_df),
            'topic_sentiment': self._analyze_topic_sentiment(comments_df),
            'temporal_analysis': self._analyze_temporal_sentiment(comments_df),
            'engagement_correlation': self._analyze_engagement_sentiment_correlation(comments_df),
            'recommendations': self._generate_sentiment_recommendations(comments_df)
        }
        
        # Store results
        video_title = video_info.get('title', 'Unknown Video') if video_info else 'Unknown Video'
        self.sentiment_results[video_title] = analysis
        
        return analysis
    
    def compare_video_sentiments(self, videos_sentiment_data: List[Dict]) -> Dict[str, Any]:
        """
        Compare sentiment across multiple videos.
        
        Args:
            videos_sentiment_data: List of dictionaries containing video info and comments
            
        Returns:
            Dictionary with comparative sentiment analysis
        """
        logger.info(f"Comparing sentiment across {len(videos_sentiment_data)} videos")
        
        if len(videos_sentiment_data) < 2:
            return {"error": "At least 2 videos required for comparison"}
        
        # Analyze each video individually
        individual_analyses = []
        for video_data in videos_sentiment_data:
            video_info = video_data.get('video_info', {})
            comments_data = video_data.get('comments_data', [])
            analysis = self.analyze_video_sentiment(comments_data, video_info)
            individual_analyses.append(analysis)
        
        # Perform comparison
        comparison = {
            'comparison_overview': {
                'videos_compared': len(videos_sentiment_data),
                'total_comments_analyzed': sum(len(v.get('comments_data', [])) for v in videos_sentiment_data),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'sentiment_comparison': self._compare_overall_sentiment(individual_analyses),
            'engagement_comparison': self._compare_engagement_patterns(individual_analyses),
            'topic_comparison': self._compare_topic_sentiments(individual_analyses),
            'audience_insights': self._generate_audience_insights(individual_analyses),
            'recommendations': self._generate_comparison_recommendations(individual_analyses)
        }
        
        return comparison
    
    def _calculate_sentiment_scores(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment scores using multiple methods."""
        sentiment_data = []
        
        for comment_text in comments_df['text']:
            # Clean the text
            cleaned_text = self._clean_text(comment_text)
            
            # TextBlob sentiment
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(cleaned_text)
            
            # Custom sentiment indicators
            custom_sentiment = self._calculate_custom_sentiment(cleaned_text)
            
            sentiment_data.append({
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'custom_sentiment': custom_sentiment,
                'cleaned_text': cleaned_text
            })
        
        return pd.DataFrame(sentiment_data)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s:;=\-\)\(\[\]<>]', '', text)
        
        return text
    
    def _calculate_custom_sentiment(self, text: str) -> float:
        """Calculate custom sentiment score based on domain-specific indicators."""
        positive_indicators = [
            'love', 'amazing', 'awesome', 'great', 'excellent', 'fantastic', 'wonderful',
            'perfect', 'brilliant', 'outstanding', 'incredible', 'superb', 'magnificent',
            'thank you', 'thanks', 'appreciate', 'helpful', 'useful', 'informative',
            'inspiring', 'motivating', 'educational', 'learned', 'subscribe', 'subscribed'
        ]
        
        negative_indicators = [
            'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'stupid', 'dumb',
            'boring', 'waste', 'disappointed', 'annoying', 'frustrating', 'confusing',
            'misleading', 'clickbait', 'fake', 'lies', 'dislike', 'unsubscribe',
            'worst', 'pathetic', 'useless', 'pointless'
        ]
        
        # Count positive and negative indicators
        positive_count = sum(1 for word in positive_indicators if word in text)
        negative_count = sum(1 for word in negative_indicators if word in text)
        
        # Calculate score
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_indicators
    
    def _generate_sentiment_overview(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall sentiment overview."""
        # Classify sentiments
        comments_df['sentiment_category'] = comments_df['vader_compound'].apply(
            lambda x: 'Positive' if x >= 0.05 else 'Negative' if x <= -0.05 else 'Neutral'
        )
        
        sentiment_counts = comments_df['sentiment_category'].value_counts()
        total_comments = len(comments_df)
        
        overview = {
            'total_comments_analyzed': total_comments,
            'sentiment_distribution': {
                'positive': int(sentiment_counts.get('Positive', 0)),
                'negative': int(sentiment_counts.get('Negative', 0)),
                'neutral': int(sentiment_counts.get('Neutral', 0))
            },
            'sentiment_percentages': {
                'positive': round((sentiment_counts.get('Positive', 0) / total_comments) * 100, 1),
                'negative': round((sentiment_counts.get('Negative', 0) / total_comments) * 100, 1),
                'neutral': round((sentiment_counts.get('Neutral', 0) / total_comments) * 100, 1)
            },
            'overall_sentiment_score': round(comments_df['vader_compound'].mean(), 3),
            'sentiment_intensity': round(comments_df['vader_compound'].std(), 3),
            'overall_classification': self._classify_overall_sentiment(comments_df['vader_compound'].mean())
        }
        
        return overview
    
    def _perform_detailed_analysis(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed sentiment analysis."""
        detailed = {
            'score_statistics': {
                'mean_compound_score': round(comments_df['vader_compound'].mean(), 3),
                'median_compound_score': round(comments_df['vader_compound'].median(), 3),
                'std_compound_score': round(comments_df['vader_compound'].std(), 3),
                'min_compound_score': round(comments_df['vader_compound'].min(), 3),
                'max_compound_score': round(comments_df['vader_compound'].max(), 3)
            },
            'textblob_analysis': {
                'mean_polarity': round(comments_df['textblob_polarity'].mean(), 3),
                'mean_subjectivity': round(comments_df['textblob_subjectivity'].mean(), 3),
                'subjectivity_interpretation': self._interpret_subjectivity(comments_df['textblob_subjectivity'].mean())
            },
            'extreme_sentiments': {
                'most_positive_comments': self._get_extreme_comments(comments_df, 'positive', 3),
                'most_negative_comments': self._get_extreme_comments(comments_df, 'negative', 3)
            },
            'sentiment_consistency': {
                'consistency_score': self._calculate_sentiment_consistency(comments_df),
                'polarization_level': self._calculate_polarization(comments_df)
            }
        }
        
        return detailed
    
    def _analyze_emotions(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze specific emotions in comments."""
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'delighted', 'cheerful', 'elated'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'pissed'],
            'sadness': ['sad', 'depressed', 'disappointed', 'upset', 'heartbroken', 'miserable'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'frightened'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled']
        }
        
        emotion_counts = {}
        for emotion, keywords in emotion_keywords.items():
            count = 0
            for text in comments_df['cleaned_text']:
                if any(keyword in text for keyword in keywords):
                    count += 1
            emotion_counts[emotion] = count
        
        total_emotional_comments = sum(emotion_counts.values())
        
        emotion_analysis = {
            'emotion_counts': emotion_counts,
            'emotion_percentages': {
                emotion: round((count / len(comments_df)) * 100, 1) 
                for emotion, count in emotion_counts.items()
            },
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'None',
            'emotional_intensity': round((total_emotional_comments / len(comments_df)) * 100, 1)
        }
        
        return emotion_analysis
    
    def _analyze_topic_sentiment(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment around specific topics mentioned in comments."""
        # Extract common topics/keywords
        all_text = ' '.join(comments_df['cleaned_text'])
        words = word_tokenize(all_text)
        words = [word for word in words if word.lower() not in self.stop_words and len(word) > 3]
        
        # Get most common words as topics
        word_freq = Counter(words)
        top_topics = [word for word, count in word_freq.most_common(10)]
        
        topic_sentiments = {}
        for topic in top_topics:
            topic_comments = comments_df[comments_df['cleaned_text'].str.contains(topic, case=False, na=False)]
            if len(topic_comments) > 0:
                avg_sentiment = topic_comments['vader_compound'].mean()
                topic_sentiments[topic] = {
                    'average_sentiment': round(avg_sentiment, 3),
                    'comment_count': len(topic_comments),
                    'sentiment_classification': self._classify_overall_sentiment(avg_sentiment)
                }
        
        return {
            'topic_sentiments': topic_sentiments,
            'most_positive_topic': max(topic_sentiments.items(), key=lambda x: x[1]['average_sentiment'])[0] if topic_sentiments else 'None',
            'most_negative_topic': min(topic_sentiments.items(), key=lambda x: x[1]['average_sentiment'])[0] if topic_sentiments else 'None'
        }
    
    def _analyze_temporal_sentiment(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how sentiment changes over time (if timestamp data is available)."""
        temporal_analysis = {
            'temporal_data_available': 'time' in comments_df.columns,
            'sentiment_trend': 'Unable to determine - no timestamp data'
        }
        
        if 'time' in comments_df.columns:
            try:
                # Sort by time and calculate rolling sentiment
                comments_df_sorted = comments_df.sort_values('time')
                
                # Calculate rolling average sentiment (window of 10 comments)
                window_size = min(10, len(comments_df) // 4)
                if window_size > 1:
                    rolling_sentiment = comments_df_sorted['vader_compound'].rolling(window=window_size).mean()
                    
                    # Determine trend
                    first_half_avg = rolling_sentiment.iloc[:len(rolling_sentiment)//2].mean()
                    second_half_avg = rolling_sentiment.iloc[len(rolling_sentiment)//2:].mean()
                    
                    if second_half_avg > first_half_avg + 0.1:
                        trend = "Improving"
                    elif second_half_avg < first_half_avg - 0.1:
                        trend = "Declining"
                    else:
                        trend = "Stable"
                    
                    temporal_analysis.update({
                        'sentiment_trend': trend,
                        'early_sentiment': round(first_half_avg, 3),
                        'late_sentiment': round(second_half_avg, 3),
                        'trend_strength': round(abs(second_half_avg - first_half_avg), 3)
                    })
            except Exception as e:
                logger.warning(f"Could not perform temporal analysis: {e}")
        
        return temporal_analysis
    
    def _analyze_engagement_sentiment_correlation(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between comment sentiment and engagement (likes)."""
        correlation_analysis = {
            'likes_data_available': 'likes' in comments_df.columns
        }
        
        if 'likes' in comments_df.columns:
            try:
                # Convert likes to numeric
                comments_df['likes_numeric'] = pd.to_numeric(comments_df['likes'], errors='coerce')
                
                # Calculate correlation
                sentiment_likes_corr = comments_df['vader_compound'].corr(comments_df['likes_numeric'])
                
                # Analyze sentiment by like ranges
                comments_df['like_category'] = pd.cut(comments_df['likes_numeric'], 
                                                    bins=[-1, 0, 5, 20, float('inf')], 
                                                    labels=['No likes', 'Few likes (1-5)', 'Some likes (6-20)', 'Many likes (20+)'])
                
                sentiment_by_likes = comments_df.groupby('like_category')['vader_compound'].mean()
                
                correlation_analysis.update({
                    'sentiment_likes_correlation': round(sentiment_likes_corr, 3),
                    'correlation_strength': self._interpret_correlation(sentiment_likes_corr),
                    'sentiment_by_like_ranges': sentiment_by_likes.to_dict(),
                    'highly_liked_comments_sentiment': round(comments_df[comments_df['likes_numeric'] > 10]['vader_compound'].mean(), 3) if len(comments_df[comments_df['likes_numeric'] > 10]) > 0 else 'N/A'
                })
            except Exception as e:
                logger.warning(f"Could not analyze engagement correlation: {e}")
                correlation_analysis['error'] = str(e)
        
        return correlation_analysis
    
    def _get_extreme_comments(self, comments_df: pd.DataFrame, sentiment_type: str, count: int) -> List[Dict]:
        """Get the most positive or negative comments."""
        if sentiment_type == 'positive':
            extreme_comments = comments_df.nlargest(count, 'vader_compound')
        else:
            extreme_comments = comments_df.nsmallest(count, 'vader_compound')
        
        return [
            {
                'text': row['text'][:200] + '...' if len(row['text']) > 200 else row['text'],
                'sentiment_score': round(row['vader_compound'], 3),
                'author': row.get('author', 'Unknown')
            }
            for _, row in extreme_comments.iterrows()
        ]
    
    def _calculate_sentiment_consistency(self, comments_df: pd.DataFrame) -> float:
        """Calculate how consistent the sentiment is across comments."""
        sentiment_std = comments_df['vader_compound'].std()
        # Convert to 0-100 scale where 100 is most consistent
        consistency_score = max(0, 100 - (sentiment_std * 100))
        return round(consistency_score, 1)
    
    def _calculate_polarization(self, comments_df: pd.DataFrame) -> str:
        """Calculate the level of polarization in comments."""
        positive_count = len(comments_df[comments_df['vader_compound'] >= 0.05])
        negative_count = len(comments_df[comments_df['vader_compound'] <= -0.05])
        neutral_count = len(comments_df) - positive_count - negative_count
        
        total = len(comments_df)
        extreme_percentage = ((positive_count + negative_count) / total) * 100
        
        if extreme_percentage > 70:
            return "Highly Polarized"
        elif extreme_percentage > 50:
            return "Moderately Polarized"
        else:
            return "Low Polarization"
    
    def _classify_overall_sentiment(self, compound_score: float) -> str:
        """Classify overall sentiment based on compound score."""
        if compound_score >= 0.5:
            return "Very Positive"
        elif compound_score >= 0.05:
            return "Positive"
        elif compound_score > -0.05:
            return "Neutral"
        elif compound_score > -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def _interpret_subjectivity(self, subjectivity_score: float) -> str:
        """Interpret TextBlob subjectivity score."""
        if subjectivity_score > 0.7:
            return "Highly Subjective (Opinion-based)"
        elif subjectivity_score > 0.3:
            return "Moderately Subjective"
        else:
            return "Mostly Objective (Fact-based)"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            return "Strong"
        elif abs_corr > 0.3:
            return "Moderate"
        elif abs_corr > 0.1:
            return "Weak"
        else:
            return "Very Weak/None"
    
    def _compare_overall_sentiment(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare overall sentiment across videos."""
        comparison = {
            'sentiment_rankings': [],
            'sentiment_distribution_comparison': [],
            'engagement_patterns': []
        }
        
        for analysis in analyses:
            video_title = analysis['video_info'].get('title', 'Unknown')
            overview = analysis['overview']
            
            comparison['sentiment_rankings'].append({
                'video': video_title,
                'overall_score': overview['overall_sentiment_score'],
                'classification': overview['overall_classification'],
                'positive_percentage': overview['sentiment_percentages']['positive']
            })
            
            comparison['sentiment_distribution_comparison'].append({
                'video': video_title,
                'positive': overview['sentiment_distribution']['positive'],
                'negative': overview['sentiment_distribution']['negative'],
                'neutral': overview['sentiment_distribution']['neutral']
            })
        
        # Sort by sentiment score
        comparison['sentiment_rankings'].sort(key=lambda x: x['overall_score'], reverse=True)
        
        return comparison
    
    def _compare_engagement_patterns(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare engagement patterns across videos."""
        engagement_comparison = []
        
        for analysis in analyses:
            video_title = analysis['video_info'].get('title', 'Unknown')
            engagement = analysis.get('engagement_correlation', {})
            
            if engagement.get('likes_data_available', False):
                engagement_comparison.append({
                    'video': video_title,
                    'sentiment_likes_correlation': engagement.get('sentiment_likes_correlation', 0),
                    'correlation_strength': engagement.get('correlation_strength', 'Unknown')
                })
        
        return {'engagement_patterns': engagement_comparison}
    
    def _compare_topic_sentiments(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare topic sentiments across videos."""
        all_topics = {}
        
        for analysis in analyses:
            video_title = analysis['video_info'].get('title', 'Unknown')
            topic_sentiment = analysis.get('topic_sentiment', {}).get('topic_sentiments', {})
            
            for topic, data in topic_sentiment.items():
                if topic not in all_topics:
                    all_topics[topic] = []
                
                all_topics[topic].append({
                    'video': video_title,
                    'sentiment': data['average_sentiment'],
                    'comment_count': data['comment_count']
                })
        
        # Find topics that appear in multiple videos
        common_topics = {topic: data for topic, data in all_topics.items() if len(data) > 1}
        
        return {
            'common_topics': common_topics,
            'topic_consistency': len(common_topics)
        }
    
    def _generate_audience_insights(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Generate insights about audience behavior across videos."""
        insights = {
            'audience_sentiment_patterns': [],
            'engagement_insights': [],
            'content_reception_patterns': []
        }
        
        # Analyze patterns
        positive_videos = []
        negative_videos = []
        
        for analysis in analyses:
            video_title = analysis['video_info'].get('title', 'Unknown')
            overview = analysis['overview']
            
            if overview['overall_sentiment_score'] > 0.1:
                positive_videos.append(video_title)
            elif overview['overall_sentiment_score'] < -0.1:
                negative_videos.append(video_title)
        
        insights['audience_sentiment_patterns'] = {
            'consistently_positive_videos': positive_videos,
            'consistently_negative_videos': negative_videos,
            'audience_engagement_level': 'High' if len(positive_videos) > len(negative_videos) else 'Mixed'
        }
        
        return insights
    
    def _generate_sentiment_recommendations(self, comments_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on sentiment analysis."""
        recommendations = []
        
        overall_sentiment = comments_df['vader_compound'].mean()
        positive_percentage = len(comments_df[comments_df['vader_compound'] >= 0.05]) / len(comments_df) * 100
        
        # Overall sentiment recommendations
        if overall_sentiment < -0.1:
            recommendations.append("The overall sentiment is negative. Consider addressing common concerns mentioned in comments.")
        elif overall_sentiment > 0.3:
            recommendations.append("Great job! The audience sentiment is very positive. Continue with similar content.")
        
        # Engagement recommendations
        if positive_percentage < 30:
            recommendations.append("Less than 30% of comments are positive. Consider improving content quality or addressing audience feedback.")
        
        # Polarization recommendations
        polarization = self._calculate_polarization(comments_df)
        if polarization == "Highly Polarized":
            recommendations.append("Comments are highly polarized. This could indicate controversial content - consider clarifying your position or addressing concerns.")
        
        # Subjectivity recommendations
        avg_subjectivity = comments_df['textblob_subjectivity'].mean()
        if avg_subjectivity > 0.7:
            recommendations.append("Comments are highly opinion-based. Engage with your audience to build community.")
        
        return recommendations
    
    def _generate_comparison_recommendations(self, analyses: List[Dict]) -> List[str]:
        """Generate recommendations based on video comparison."""
        recommendations = []
        
        # Find best and worst performing videos
        sentiment_scores = [(analysis['video_info'].get('title', 'Unknown'), 
                           analysis['overview']['overall_sentiment_score']) 
                          for analysis in analyses]
        
        best_video = max(sentiment_scores, key=lambda x: x[1])
        worst_video = min(sentiment_scores, key=lambda x: x[1])
        
        recommendations.append(f"'{best_video[0]}' has the most positive audience reception. Analyze what made it successful.")
        
        if worst_video[1] < -0.1:
            recommendations.append(f"'{worst_video[0]}' received negative feedback. Review comments to understand audience concerns.")
        
        # Overall strategy recommendations
        avg_sentiment = np.mean([score for _, score in sentiment_scores])
        if avg_sentiment > 0.2:
            recommendations.append("Overall audience sentiment is positive across videos. Maintain your current content strategy.")
        elif avg_sentiment < -0.1:
            recommendations.append("Consider revising your content strategy based on consistently negative feedback patterns.")
        
        return recommendations
