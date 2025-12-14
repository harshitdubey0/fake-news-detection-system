"""
Feature Engineering Module for Fake News Detection

This module provides comprehensive feature extraction methods including:
- TF-IDF vectorization
- Linguistic features
- Sentiment analysis
- Readability metrics
"""

import re
import string
import warnings
from typing import Tuple, Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

warnings.filterwarnings('ignore')

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
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TFIDFFeatures:
    """TF-IDF Vectorization for text data"""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2),
                 max_df: float = 0.95, min_df: int = 2, lowercase: bool = True):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (unigrams and bigrams by default)
            max_df: Maximum document frequency
            min_df: Minimum document frequency
            lowercase: Whether to convert to lowercase
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            lowercase=lowercase,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'TFIDFFeatures':
        """Fit the TF-IDF vectorizer"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.vectorizer.get_feature_names_out().tolist()


class LinguisticFeatures:
    """Extract linguistic features from text"""
    
    @staticmethod
    def extract_features(text: str) -> Dict[str, float]:
        """
        Extract comprehensive linguistic features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with linguistic features
        """
        features = {}
        
        # Clean text
        clean_text = LinguisticFeatures._clean_text(text)
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(clean_text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = LinguisticFeatures._avg_word_length(clean_text)
        
        # Punctuation features
        features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
        features['punctuation_ratio'] = features['punctuation_count'] / max(len(text), 1)
        
        # Capital letters
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = features['uppercase_count'] / max(len(text), 1)
        
        # Digit features
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        features['digit_ratio'] = features['digit_count'] / max(len(text), 1)
        
        # Special characters
        features['special_char_count'] = sum(1 for c in text if not c.isalnum() and c not in string.whitespace)
        
        # POS tagging
        tokens = word_tokenize(clean_text.lower())
        pos_tags = pos_tag(tokens)
        pos_counts = LinguisticFeatures._count_pos_tags(pos_tags)
        features.update(pos_counts)
        
        # URL and mention patterns
        features['url_count'] = len(re.findall(r'http[s]?://\S+', text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        
        # Quote features
        features['quote_count'] = text.count('"')
        features['single_quote_count'] = text.count("'")
        
        # Exclamation and question marks
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for processing"""
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def _avg_word_length(text: str) -> float:
        """Calculate average word length"""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    @staticmethod
    def _count_pos_tags(pos_tags: List[Tuple[str, str]]) -> Dict[str, float]:
        """Count POS tags"""
        pos_counts = {}
        pos_types = ['NN', 'VB', 'JJ', 'RB', 'PRP', 'IN', 'DT', 'CC']
        total_tokens = len(pos_tags)
        
        for pos in pos_types:
            count = sum(1 for _, tag in pos_tags if tag.startswith(pos))
            pos_counts[f'pos_{pos.lower()}'] = count / max(total_tokens, 1)
        
        return pos_counts


class SentimentFeatures:
    """Extract sentiment analysis features"""
    
    @staticmethod
    def extract_features(text: str) -> Dict[str, float]:
        """
        Extract sentiment features using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment features
        """
        features = {}
        
        try:
            blob = TextBlob(text)
            features['polarity'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            features['polarity'] = 0.0
            features['subjectivity'] = 0.0
        
        # Emotion-based features
        features.update(SentimentFeatures._extract_emotional_intensity(text))
        
        return features
    
    @staticmethod
    def _extract_emotional_intensity(text: str) -> Dict[str, float]:
        """
        Extract emotional intensity indicators
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotional intensity features
        """
        features = {}
        
        # Positive and negative word patterns
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'best', 'love'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'evil', 'stupid'}
        
        words = text.lower().split()
        features['positive_word_ratio'] = sum(1 for w in words if w in positive_words) / max(len(words), 1)
        features['negative_word_ratio'] = sum(1 for w in words if w in negative_words) / max(len(words), 1)
        
        # Emotion intensifiers (ALL CAPS, multiple punctuation)
        features['caps_intensity'] = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
        
        return features


class ReadabilityMetrics:
    """Extract readability metrics from text"""
    
    @staticmethod
    def extract_features(text: str) -> Dict[str, float]:
        """
        Extract readability metrics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with readability features
        """
        features = {}
        
        # Flesch Reading Ease score
        features['flesch_kincaid_grade'] = ReadabilityMetrics._flesch_kincaid_grade(text)
        
        # Flesch Reading Ease
        features['flesch_reading_ease'] = ReadabilityMetrics._flesch_reading_ease(text)
        
        # Gunning Fog Index
        features['gunning_fog'] = ReadabilityMetrics._gunning_fog_index(text)
        
        # Dale-Chall Score
        features['dale_chall_score'] = ReadabilityMetrics._dale_chall_score(text)
        
        # Average sentence length
        sentences = sent_tokenize(text)
        words = text.split()
        features['avg_sentence_length'] = len(words) / max(len(sentences), 1)
        
        return features
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower()
        syllable_count = 0
        vowels = "aeiouy"
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    @staticmethod
    def _flesch_kincaid_grade(text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        sentences = sent_tokenize(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        syllables = sum(ReadabilityMetrics._count_syllables(word) for word in words)
        
        grade = (0.39 * len(words) / max(len(sentences), 1) + 
                 11.8 * syllables / max(len(words), 1) - 15.59)
        
        return max(0, grade)
    
    @staticmethod
    def _flesch_reading_ease(text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = sent_tokenize(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        syllables = sum(ReadabilityMetrics._count_syllables(word) for word in words)
        
        score = (206.835 - 1.015 * len(words) / max(len(sentences), 1) - 
                 84.6 * syllables / max(len(words), 1))
        
        return max(0, min(100, score))
    
    @staticmethod
    def _gunning_fog_index(text: str) -> float:
        """Calculate Gunning Fog Index"""
        sentences = sent_tokenize(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Count complex words (3+ syllables)
        complex_words = sum(1 for word in words 
                           if ReadabilityMetrics._count_syllables(word) >= 3)
        
        fog_index = (0.4 * (len(words) / max(len(sentences), 1) + 
                            100 * complex_words / max(len(words), 1)))
        
        return max(0, fog_index)
    
    @staticmethod
    def _dale_chall_score(text: str) -> float:
        """Calculate Dale-Chall Readability Score"""
        sentences = sent_tokenize(text)
        words = [w.lower() for w in text.split() if w.lower().isalpha()]
        
        if not sentences or not words:
            return 0.0
        
        # Simplified word difficulty (using common words list)
        difficult_words = sum(1 for word in words 
                             if len(word) > 6 and not word in ['through', 'between', 'without'])
        
        score = (0.1579 * difficult_words / max(len(words), 1) * 100 + 
                 0.0496 * len(words) / max(len(sentences), 1))
        
        return max(0, score)


class FeatureEngineering:
    """Main feature engineering class that combines all feature extractors"""
    
    def __init__(self, max_tfidf_features: int = 5000):
        """
        Initialize feature engineering
        
        Args:
            max_tfidf_features: Maximum TF-IDF features
        """
        self.tfidf = TFIDFFeatures(max_features=max_tfidf_features)
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'FeatureEngineering':
        """Fit feature extractors"""
        self.tfidf.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> pd.DataFrame:
        """
        Transform texts to feature dataframe
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame with all extracted features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineering must be fitted before transforming")
        
        features_list = []
        
        for text in texts:
            row_features = {}
            
            # Extract TF-IDF features
            tfidf_vector = self.tfidf.transform([text])[0]
            tfidf_feature_names = self.tfidf.get_feature_names()
            for i, name in enumerate(tfidf_feature_names):
                row_features[f'tfidf_{name}'] = tfidf_vector[i]
            
            # Extract linguistic features
            linguistic_feat = LinguisticFeatures.extract_features(text)
            row_features.update(linguistic_feat)
            
            # Extract sentiment features
            sentiment_feat = SentimentFeatures.extract_features(text)
            row_features.update(sentiment_feat)
            
            # Extract readability features
            readability_feat = ReadabilityMetrics.extract_features(text)
            row_features.update(readability_feat)
            
            features_list.append(row_features)
        
        return pd.DataFrame(features_list)
    
    def fit_transform(self, texts: List[str]) -> pd.DataFrame:
        """Fit and transform texts"""
        self.fit(texts)
        return self.transform(texts)


def extract_all_features(texts: Union[List[str], pd.Series], 
                        fit: bool = True) -> Tuple[pd.DataFrame, FeatureEngineering]:
    """
    Extract all features from texts
    
    Args:
        texts: List or Series of texts
        fit: Whether to fit the feature engineering pipeline
        
    Returns:
        Tuple of (features DataFrame, fitted FeatureEngineering object)
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    fe = FeatureEngineering()
    
    if fit:
        features_df = fe.fit_transform(texts)
    else:
        features_df = fe.transform(texts)
    
    return features_df, fe


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a great news article about recent developments in technology.",
        "BREAKING NEWS!!! You won't believe what happened!!! Click here NOW!!!",
        "The scientist conducted a rigorous study with peer review and published findings."
    ]
    
    # Extract all features
    features_df, fe = extract_all_features(sample_texts, fit=True)
    
    print("Features extracted successfully!")
    print(f"Shape: {features_df.shape}")
    print(f"Columns: {features_df.columns.tolist()[:10]}...")  # Print first 10 columns
