"""
Advanced Text Preprocessing Module for Fake News Detection

This module provides comprehensive text preprocessing capabilities including:
- URL processing and removal
- Emoji handling and conversion
- Contraction expansion
- Clickbait detection
- Sarcasm detection
- Tokenization and normalization
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Comprehensive text preprocessing for fake news detection.
    
    Attributes:
        lemmatizer: WordNet lemmatizer for word normalization
        stemmer: Porter stemmer for word stemming
        stopwords_set: Set of English stopwords
    """
    
    def __init__(self, remove_stopwords: bool = False, use_lemmatization: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords (default: False)
            use_lemmatization: Whether to use lemmatization (default: True)
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stopwords_set = set(stopwords.words('english'))
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        
        # Emoji dictionary for conversion
        self.emoji_dict = {
            '😀': 'happy', '😁': 'grinning', '😂': 'laughing', '😃': 'smiling_face_with_open_mouth',
            '😄': 'smiling_face_with_open_mouth_and_cold_sweat', '😅': 'smiling_face_with_open_mouth_and_cold_sweat',
            '😆': 'laughing', '😉': 'winking_face', '😊': 'smiling_face_with_smiling_eyes',
            '😎': 'cool', '😍': 'heart_eyes', '😘': 'face_blowing_a_kiss', '😗': 'kissing_face',
            '😚': 'kissing_face_with_closed_eyes', '😙': 'kissing_face_with_smiling_eyes',
            '🙂': 'slightly_smiling_face', '🤗': 'hugging_face', '🤩': 'star_struck',
            '😐': 'neutral_face', '😑': 'expressionless_face', '😶': 'face_with_mouth_covered',
            '😏': 'smirking_face', '😣': 'persevering_face', '😥': 'disappointed_but_relieved_face',
            '😌': 'relieved_face', '😔': 'pensive_face', '😓': 'downcast_face_with_sweat',
            '😪': 'sleepy_face', '😒': 'unamused_face', '😬': 'grimacing_face',
            '🤔': 'thinking_face', '😌': 'relieved_face', '😜': 'face_with_tongue',
            '😔': 'pensive_face', '😡': 'pouting_face', '😠': 'angry_face',
            '🤬': 'face_with_symbols_on_mouth', '😈': 'smiling_face_with_horns',
            '😭': 'loudly_crying_face', '💔': 'broken_heart', '💪': 'flexed_biceps',
            '🔥': 'fire', '👍': 'thumbs_up', '👎': 'thumbs_down', '💯': 'hundred_points'
        }
        
        # Contractions dictionary
        self.contractions_dict = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "could've": "could have",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who'd": "who would",
            "who'll": "who will",
            "who're": "who are",
            "who's": "who is",
            "who've": "who have",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }

    def process_urls(self, text: str) -> Tuple[str, List[str]]:
        """
        Process and extract URLs from text.
        
        Args:
            text: Input text containing URLs
            
        Returns:
            Tuple of (text with URLs replaced, list of extracted URLs)
        """
        url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        urls = re.findall(url_pattern, text)
        text_without_urls = re.sub(url_pattern, 'URL', text)
        return text_without_urls, urls

    def handle_emojis(self, text: str) -> str:
        """
        Convert emojis to their text representations.
        
        Args:
            text: Input text containing emojis
            
        Returns:
            Text with emojis converted to text descriptions
        """
        for emoji, description in self.emoji_dict.items():
            text = text.replace(emoji, f' {description} ')
        
        # Handle other emojis by removing them if not in dictionary
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        return text

    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text with contractions
            
        Returns:
            Text with contractions expanded
        """
        pattern = re.compile(r'\b(' + '|'.join(self.contractions_dict.keys()) + r')\b')
        return pattern.sub(lambda x: self.contractions_dict[x.group()], text.lower())

    def detect_clickbait(self, text: str) -> Dict[str, any]:
        """
        Detect clickbait characteristics in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with clickbait features and score
        """
        clickbait_indicators = {
            'excessive_caps': False,
            'excessive_punctuation': False,
            'numbered_lists': False,
            'sensational_words': False,
            'question_marks': False,
            'exclamation_marks': False
        }
        
        # Check for excessive capitalization
        capital_letters = sum(1 for c in text if c.isupper())
        if capital_letters > len(text) * 0.4:  # More than 40% capitals
            clickbait_indicators['excessive_caps'] = True
        
        # Check for excessive punctuation
        punctuation_count = sum(1 for c in text if c in '!?.')
        if punctuation_count > len(text.split()) * 0.3:  # More than 30% punctuation
            clickbait_indicators['excessive_punctuation'] = True
        
        # Check for numbered lists
        if re.search(r'(\d+\s*[\)\.\-:]\s*\w+)', text):
            clickbait_indicators['numbered_lists'] = True
        
        # Check for sensational words
        sensational_words = ['shocking', 'surprising', 'unbelievable', 'incredible',
                            'amazing', 'revealed', 'exposed', 'secret', 'banned',
                            'controversial', 'hate', 'jealous', 'destroyed']
        if any(word in text.lower() for word in sensational_words):
            clickbait_indicators['sensational_words'] = True
        
        # Count question marks and exclamation marks
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        if question_marks > 2:
            clickbait_indicators['question_marks'] = True
        if exclamation_marks > 2:
            clickbait_indicators['exclamation_marks'] = True
        
        # Calculate clickbait score
        clickbait_score = sum(clickbait_indicators.values()) / len(clickbait_indicators)
        
        return {
            'indicators': clickbait_indicators,
            'score': clickbait_score,
            'is_clickbait': clickbait_score > 0.4
        }

    def detect_sarcasm(self, text: str) -> Dict[str, any]:
        """
        Detect sarcasm patterns in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sarcasm features and confidence score
        """
        sarcasm_patterns = {
            'opposite_emotions': False,
            'excessive_punctuation': False,
            'quoted_phrases': False,
            'intensifiers': False
        }
        
        text_lower = text.lower()
        
        # Check for opposite emotions (positive words followed by negative context)
        positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent']
        negative_words = ['bad', 'terrible', 'horrible', 'awful', 'poor', 'worst']
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_words)
        
        if has_positive and has_negative:
            sarcasm_patterns['opposite_emotions'] = True
        
        # Check for excessive punctuation (ellipsis, multiple marks)
        if re.search(r'\.{2,}|[!?]{2,}', text):
            sarcasm_patterns['excessive_punctuation'] = True
        
        # Check for quoted phrases
        if re.search(r'["\'].*?["\']', text):
            sarcasm_patterns['quoted_phrases'] = True
        
        # Check for intensifiers (really, very, so)
        intensifiers = ['really', 'very', 'so', 'absolutely', 'totally', 'completely']
        if any(word in text_lower for word in intensifiers):
            sarcasm_patterns['intensifiers'] = True
        
        # Calculate sarcasm confidence
        sarcasm_score = sum(sarcasm_patterns.values()) / len(sarcasm_patterns)
        
        return {
            'patterns': sarcasm_patterns,
            'confidence': sarcasm_score,
            'likely_sarcastic': sarcasm_score > 0.5
        }

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-ASCII characters (optional, can be controlled by parameter)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()

    def tokenize(self, text: str, sentence_level: bool = False) -> List[str]:
        """
        Tokenize text into words or sentences.
        
        Args:
            text: Input text to tokenize
            sentence_level: If True, return sentences; if False, return words
            
        Returns:
            List of tokens (words or sentences)
        """
        if sentence_level:
            return sent_tokenize(text)
        else:
            return word_tokenize(text)

    def normalize_text(self, text: str) -> str:
        """
        Normalize text using lemmatization or stemming.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        tokens = self.tokenize(text)
        
        if self.use_lemmatization:
            normalized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            normalized_tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            normalized_tokens = [token for token in normalized_tokens 
                               if token.lower() not in self.stopwords_set]
        
        return ' '.join(normalized_tokens)

    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def process(self, text: str, 
               remove_urls: bool = True,
               handle_emojis_flag: bool = True,
               expand_contractions_flag: bool = True,
               remove_punctuation_flag: bool = False,
               normalize: bool = True,
               lowercase: bool = True) -> Dict[str, any]:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            text: Input text to process
            remove_urls: Whether to remove URLs
            handle_emojis_flag: Whether to handle emojis
            expand_contractions_flag: Whether to expand contractions
            remove_punctuation_flag: Whether to remove punctuation
            normalize: Whether to normalize text
            lowercase: Whether to convert to lowercase
            
        Returns:
            Dictionary containing processed text and analysis results
        """
        # Store original text
        original_text = text
        
        # Step 1: Process URLs
        if remove_urls:
            text, urls = self.process_urls(text)
        else:
            urls = []
        
        # Step 2: Handle emojis
        if handle_emojis_flag:
            text = self.handle_emojis(text)
        
        # Step 3: Expand contractions
        if expand_contractions_flag:
            text = self.expand_contractions(text)
        else:
            text = text.lower() if lowercase else text
        
        # Step 4: Clean text
        text = self.clean_text(text)
        
        # Step 5: Remove punctuation if requested
        if remove_punctuation_flag:
            text = self.remove_punctuation(text)
        
        # Step 6: Normalize
        if normalize:
            text = self.normalize_text(text)
        
        # Step 7: Detect clickbait
        clickbait_analysis = self.detect_clickbait(original_text)
        
        # Step 8: Detect sarcasm
        sarcasm_analysis = self.detect_sarcasm(original_text)
        
        return {
            'original_text': original_text,
            'processed_text': text,
            'urls_found': urls,
            'clickbait_analysis': clickbait_analysis,
            'sarcasm_analysis': sarcasm_analysis,
            'text_length': {
                'original': len(original_text),
                'processed': len(text)
            }
        }

    def batch_process(self, texts: List[str], **kwargs) -> List[Dict[str, any]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
            **kwargs: Additional arguments to pass to process method
            
        Returns:
            List of processing results
        """
        results = []
        for idx, text in enumerate(texts):
            try:
                result = self.process(text, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {idx}: {str(e)}")
                results.append({'error': str(e), 'index': idx})
        
        return results


# Utility functions
def create_preprocessor(remove_stopwords: bool = False, 
                       use_lemmatization: bool = True) -> TextPreprocessor:
    """
    Factory function to create a TextPreprocessor instance.
    
    Args:
        remove_stopwords: Whether to remove stopwords
        use_lemmatization: Whether to use lemmatization
        
    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        remove_stopwords=remove_stopwords,
        use_lemmatization=use_lemmatization
    )


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = """
    🔥 SHOCKING NEWS!!! 👀 This politician's SECRET REVEALED... 
    You won't BELIEVE what they said! Visit https://example.com/news for more details.
    Can't believe it? Check it out! #incredible #amazing
    """
    
    result = preprocessor.process(sample_text)
    
    print("Original Text:", result['original_text'])
    print("\nProcessed Text:", result['processed_text'])
    print("\nURLs Found:", result['urls_found'])
    print("\nClickbait Analysis:", result['clickbait_analysis'])
    print("\nSarcasm Analysis:", result['sarcasm_analysis'])
    print("\nText Length:", result['text_length'])
