"""
Data Loader Module for Fake News Detection System

This module provides a comprehensive DataLoader class to load and preprocess
data from multiple fake news datasets including LIAR, FakeNewsNet, and Kaggle.
"""

import os
import csv
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A comprehensive DataLoader class for loading fake news detection datasets.
    
    Supports:
    - LIAR Dataset (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
    - FakeNewsNet Dataset (https://github.com/KaiDMML/FakeNewsNet)
    - Kaggle Fake News Dataset (https://www.kaggle.com/c/fake-news/data)
    
    Attributes:
        data_dir (Path): Root directory for datasets
        cache_dir (Path): Directory for cached processed data
        datasets (Dict): Dictionary storing loaded datasets
    """
    
    # Dataset-specific column mappings
    LIAR_COLUMNS = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_count',
        'false_count', 'half_true_count', 'mostly_true_count',
        'pants_on_fire_count', 'context'
    ]
    
    LABEL_MAPPINGS = {
        'liar': {
            'false': 0,
            'true': 1,
            'barely-true': 2,
            'half-true': 3,
            'mostly-true': 4,
            'pants-fire': 5
        },
        'fakenewsnet': {
            'fake': 0,
            'real': 1
        },
        'kaggle': {
            0: 0,  # Fake
            1: 1   # Real
        }
    }
    
    def __init__(
        self,
        data_dir: str = './data',
        cache_dir: str = './data/cache',
        random_state: int = 42
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Root directory for datasets
            cache_dir (str): Directory for caching processed data
            random_state (int): Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.random_state = random_state
        self.datasets = {}
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")
    
    def load_liar_dataset(
        self,
        split: str = 'all',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load the LIAR dataset.
        
        LIAR Dataset Structure:
        - 12.8K labeled short statements from politifact.com
        - 6-way classification (false, barely-true, half-true, mostly-true, true, pants-fire)
        - Rich metadata about speakers
        
        Args:
            split (str): Dataset split - 'train', 'test', 'valid', or 'all'
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Loaded LIAR dataset
            
        Raises:
            FileNotFoundError: If dataset files are not found
        """
        logger.info(f"Loading LIAR dataset (split: {split})...")
        
        liar_dir = self.data_dir / 'liar'
        
        if not liar_dir.exists():
            logger.error(f"LIAR dataset directory not found at {liar_dir}")
            raise FileNotFoundError(
                f"LIAR dataset not found at {liar_dir}. "
                "Please download from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
            )
        
        # Load data from TSV files
        dataframes = []
        
        if split == 'all':
            splits = ['train', 'test', 'valid']
        else:
            splits = [split]
        
        for sp in splits:
            file_path = liar_dir / f'{sp}.tsv'
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(
                    file_path,
                    sep='\t',
                    header=None,
                    names=self.LIAR_COLUMNS,
                    quoting=csv.QUOTE_NONE
                )
                df['split'] = sp
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} samples from {sp} split")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise
        
        if not dataframes:
            raise FileNotFoundError(f"No LIAR dataset files found for split '{split}'")
        
        df = pd.concat(dataframes, ignore_index=True)
        
        # Normalize labels
        df['label_text'] = df['label']
        df['label'] = df['label'].map(self.LABEL_MAPPINGS['liar'])
        
        # Clean and preprocess text
        df['statement'] = df['statement'].apply(self._clean_text)
        
        logger.info(f"LIAR dataset loaded successfully with {len(df)} samples")
        self.datasets['liar'] = df
        
        return df
    
    def load_fakenewsnet_dataset(
        self,
        news_source: str = 'all',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load the FakeNewsNet dataset.
        
        FakeNewsNet provides:
        - News content from Politifact and GossipCop
        - Social context features (shares, comments, user information)
        - Binary classification (fake/real)
        
        Args:
            news_source (str): 'politifact', 'gossipcop', or 'all'
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Loaded FakeNewsNet dataset
            
        Raises:
            FileNotFoundError: If dataset files are not found
        """
        logger.info(f"Loading FakeNewsNet dataset (source: {news_source})...")
        
        fakenewsnet_dir = self.data_dir / 'fakenewsnet'
        
        if not fakenewsnet_dir.exists():
            logger.error(f"FakeNewsNet dataset directory not found at {fakenewsnet_dir}")
            raise FileNotFoundError(
                f"FakeNewsNet dataset not found at {fakenewsnet_dir}. "
                "Please download from https://github.com/KaiDMML/FakeNewsNet"
            )
        
        dataframes = []
        sources = ['politifact', 'gossipcop'] if news_source == 'all' else [news_source]
        
        for source in sources:
            source_dir = fakenewsnet_dir / source
            
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue
            
            for label in ['fake', 'real']:
                label_dir = source_dir / label
                
                if not label_dir.exists():
                    continue
                
                for news_file in label_dir.glob('*.json'):
                    try:
                        with open(news_file, 'r', encoding='utf-8') as f:
                            news_data = json.load(f)
                        
                        # Extract relevant information
                        record = {
                            'id': news_data.get('id', news_file.stem),
                            'title': news_data.get('title', ''),
                            'text': news_data.get('text', ''),
                            'label': label,
                            'source': source,
                            'url': news_data.get('url', ''),
                            'date': news_data.get('date', ''),
                        }
                        
                        # Add social context if available
                        if 'social' in news_data:
                            record['engagement'] = news_data['social'].get('engagements', 0)
                            record['shares'] = news_data['social'].get('shares', 0)
                            record['comments'] = news_data['social'].get('comments', 0)
                        
                        dataframes.append(record)
                        
                    except Exception as e:
                        logger.warning(f"Error reading {news_file}: {e}")
                        continue
        
        if not dataframes:
            raise FileNotFoundError(
                f"No FakeNewsNet data found for source '{news_source}'"
            )
        
        df = pd.DataFrame(dataframes)
        
        # Fill missing engagement columns
        for col in ['engagement', 'shares', 'comments']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Normalize labels
        df['label_text'] = df['label']
        df['label'] = df['label'].map(self.LABEL_MAPPINGS['fakenewsnet'])
        
        # Combine title and text
        df['text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
        df['text'] = df['text'].apply(self._clean_text)
        
        logger.info(f"FakeNewsNet dataset loaded with {len(df)} samples")
        self.datasets['fakenewsnet'] = df
        
        return df
    
    def load_kaggle_dataset(
        self,
        csv_path: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load the Kaggle Fake News dataset.
        
        Kaggle dataset provides:
        - ~20K news articles
        - Binary classification (fake/real)
        - Title, text, author, and date information
        
        Args:
            csv_path (str, optional): Path to CSV file. If None, looks in data_dir
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Loaded Kaggle dataset
            
        Raises:
            FileNotFoundError: If dataset file is not found
        """
        logger.info("Loading Kaggle Fake News dataset...")
        
        if csv_path is None:
            csv_path = self.data_dir / 'kaggle' / 'news.csv'
        else:
            csv_path = Path(csv_path)
        
        if not csv_path.exists():
            logger.error(f"Kaggle dataset file not found at {csv_path}")
            raise FileNotFoundError(
                f"Kaggle dataset not found at {csv_path}. "
                "Please download from https://www.kaggle.com/c/fake-news/data"
            )
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} samples from Kaggle dataset")
            
            # Expected columns: id, title, author, text, label
            required_cols = ['title', 'text', 'label']
            
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing expected columns. Found: {df.columns.tolist()}")
            
            # Normalize labels if they are binary (0/1)
            if 'label' in df.columns:
                df['label_text'] = df['label'].map({0: 'fake', 1: 'real'})
                df['label'] = df['label'].astype(int)
            
            # Combine title and text
            if 'title' in df.columns and 'text' in df.columns:
                df['full_text'] = (
                    df['title'].fillna('') + ' ' + df['text'].fillna('')
                ).str.strip()
                df['full_text'] = df['full_text'].apply(self._clean_text)
            
            logger.info(f"Kaggle dataset loaded with {len(df)} samples")
            self.datasets['kaggle'] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
            raise
    
    def load_combined_dataset(
        self,
        datasets: List[str] = None,
        balance: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Load and combine multiple datasets.
        
        Args:
            datasets (List[str]): List of dataset names to load ('liar', 'fakenewsnet', 'kaggle')
            balance (bool): Whether to balance classes
            test_size (float): Test set proportion
            val_size (float): Validation set proportion
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with 'train', 'test', 'val' splits
        """
        if datasets is None:
            datasets = ['liar', 'fakenewsnet', 'kaggle']
        
        logger.info(f"Loading combined dataset from: {datasets}")
        
        dfs = []
        
        if 'liar' in datasets and 'liar' not in self.datasets:
            try:
                dfs.append(self.load_liar_dataset())
            except FileNotFoundError as e:
                logger.warning(f"Could not load LIAR: {e}")
        elif 'liar' in self.datasets:
            dfs.append(self.datasets['liar'])
        
        if 'fakenewsnet' in datasets and 'fakenewsnet' not in self.datasets:
            try:
                dfs.append(self.load_fakenewsnet_dataset())
            except FileNotFoundError as e:
                logger.warning(f"Could not load FakeNewsNet: {e}")
        elif 'fakenewsnet' in self.datasets:
            dfs.append(self.datasets['fakenewsnet'])
        
        if 'kaggle' in datasets and 'kaggle' not in self.datasets:
            try:
                dfs.append(self.load_kaggle_dataset())
            except FileNotFoundError as e:
                logger.warning(f"Could not load Kaggle: {e}")
        elif 'kaggle' in self.datasets:
            dfs.append(self.datasets['kaggle'])
        
        if not dfs:
            raise ValueError("No datasets could be loaded")
        
        # Combine datasets
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        logger.info(f"Combined dataset size: {len(combined_df)}")
        
        # Balance classes if requested
        if balance:
            combined_df = self._balance_classes(combined_df)
            logger.info(f"Balanced dataset size: {len(combined_df)}")
        
        # Split into train, validation, and test sets
        splits = self._create_splits(combined_df, test_size, val_size)
        
        return splits
    
    def _create_splits(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train, validation, and test splits.
        
        Args:
            df (pd.DataFrame): Input dataframe
            test_size (float): Test set proportion
            val_size (float): Validation set proportion (of training set)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with splits
        """
        # First split: test set
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df['label'] if 'label' in df.columns else None
        )
        
        # Second split: validation set
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=train_df['label'] if 'label' in train_df.columns else None
        )
        
        logger.info(
            f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        
        return {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }
    
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance classes using undersampling of majority class.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Balanced dataframe
        """
        if 'label' not in df.columns:
            logger.warning("'label' column not found. Cannot balance classes.")
            return df
        
        min_count = df['label'].value_counts().min()
        balanced_df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min_count, random_state=self.random_state)
        )
        
        logger.info(f"Classes balanced to {min_count} samples each")
        return balanced_df.reset_index(drop=True)
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ''
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def get_label_distribution(self, dataset_name: str = None) -> Dict:
        """
        Get label distribution for a dataset.
        
        Args:
            dataset_name (str, optional): Dataset to analyze. If None, shows all loaded datasets.
            
        Returns:
            Dict: Label distribution statistics
        """
        stats = {}
        
        datasets_to_check = (
            {dataset_name: self.datasets[dataset_name]}
            if dataset_name
            else self.datasets
        )
        
        for name, df in datasets_to_check.items():
            if 'label' in df.columns:
                counts = df['label'].value_counts().to_dict()
                distribution = (df['label'].value_counts(normalize=True) * 100).to_dict()
                
                stats[name] = {
                    'total_samples': len(df),
                    'label_counts': counts,
                    'label_distribution': distribution
                }
                
                logger.info(f"{name} - {stats[name]}")
        
        return stats
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict:
        """
        Get detailed information about a dataset.
        
        Args:
            dataset_name (str, optional): Dataset to analyze
            
        Returns:
            Dict: Dataset information
        """
        info = {}
        
        datasets_to_check = (
            {dataset_name: self.datasets[dataset_name]}
            if dataset_name
            else self.datasets
        )
        
        for name, df in datasets_to_check.items():
            info[name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample': df.head(1).to_dict('records')
            }
        
        return info
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = 'csv'
    ) -> Path:
        """
        Save dataset to disk.
        
        Args:
            df (pd.DataFrame): Dataset to save
            filename (str): Output filename
            format (str): Output format ('csv', 'parquet', 'json')
            
        Returns:
            Path: Path to saved file
        """
        output_path = self.cache_dir / f"{filename}.{format}"
        
        try:
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Dataset saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def load_dataset(
        self,
        filename: str,
        format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Load previously saved dataset.
        
        Args:
            filename (str): Filename (without extension)
            format (str): File format
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        input_path = self.cache_dir / f"{filename}.{format}"
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        try:
            if format == 'csv':
                df = pd.read_csv(input_path)
            elif format == 'parquet':
                df = pd.read_parquet(input_path)
            elif format == 'json':
                df = pd.read_json(input_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Dataset loaded from {input_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize DataLoader
    loader = DataLoader(data_dir='./data')
    
    # Example 1: Load LIAR dataset
    try:
        liar_data = loader.load_liar_dataset(split='all')
        print(f"LIAR Dataset shape: {liar_data.shape}")
        print(f"Columns: {liar_data.columns.tolist()}")
    except FileNotFoundError as e:
        print(f"LIAR dataset not available: {e}")
    
    # Example 2: Load FakeNewsNet dataset
    try:
        fn_data = loader.load_fakenewsnet_dataset(news_source='all')
        print(f"FakeNewsNet Dataset shape: {fn_data.shape}")
    except FileNotFoundError as e:
        print(f"FakeNewsNet dataset not available: {e}")
    
    # Example 3: Load Kaggle dataset
    try:
        kaggle_data = loader.load_kaggle_dataset()
        print(f"Kaggle Dataset shape: {kaggle_data.shape}")
    except FileNotFoundError as e:
        print(f"Kaggle dataset not available: {e}")
    
    # Example 4: Get dataset information
    dataset_info = loader.get_dataset_info()
    print(f"Loaded datasets info: {dataset_info}")
    
    # Example 5: Get label distribution
    label_dist = loader.get_label_distribution()
    print(f"Label distributions: {label_dist}")
