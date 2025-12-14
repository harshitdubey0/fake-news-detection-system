"""
Configuration settings for the Fake News Detection System project.

This module contains all configuration parameters including paths for data,
models, and hyperparameters used across the project.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"
VALIDATION_DATA_DIR = PROCESSED_DATA_DIR / "validation"

# Model subdirectories
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Dataset parameters
DATASET_NAME = "fake_news"
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
STRATIFY_COLUMN = "label"

# Data file names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
VALIDATION_FILE = "validation.csv"
RAW_DATA_FILE = "news_data.csv"

# Text preprocessing parameters
MAX_SEQUENCE_LENGTH = 512
MIN_WORD_FREQUENCY = 2
REMOVE_STOPWORDS = True
LOWERCASE = True
REMOVE_SPECIAL_CHARS = True
REMOVE_URLS = True

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model type selection
MODEL_TYPE = "bert"  # Options: "bert", "lstm", "cnn", "ensemble"
PRETRAINED_MODEL_NAME = "bert-base-uncased"

# LSTM Configuration
LSTM_EMBEDDING_DIM = 128
LSTM_UNITS = 256
LSTM_DROPOUT = 0.3
LSTM_RECURRENT_DROPOUT = 0.2

# CNN Configuration
CNN_FILTERS = [100, 100, 100]
CNN_KERNEL_SIZES = [3, 4, 5]
CNN_ACTIVATION = "relu"

# BERT Configuration
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 32
BERT_NUM_LABELS = 2
BERT_POOLING = "mean"  # Options: "mean", "max", "first"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1

# Optimizer parameters
OPTIMIZER = "adamw"  # Options: "adam", "adamw", "sgd", "rmsprop"
MOMENTUM = 0.9
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Learning rate scheduler
SCHEDULER = "linear"  # Options: "linear", "cosine", "step", "exponential"
SCHEDULER_STEP_SIZE = 2
SCHEDULER_GAMMA = 0.1

# Early stopping
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_METRIC = "val_loss"
EARLY_STOPPING_MODE = "min"  # Options: "min", "max"

# =============================================================================
# VALIDATION & EVALUATION CONFIGURATION
# =============================================================================

# Validation parameters
VALIDATION_SPLIT = 0.1
VALIDATION_FREQUENCY = 1  # Validate every N epochs

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "roc_auc"]
THRESHOLD = 0.5  # Classification threshold for binary classification

# =============================================================================
# REGULARIZATION
# =============================================================================

L1_REGULARIZATION = 0.0
L2_REGULARIZATION = 1e-5
DROPOUT_RATE = 0.3
BATCH_NORMALIZATION = True

# =============================================================================
# LOGGING & CHECKPOINTING
# =============================================================================

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_FORMAT = "fake_news_detection_{timestamp}.log"

# Model checkpointing
SAVE_CHECKPOINT = True
CHECKPOINT_FREQUENCY = 1  # Save every N epochs
KEEP_BEST_MODEL = True
BEST_MODEL_METRIC = "val_f1"
BEST_MODEL_MODE = "max"

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Inference parameters
INFERENCE_BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.7
RETURN_PROBABILITIES = True

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

# Hardware configuration
DEVICE = "cuda"  # Options: "cuda", "cpu", "mps"
MIXED_PRECISION = False  # Enable mixed precision training
NUM_WORKERS = 4
PIN_MEMORY = True

# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

# Experiment tracking
TRACK_EXPERIMENTS = True
EXPERIMENT_NAME = "fake_news_detection"
WANDB_ENABLED = False
MLFLOW_ENABLED = False

# =============================================================================
# DIRECTORIES CREATION
# =============================================================================

def create_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TRAIN_DATA_DIR,
        TEST_DATA_DIR,
        VALIDATION_DATA_DIR,
        MODELS_DIR,
        TRAINED_MODELS_DIR,
        PRETRAINED_MODELS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SUMMARY
# =============================================================================

def print_config():
    """Print current configuration settings."""
    print("=" * 80)
    print("FAKE NEWS DETECTION SYSTEM - CONFIGURATION SUMMARY")
    print("=" * 80)
    print("\nPROJECT PATHS:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Models Directory: {MODELS_DIR}")
    print(f"  Logs Directory: {LOGS_DIR}")
    print("\nMODEL CONFIGURATION:")
    print(f"  Model Type: {MODEL_TYPE}")
    print(f"  Pretrained Model: {PRETRAINED_MODEL_NAME}")
    print("\nTRAINING CONFIGURATION:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Optimizer: {OPTIMIZER}")
    print("\nDEVICE CONFIGURATION:")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed Precision: {MIXED_PRECISION}")
    print("=" * 80)


if __name__ == "__main__":
    create_directories()
    print_config()
