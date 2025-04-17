import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
import logging
from typing import Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open('src/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded configuration")
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {str(e)}")
        raise

def load_data() -> pd.DataFrame:
    """
    Load the dataset from CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If the data file is not found
        ValueError: If the loaded data is empty
    """
    try:
        df = pd.read_csv('data/data.csv')
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error("Data file not found. Please ensure 'data/data.csv' exists.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_processed = df.copy()
    
    # Define columns
    categorical_cols = ['CCA', 'learning_style', 'gender', 'direct_admission', 
                       'mode_of_transport', 'bag_color']
    numerical_cols = ['number_of_siblings', 'n_male', 'n_female', 'age', 
                     'hours_per_week', 'attendance_rate', 'final_test']
    
    # Handle missing values
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            logger.info(f"Filled missing values in {col} with mode")
    
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            logger.info(f"Filled missing values in {col} with median")
    
    return df_processed

def process_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process time-related columns in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with processed time columns
    """
    df_processed = df.copy()
    time_cols = ['sleep_time', 'wake_time']
    
    for col in time_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_datetime(df_processed[col], format='%H:%M').dt.hour
            logger.info(f"Processed time column: {col}")
    
    # Calculate sleep duration
    df_processed['sleep_duration'] = (df_processed['wake_time'] - df_processed['sleep_time']) % 24
    logger.info("Calculated sleep duration")
    
    return df_processed

def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using label encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
    """
    df_processed = df.copy()
    categorical_cols = ['CCA', 'learning_style', 'gender', 'direct_admission', 
                       'mode_of_transport', 'bag_color']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            logger.info(f"Encoded categorical variable: {col}")
    
    return df_processed

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by handling missing values, processing time columns,
    and encoding categorical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    try:
        # Drop unnecessary columns
        df_processed = df.drop(['index', 'student_id', 'tuition'], axis=1)
        logger.info("Dropped unnecessary columns")
        
        # Handle missing values
        df_processed = handle_missing_values(df_processed)
        
        # Process time columns
        df_processed = process_time_columns(df_processed)
        
        # Encode categorical variables
        df_processed = encode_categorical_variables(df_processed)
        
        logger.info("Data preprocessing completed successfully")
        return df_processed
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def split_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    try:
        # Validate configuration parameters
        if not 0 < config['data']['test_size'] < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        # Split the data
        X = df.drop('final_test', axis=1)
        y = df['final_test']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        logger.info(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in data splitting: {str(e)}")
        raise 