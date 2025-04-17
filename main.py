import logging
import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.data_preparation import (
    load_data,
    preprocess_data,
    split_data,
    load_config
)
from src.model_training import (
    scale_features,
    train_linear_regression,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    print_evaluation_results,
    save_model
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    """
    Run the complete machine learning pipeline.
    
    This function:
    1. Loads and preprocesses the data
    2. Trains multiple models
    3. Evaluates model performance
    4. Saves the best model
    """
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        df = load_data()
        df_processed = preprocess_data(df)
        
        # Split data
        logger.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = split_data(df_processed, config)
        
        # Scale features
        logger.info("Scaling features")
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        
        # Train and evaluate models
        models = {
            'Linear Regression': train_linear_regression,
            'Random Forest': train_random_forest,
            'Gradient Boosting': train_gradient_boosting
        }
        
        results = {}
        for model_name, train_function in models.items():
            logger.info(f"Training {model_name}")
            model = train_function(X_train_scaled, y_train, config)
            model_results = evaluate_model(
                model, X_train_scaled, X_test_scaled, y_train, y_test, config
            )
            results[model_name] = model_results
            print_evaluation_results(model_name, model_results)
        
        # Select and save the best model
        best_model_name = min(
            results.items(),
            key=lambda x: x[1]['test_mse']
        )[0]
        logger.info(f"Best model: {best_model_name}")
        
        # Save the best model
        best_model = models[best_model_name](X_train_scaled, y_train, config)
        save_model(best_model, 'best_model')
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline() 