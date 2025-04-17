from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from typing import Dict, Any, Tuple, Union
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled training and testing features
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Features scaled successfully")
        return X_train_scaled, X_test_scaled
    except Exception as e:
        logger.error(f"Error scaling features: {str(e)}")
        raise

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray, 
                          config: Dict[str, Any]) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        LinearRegression: Trained model
    """
    try:
        model = LinearRegression(**config['models']['linear_regression'])
        model.fit(X_train, y_train)
        logger.info("Linear Regression model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training Linear Regression model: {str(e)}")
        raise

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       config: Dict[str, Any]) -> RandomForestRegressor:
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        RandomForestRegressor: Trained model
    """
    try:
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize base model
        base_model = RandomForestRegressor(random_state=config['data']['random_state'])
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=config['evaluation']['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Random Forest best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error training Random Forest model: {str(e)}")
        raise

def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray, 
                          config: Dict[str, Any]) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting model with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        GradientBoostingRegressor: Trained model
    """
    try:
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize base model
        base_model = GradientBoostingRegressor(random_state=config['data']['random_state'])
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=config['evaluation']['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Gradient Boosting best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error training Gradient Boosting model: {str(e)}")
        raise

def evaluate_model(model: Union[LinearRegression, RandomForestRegressor, GradientBoostingRegressor],
                  X_train: np.ndarray, X_test: np.ndarray,
                  y_train: np.ndarray, y_test: np.ndarray,
                  config: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate a model's performance using various metrics.
    
    Args:
        model: Trained model
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training target
        y_test (np.ndarray): Testing target
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=config['evaluation']['cv_folds'],
            scoring='neg_mean_squared_error'
        )
        cv_mean = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Calculate feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'feature_importance': feature_importance
        }
        
        logger.info(f"Model evaluation completed. Test R²: {test_r2:.4f}")
        return results
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_model(model: Union[LinearRegression, RandomForestRegressor, GradientBoostingRegressor],
              model_name: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model_path = f'models/{model_name}.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def print_evaluation_results(model_name: str, results: Dict[str, float]) -> None:
    """
    Print model evaluation results in a formatted way.
    
    Args:
        model_name (str): Name of the model
        results (Dict[str, float]): Dictionary of evaluation metrics
    """
    print(f"\n{model_name} Results:")
    print("-" * 50)
    print(f"Training MSE: {results['train_mse']:.4f}")
    print(f"Testing MSE: {results['test_mse']:.4f}")
    print(f"Training R²: {results['train_r2']:.4f}")
    print(f"Testing R²: {results['test_r2']:.4f}")
    print(f"Cross-validation MSE: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
    
    if results['feature_importance'] is not None:
        print("\nFeature Importance:")
        for i, importance in enumerate(results['feature_importance']):
            print(f"Feature {i}: {importance:.4f}") 