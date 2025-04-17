# Student Performance Prediction

This project aims to predict students' O-level mathematics examination scores using various machine learning models. The goal is to help schools identify weaker students prior to the examination.

## Project Structure

```
root /
    |- eda.ipynb                 # Exploratory Data Analysis notebook
    |- README.md                 # Project documentation
    |- requirements.txt          # Python package dependencies
    |- data /
           |- data.csv           # Dataset
    |- src /
           |- data_preparation.py # Data loading and preprocessing
           |- model_training.py   # Model training and evaluation
           |- config.yaml        # Configuration file
    |- main.py                   # Main script to run the pipeline
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. For exploratory data analysis, open the Jupyter notebook:
```bash
jupyter notebook eda.ipynb
```

## Models

The project evaluates three regression models:
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The models are compared based on:
- Mean Squared Error (MSE)
- R-squared score
- Cross-validation performance
- Training vs Testing performance 