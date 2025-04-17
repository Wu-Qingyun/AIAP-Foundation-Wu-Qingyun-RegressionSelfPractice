import nbformat as nbf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell for title and project overview
nb.cells.append(nbf.v4.new_markdown_cell("""
# Exploratory Data Analysis (EDA) - Student Performance Dataset

## Project Overview
This notebook contains a comprehensive exploratory data analysis for the student performance dataset. The analysis aims to:
1. Understand the distribution and characteristics of the target variable (final test scores)
2. Explore relationships between features and the target variable
3. Identify potential patterns and insights that could inform model selection
4. Document data quality issues and preprocessing requirements

## Data Description
The dataset contains information about students' characteristics, study habits, and academic performance. Key features include:
- Demographic information (age, gender, number of siblings)
- Study habits (hours per week, learning style)
- Lifestyle factors (sleep patterns, CCA participation)
- Academic performance (final test scores)

## Analysis Structure
1. Data Loading and Basic Information
2. Data Quality Assessment
3. Target Variable Analysis
4. Feature Analysis
5. Correlation Analysis
6. Time-based Analysis
7. Key Findings and Insights
"""))

# Add code cell for imports and setup
nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)
"""))

# Add markdown cell for data loading
nb.cells.append(nbf.v4.new_markdown_cell("""
## 1. Data Loading and Basic Information

In this section, we:
1. Load the dataset
2. Display basic information about the data
3. Check for missing values
4. Examine data types
"""))

# Add code cell for loading data with error handling
nb.cells.append(nbf.v4.new_code_cell("""
try:
    # Load the dataset
    df = pd.read_csv('data/data.csv')
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    
    # Display basic information
    print("Dataset shape:", df.shape)
    print("\\nFirst few rows:")
    display(df.head())
    
    print("\\nData types:")
    display(df.dtypes)
    
    print("\\nMissing values:")
    display(df.isnull().sum())
    
    # Check for duplicate rows
    print("\\nNumber of duplicate rows:", df.duplicated().sum())
    
except FileNotFoundError:
    logger.error("Data file not found. Please ensure 'data/data.csv' exists.")
    raise
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise
"""))

# Add markdown cell for data quality assessment
nb.cells.append(nbf.v4.new_markdown_cell("""
## 2. Data Quality Assessment

In this section, we:
1. Examine data distributions
2. Identify outliers
3. Check for data consistency
4. Assess feature relationships
"""))

# Add code cell for data quality assessment
nb.cells.append(nbf.v4.new_code_cell("""
# Create a copy for analysis
df_analysis = df.copy()

# Numerical features summary
numerical_cols = ['number_of_siblings', 'n_male', 'n_female', 'age', 
                 'hours_per_week', 'attendance_rate', 'final_test']
print("\\nNumerical Features Summary:")
display(df_analysis[numerical_cols].describe())

# Categorical features summary
categorical_cols = ['CCA', 'learning_style', 'gender', 'direct_admission', 
                   'mode_of_transport', 'bag_color']
print("\\nCategorical Features Summary:")
for col in categorical_cols:
    print(f"\\n{col}:")
    display(df_analysis[col].value_counts(normalize=True))

# Check for outliers using box plots
plt.figure(figsize=(15, 6))
df_analysis[numerical_cols].boxplot()
plt.title('Box Plots of Numerical Features')
plt.xticks(rotation=45)
plt.show()

# Check for data consistency
print("\\nData Consistency Checks:")
# Check if age is within reasonable range
age_range = df_analysis['age'].between(10, 20)
print(f"Students with age outside 10-20 range: {~age_range.sum()}")

# Check if hours_per_week is reasonable
hours_range = df_analysis['hours_per_week'].between(0, 168)  # 168 hours in a week
print(f"Students with unreasonable study hours: {~hours_range.sum()}")

# Check if attendance_rate is within 0-100%
attendance_range = df_analysis['attendance_rate'].between(0, 100)
print(f"Students with attendance outside 0-100% range: {~attendance_range.sum()}")
"""))

# Add markdown cell for target variable analysis
nb.cells.append(nbf.v4.new_markdown_cell("""
## 3. Target Variable Analysis (final_test)

In this section, we:
1. Examine the distribution of final test scores
2. Identify patterns and outliers
3. Analyze the relationship with other features
"""))

# Add code cell for target variable analysis
nb.cells.append(nbf.v4.new_code_cell("""
# Basic statistics
print("Basic statistics for final_test:")
display(df_analysis['final_test'].describe())

# Distribution plot with kernel density estimation
plt.figure(figsize=(12, 6))
sns.histplot(data=df_analysis, x='final_test', kde=True, bins=30)
plt.title('Distribution of Final Test Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(y=df_analysis['final_test'])
plt.title('Box Plot of Final Test Scores')
plt.ylabel('Score')
plt.show()

# QQ plot for normality check
plt.figure(figsize=(10, 6))
from scipy import stats
stats.probplot(df_analysis['final_test'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Final Test Scores')
plt.show()

# Calculate skewness and kurtosis
print("\\nSkewness:", df_analysis['final_test'].skew())
print("Kurtosis:", df_analysis['final_test'].kurtosis())
"""))

# Add markdown cell for feature analysis
nb.cells.append(nbf.v4.new_markdown_cell("""
## 4. Feature Analysis

In this section, we:
1. Analyze numerical features
2. Analyze categorical features
3. Examine feature distributions
4. Identify potential feature engineering opportunities
"""))

# Add code cell for numerical features analysis
nb.cells.append(nbf.v4.new_code_cell("""
# Numerical features analysis
numerical_features = ['number_of_siblings', 'n_male', 'n_female', 'age', 
                     'hours_per_week', 'attendance_rate']

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df_analysis[numerical_features + ['final_test']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Scatter plots with final_test
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_analysis, x=feature, y='final_test', alpha=0.5)
    plt.title(f'Scatter Plot: {feature} vs Final Test Score')
    plt.xlabel(feature)
    plt.ylabel('Final Test Score')
    plt.show()
    
    # Calculate correlation coefficient
    corr = df_analysis[feature].corr(df_analysis['final_test'])
    print(f"Correlation between {feature} and final_test: {corr:.3f}")
"""))

# Add code cell for categorical features analysis
nb.cells.append(nbf.v4.new_code_cell("""
# Categorical features analysis
categorical_features = ['learning_style', 'gender', 'direct_admission', 
                       'mode_of_transport', 'bag_color']

# Box plots for categorical features
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_analysis, x=feature, y='final_test')
    plt.title(f'Box Plot: {feature} vs Final Test Score')
    plt.xlabel(feature)
    plt.ylabel('Final Test Score')
    plt.xticks(rotation=45)
    plt.show()
    
    # Calculate mean scores by category
    print(f"\\nMean scores by {feature}:")
    display(df_analysis.groupby(feature)['final_test'].agg(['mean', 'std', 'count']))
"""))

# Add markdown cell for time-based analysis
nb.cells.append(nbf.v4.new_markdown_cell("""
## 5. Time-based Analysis

In this section, we:
1. Analyze sleep patterns
2. Examine wake-up times
3. Study the relationship between sleep patterns and academic performance
"""))

# Add code cell for time-based analysis
nb.cells.append(nbf.v4.new_code_cell("""
# Sleep and wake time analysis
time_features = ['sleep_time', 'wake_time']

# Convert time columns to datetime
for col in time_features:
    df_analysis[col] = pd.to_datetime(df_analysis[col], format='%H:%M').dt.hour

# Distribution of sleep and wake times
for feature in time_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_analysis, x=feature, bins=24)
    plt.title(f'Distribution of {feature}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.show()

# Scatter plot of sleep time vs final test score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_analysis, x='sleep_time', y='final_test', alpha=0.5)
plt.title('Sleep Time vs Final Test Score')
plt.xlabel('Sleep Time (Hour)')
plt.ylabel('Final Test Score')
plt.show()

# Scatter plot of wake time vs final test score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_analysis, x='wake_time', y='final_test', alpha=0.5)
plt.title('Wake Time vs Final Test Score')
plt.xlabel('Wake Time (Hour)')
plt.ylabel('Final Test Score')
plt.show()

# Calculate sleep duration
df_analysis['sleep_duration'] = (df_analysis['wake_time'] - df_analysis['sleep_time']) % 24

# Plot sleep duration vs final test score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_analysis, x='sleep_duration', y='final_test', alpha=0.5)
plt.title('Sleep Duration vs Final Test Score')
plt.xlabel('Sleep Duration (Hours)')
plt.ylabel('Final Test Score')
plt.show()

# Calculate correlation between sleep duration and final test score
corr = df_analysis['sleep_duration'].corr(df_analysis['final_test'])
print(f"Correlation between sleep duration and final test score: {corr:.3f}")
"""))

# Add markdown cell for conclusion
nb.cells.append(nbf.v4.new_markdown_cell("""
## 6. Key Findings and Insights

Based on the exploratory data analysis, here are the key findings:

1. **Target Variable (final_test)**:
   - The distribution of final test scores shows [specific characteristics]
   - [Number] outliers were identified
   - The scores range from [min] to [max] with a mean of [mean]
   - The distribution is [skewed/symmetric] with [skewness value] skewness

2. **Numerical Features**:
   - [Feature 1] shows a [positive/negative] correlation with final test scores (r = [value])
   - [Feature 2] has a [strong/weak] relationship with the target variable
   - [Feature 3] appears to have no significant correlation with the scores

3. **Categorical Features**:
   - [Category 1] students tend to perform better than [Category 2]
   - [Feature] shows significant variation in performance across categories
   - [Feature] has minimal impact on the final test scores

4. **Time-based Features**:
   - Sleep and wake times show [some/no] correlation with performance
   - Optimal sleep patterns appear to be [description]
   - Sleep duration shows a [positive/negative] correlation with performance (r = [value])

5. **Data Quality Issues**:
   - [Number] missing values were found in [features]
   - [Number] outliers were identified in [features]
   - [Number] data points with inconsistent values were found

6. **Feature Engineering Opportunities**:
   - Sleep duration could be a useful derived feature
   - [Other potential feature engineering opportunities]

These insights will help inform the feature selection and model building process.
"""))

# Write the notebook to a file
with open('eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 