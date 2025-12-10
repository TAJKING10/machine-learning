# Credit Card Default Prediction

A comprehensive machine learning project for predicting credit card payment defaults using various classification algorithms.

## Overview

This project analyzes credit card customer data to predict the likelihood of default payment in the next month. It implements and compares four different machine learning models to identify the most effective approach for default prediction.

## Dataset

The analysis uses the **Credit Card Default Dataset** (`credit_card_default.csv`), which contains:
- **30,000 samples** of credit card customer data
- **23 features** including demographic information, credit data, payment history, and bill statements
- **Target variable**: Default payment next month (binary: 0 = No, 1 = Yes)

### Key Features
- `LIMIT_BAL`: Credit limit
- `SEX`: Gender (1=Male, 2=Female)
- `EDUCATION`: Education level (1=Graduate School, 2=University, 3=High School, 4=Others)
- `MARRIAGE`: Marital status
- `AGE`: Age in years
- `PAY_0` to `PAY_6`: Payment status for past 6 months
- `BILL_AMT1` to `BILL_AMT6`: Bill statement amounts for past 6 months
- `PAY_AMT1` to `PAY_AMT6`: Previous payment amounts for past 6 months

## Project Structure

```
machine-learning/
├── credit_card_analysis.py       # Main Python script
├── credit_card_analysis.ipynb    # Jupyter notebook version
├── credit_card_default.csv       # Dataset
├── model_results.csv             # Model performance metrics
├── Report.pdf                    # Detailed analysis report
├── Report.docx                   # Report source document
├── plots/                        # Generated visualizations
│   ├── 01_histogram_age.png
│   ├── 02_histogram_limit_bal.png
│   ├── 03_histogram_bill_amt1.png
│   ├── 04_countplot_sex_vs_default.png
│   ├── 05_barplot_education_vs_default.png
│   ├── 06_boxplot_limit_bal_by_default.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_confusion_matrices.png
│   ├── 09_roc_curves.png
│   ├── 10_model_performance_comparison.png
│   └── 11_feature_importance.png
└── README.md                     # This file
```

## Machine Learning Models

The project implements and compares four classification algorithms:

1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Non-linear tree-based classifier
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Advanced boosting ensemble method

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation

Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Running the Python Script

```bash
python credit_card_analysis.py
```

### Running the Jupyter Notebook

```bash
jupyter notebook credit_card_analysis.ipynb
```

## Analysis Pipeline

The analysis follows these steps:

1. **Data Loading** - Import and inspect the dataset
2. **Data Cleaning** - Handle missing values, remove duplicates, drop ID column
3. **Exploratory Data Analysis (EDA)** - Generate 11 visualizations including:
   - Distribution plots (Age, Credit Limit, Bill Amount)
   - Categorical analysis (Gender, Education vs Default)
   - Correlation heatmap
4. **Data Preprocessing** - Feature-target split, train-test split (80-20), feature scaling
5. **Model Training** - Train all four models with optimized parameters
6. **Model Evaluation** - Calculate metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
7. **Results Comparison** - Generate confusion matrices, ROC curves, and performance charts
8. **Feature Importance** - Identify the most influential features using Gradient Boosting

## Results

The models are evaluated using multiple metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to identify all positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve

All results are saved to `model_results.csv` and visualized in the `plots/` directory.

## Visualizations

The project generates 11 comprehensive visualizations:

- Histograms for numerical features (Age, Credit Limit, Bill Amount)
- Count plots showing gender distribution vs default status
- Bar plots analyzing education level impact on default rates
- Box plots comparing credit limits between defaulters and non-defaulters
- Correlation heatmap showing feature relationships
- Confusion matrices for all four models
- ROC curves comparing model performance
- Performance comparison bar charts
- Feature importance analysis

## Key Insights

- The most important features for predicting default are payment status variables (PAY_0 through PAY_6)
- Credit limit balance and bill amounts show significant correlation with default probability
- Ensemble methods (Random Forest, Gradient Boosting) typically outperform single classifiers
- The dataset shows class imbalance with more non-defaulters than defaulters

## Report

A detailed analysis report is available in both PDF and DOCX formats (`Report.pdf` and `Report.docx`), containing:
- Complete methodology
- Detailed results analysis
- Visualizations
- Conclusions and recommendations

## Author

Toufic Jandah 
