# College Data Analysis and Classification Project

## Overview

This project performs comprehensive analysis on a college dataset containing information about various US colleges and universities. The main objectives are to explore the data, create meaningful visualizations, and build machine learning models to predict whether a college is private or public, as well as predict graduation rates.

## Dataset Description

The dataset (`College_Data.xls`) contains information about 777 colleges and universities with the following features:

### Original Features:
- **Private**: Whether the college is private ("Yes") or public ("No")
- **Apps**: Number of applications received
- **Accept**: Number of applications accepted  
- **Enroll**: Number of new students enrolled
- **Top10perc**: Percentage of new students from top 10% of high school class
- **Top25perc**: Percentage of new students from top 25% of high school class
- **F.Undergrad**: Number of full-time undergraduates
- **P.Undergrad**: Number of part-time undergraduates
- **Outstate**: Out-of-state tuition
- **Room.Board**: Room and board costs
- **Books**: Estimated book costs
- **Personal**: Estimated personal spending
- **PhD**: Percentage of faculty with Ph.D.'s
- **Terminal**: Percentage of faculty with terminal degree
- **S.F.Ratio**: Student/faculty ratio
- **perc.alumni**: Percentage of alumni who donate
- **Expend**: Instructional expenditure per student
- **Grad.Rate**: Graduation rate

### Engineered Features:
The notebook creates several new meaningful features:
- **Private_bin**: Binary encoding of Private (1 for Yes, 0 for No)
- **accept_rate**: Acceptance rate (Accept/Apps)
- **yield_rate**: Yield rate (Enroll/Accept)
- **total_undergrad**: Total undergraduates (F.Undergrad + P.Undergrad)
- **pct_female_undergrad**: Percentage of female undergraduates
- **total_cost**: Estimated total annual cost (Outstate + Room.Board + Books + Personal)

## Project Structure

```
├── college_data.ipynb          # Main analysis notebook
├── College_Data.xls           # Raw dataset
├── college_private_classifier.pkl  # Saved trained model
└── README.md                  # This file
```

## Analysis Workflow

### 1. Data Loading and Exploration
- Load the dataset using pandas
- Inspect data shape, types, and basic statistics
- Check for missing values and class balance
- Examine the distribution of key variables

### 2. Data Preprocessing
- Standardize column names
- Handle missing values using median imputation for numeric columns
- Remove duplicate entries
- Create binary encoding for categorical variables
- Convert data types as needed

### 3. Feature Engineering
- Calculate acceptance and yield rates
- Create total cost estimates
- Generate demographic ratios
- Handle potential divide-by-zero cases

### 4. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Distribution of tuition, graduation rates, etc.
- **Bivariate Analysis**: Relationships between variables (e.g., tuition vs graduation rate)
- **Correlation Analysis**: Heatmap showing relationships between numeric features
- **Group Comparisons**: Private vs public college characteristics

### 5. Machine Learning Models

#### Classification Models (Predicting Private vs Public)

**Logistic Regression with Pipeline**
- Uses StandardScaler for feature normalization
- Implements logistic regression for binary classification
- Provides baseline performance metrics

**Random Forest Classifier**
- Ensemble method with 200 trees
- Provides feature importance rankings
- Better handling of non-linear relationships

**Hyperparameter Tuned Random Forest**
- Grid search optimization for:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20] 
  - `min_samples_split`: [2, 5]
- Cross-validation for robust performance estimation
- **Final model saved as `college_private_classifier.pkl`**

#### Regression Models (Predicting Graduation Rate)

**Linear Regression**
- Simple baseline model for graduation rate prediction
- Provides interpretable coefficients

**Random Forest Regressor**
- Non-linear model for improved prediction accuracy
- Handles feature interactions automatically

## Key Findings and Insights

### Model Performance
The classification models show strong performance in distinguishing between private and public colleges, with the Random Forest classifier achieving the best results after hyperparameter tuning.

### Important Features
Based on Random Forest feature importance analysis, the most predictive features for college type include:
- Financial factors (tuition, costs)
- Admission statistics (acceptance rates, enrollment)
- Academic quality indicators (graduation rates, faculty credentials)

### Data Patterns
- Clear differences exist between private and public institutions in terms of costs and selectivity
- Strong correlations between various cost components
- Academic performance indicators show meaningful relationships with institutional characteristics

## Usage

### Running the Analysis
1. Ensure you have the required dependencies installed:
   ```python
   pandas, matplotlib, seaborn, scikit-learn, joblib
   ```

2. Open and run `college_data.ipynb` in Jupyter Notebook or VS Code

3. The notebook runs from start to finish, performing all analysis steps

### Using the Trained Model
```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("college_private_classifier.pkl")

# Make predictions on new data
# (ensure new data has the same features used in training)
predictions = model.predict(new_college_data)
```

## Technical Requirements

- **Python 3.7+**
- **Libraries**: pandas, matplotlib, seaborn, scikit-learn, joblib, numpy
- **Environment**: Jupyter Notebook or compatible notebook environment

## Model Files

- **`college_private_classifier.pkl`**: Trained Random Forest classifier for predicting whether a college is private or public. This model achieved optimal performance through grid search hyperparameter tuning and can be loaded using joblib for making predictions on new college data.

## Future Enhancements

Potential areas for improvement:
1. **Advanced Feature Engineering**: Create more sophisticated derived features
2. **Model Ensemble**: Combine multiple models for improved performance
3. **Deep Learning**: Experiment with neural network approaches
4. **External Data**: Incorporate additional college ranking or demographic data
5. **Interactive Visualizations**: Create dashboards for data exploration
6. **Model Deployment**: Package model for web application or API use

## Contact

For questions or suggestions about this analysis, please refer to the notebook documentation or create an issue in the repository.