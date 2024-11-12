# ML - Titanic Survival Prediction with H2O and Scikit-Learn

# Titanic Survival Prediction with H2O and Scikit-Learn

This project uses machine learning techniques to predict Titanic passenger survival. The workflow includes data preprocessing, exploratory analysis, model building, and evaluation, using both traditional machine learning (Logistic Regression) and advanced methods (Ensemble Stacking, H2O AutoML).

## Requirements

- Python 3.x
- Libraries: pandas, scikit-learn, h2o, xgboost

## Files

- `train.csv`: Training dataset with survival labels and passenger features.
- `test.csv`: Test dataset without survival labels.
- `gender_submission.csv`: Sample submission file format.

## Data Processing Steps

1. **Data Loading**: Load the training, test, and submission data.
2. **Exploratory Data Analysis (EDA)**:
   - Check dataset shapes, data types, and null values.
   - Analyze unique values for categorical features.
3. **Missing Value Handling**:
   - Fill missing `Age` and `Fare` values with mean.
   - Use most frequent value to fill missing `Cabin` and `Embarked` entries.
4. **Label Encoding**:
   - Encode categorical features (`Sex`, `Name`, `Ticket`, `Cabin`, `Embarked`) for model compatibility.
5. **Feature Selection**:
   - Drop irrelevant features like `PassengerId` and `Name` to reduce feature space.

## Model Training

1. **Train-Test Split**:
   - Split the data into 80% training and 20% validation for model evaluation.
   
2. **Models**:
   - **Logistic Regression**: Baseline model with 1000 max iterations for initial prediction.
   - **Ensemble Stacking**: Combined model with Random Forest, SVM, and XGBoost as base models, using Logistic Regression as the meta-model.
   - **H2O AutoML**: Automated model selection with H2O’s AutoML for additional performance tuning.

3. **Model Evaluation**:
   - Evaluate models using accuracy on the validation set.

## Running the Pipeline

To execute the code, follow these steps:
1. Install the required libraries, specifically H2O (`!pip install h2o`).
2. Load data from the provided paths.
3. Run each section sequentially: data loading, EDA, data preprocessing, model training, and evaluation.

## Output

- The final survival predictions are saved in a CSV file (`/content/drive/MyDrive/Colab Notebooks/sub.csv`) formatted for submission.
- Additional predictions from H2O AutoML are saved as `h2o_submission.csv`.

## Conclusion

This pipeline offers a comprehensive approach to predict survival on the Titanic, utilizing traditional and advanced machine learning methods to improve accuracy and robustness.
