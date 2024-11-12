# Titanic Survival Prediction

### This project uses machine learning techniques to predict passenger survival on the Titanic based on various features like age, gender, ticket class, and fare.

---

## Requirements
- Python 3.x
- Libraries:
   - Data manipulation: pandas, numpy
   - Data visualization: matplotlib, seaborn
   - Machine learning models: scikit-learn
     
## Setup
1. **Library Imports:**
   - Import necessary libraries for data manipulation (numpy, pandas), visualization (seaborn, matplotlib), and machine learning (scikit-learn).
     
2. **Google Drive Setup:**
   - If using Google Colab, mount Google Drive to access data files.

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Load Data:**

   - Load train.csv and test.csv datasets from Google Drive or local storage.
```python
test_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/test.csv")
train_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/train.csv")
```

## Data Preprocessing Steps
1. **Missing Data Analysis:**
   - Check and summarize missing values in each column for targeted handling.

2. **Exploratory Data Analysis (EDA):**
   - Visualize survival rates by gender and age, and examine relationships between other features and survival.
     
3. **Feature Engineering:**
   - Create new features: relatives and not_alone indicate the number of family members onboard.
   - Extract deck from cabin: Extract and encode deck information from the Cabin column and drop the original Cabin feature.
   - Fill missing values in Age: Randomly assign ages based on mean and standard deviation of existing values.
   - Encode categorical data: Map categorical values to numeric labels for Sex, Embarked, and Title.
     
4. **Feature Transformation:**
   - Binning: Group Age and Fare into bins to simplify the model’s decision process.
   - New combined features: Age_Class (product of Age and Pclass) and Fare_Per_Person (fare divided by the number of family members onboard).

## Model Preparation

1. **Define Features and Target Variable:**
   - Separate target variable Survived and features from train_df for model training.
```python
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
```
2. **Algorithm Selection:**
   - Prepare to experiment with various algorithms including LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier, GaussianNB, and more to find the best model for predicting survival.

## Running the Project
To replicate this analysis:
1. ***Load the necessary datasets.***
2. ***Execute each preprocessing and visualization step to inspect and transform the data.***
3. ***Train and evaluate different models using the prepared X_train, Y_train, and X_test.***
