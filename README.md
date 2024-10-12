# Titanic Survival Prediction

This project predicts the survival of passengers aboard the Titanic using machine learning techniques. The dataset contains information about the passengers, such as age, gender, ticket class, and other attributes that could help determine their likelihood of survival.

### Project Overview
The Titanic Survival Prediction project aims to demonstrate the power of data analysis and machine learning in deriving insights from historical data. By applying various classification algorithms, we build a model that predicts whether a passenger survived or not, based on certain features from the Titanic dataset.

### Dataset
The dataset used in this project is sourced from Kaggle's Titanic competition. The data consists of the following features:

- PassengerId: A unique identifier for each passenger
- Pclass: The class of the ticket (1st, 2nd, 3rd)
- Name: Passenger’s name
- Sex: Passenger’s gender
- Age: Passenger’s age
- SibSp: Number of siblings/spouses aboard the Titanic
- Parch: Number of parents/children aboard the Titanic
- Ticket: Ticket number
- Fare: Amount of money the passenger paid for the ticket
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- Survived: Survival status (0 = Did not survive, 1 = Survived)
  
# Project Workflow

### 1.Exploratory Data Analysis (EDA)

- Cleaning the dataset (handling missing values, encoding categorical variables).
- Visualizing key features and their relationships to survival using libraries like matplotlib and seaborn.

### 2.Feature Engineering

- Creating new features from the existing data (e.g., family size, title extraction from names).
- Encoding categorical variables such as gender and embarked port.

### 3.Modeling

- Applying various machine learning algorithms such as Logistic Regression, Random Forest, Support Vector Machines, and others.
- Using scikit-learn to train, test, and evaluate the models.
- Hyperparameter tuning using GridSearchCV to improve model performance.

### 4.Evaluation

- Evaluating model performance using accuracy, precision, recall, and F1-score.
- Comparison of model performances using cross-validation.

### 5.Prediction

- Making predictions on the test dataset and exporting the results in the required format.

### Libraries Used

- Pandas: For data manipulation and analysis.
- NumPy: For numerical computing.
- Matplotlib/Seaborn: For data visualization.
- Scikit-learn: For machine learning model building and evaluation.
