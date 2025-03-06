# Car Evaluation Machine Learning Project

## Overview

This project applies various machine learning models to classify cars based on different evaluation criteria. The dataset used for this project is the Car Evaluation Dataset, which contains categorical and numerical features related to car specifications and their classifications. The goal is to compare the performance of different models and optimize them for better accuracy.

## Dataset

The dataset contains features representing car attributes, such as:

- Buying price
- Maintenance cost
- Number of doors
- Capacity
- Safety rating
- Class label (Target variable)

## Question
Out of the six features given in the dataset (Buying Price, Maintenance Price, Doors, Persons, Capacity, and Safety), which feature has the greatest influence on finding the target (unacceptable, acceptable, good, or very good)? What is the best Machine Learning Algorithm to most accurately represent the data, Decision Trees, Support Vector Machines (SVM), or Naive Bayes?

## Models Implemented

### 1. Decision Tree Classifier
- Trains a Decision Tree model to classify cars based on the most important features.
- Hyperparameter tuning using RandomizedSearchCV to optimize performance.
- Feature importance analysis to select the most relevant features.
- Model performance measured using accuracy and classification reports.
- **Note**: Adding hyperparameters such as max depth and max leaf nodes caused a significant drop in accuracy. This is because the parameters for the decision tree were restricted more than in the default model.  
  - **Accuracy**: 98%

### 2. Naive Bayes Classifiers

#### Gaussian Naive Bayes
- Assumes every variable has a normal distribution (also called Gaussian Distribution).
- The normal distribution gets rid of any skewed data and analyzes data when there is an equally likely chance of the data being above or below the mean.
- Uses Bayes Theorem to find probability.
- Works well with high-dimensional data.
- **Accuracy**: 84%

#### Bernoulli Naive Bayes
- Uses a Bernoulli distribution, which is essentially a binary classification (0 or 1) of the data.
- Best used for text classification (spam detection).
- Very computationally efficient.
- Assumes features are independent of one another.
- **Accuracy**: 80%

- **Note**: I decided to compare the difference in accuracies between Gaussian Naive Bayes and Bernoulli Naive Bayes. With Gaussian Naive Bayes, the algorithm uses a normal distribution and Bayes Theorem to calculate probability. In contrast, Bernoulli Naive Bayes uses a Bernoulli distribution, which is ideal for binary classification problems like text classification.

### 3. Support Vector Machines (SVM)

#### Radial Basis Function (RBF) SVM
- Best used for linearly inseparable data points.
- Creates a hyperplane to separate the data points in a higher-dimensional space.
- Better suited for more complex datasets.
- **Accuracy**: 84%
- **F1 Score**: 0.8385
- **Explanation**: The Radial Basis Function (RBF) SVM uses an equation to measure the Euclidean distance between data points, creating a hyperplane to separate them based on the dataset's dimensions.

#### Polynomial SVM
- Compares nth-dimensional relationships between observations.
- Like RBF, it uses a hyperplane to separate data points, but with a polynomial transformation of the input data into a higher-dimensional space.
- **Accuracy**: 70.5%
- **F1 Score**: 0.7061
- **Explanation**: Polynomial SVM uses a polynomial function to transform the input data into a higher-dimensional space and then uses a hyperplane to separate the points based on target values. Polynomial SVM is computationally less expensive compared to RBF SVM.

- **Note**: Two different types of SVM (RBF and Polynomial kernels) were used in this project. While the RBF kernel is better for complex datasets with non-linear relationships, the Polynomial kernel is simpler and less computationally expensive.

## Performance Comparison

- **Naive Bayes**: Higher overall accuracies and much quicker run times compared to SVM.
  - **Gaussian Naive Bayes Accuracy**: 84%
  - **Bernoulli Naive Bayes Accuracy**: 80%
  
- **SVM**: Lower overall accuracies and much slower run times (sometimes didn’t even run at all).
  - **RBF SVM Accuracy**: 84%
  - **Polynomial SVM Accuracy**: 70.5%

### Conclusion
- Out of the three algorithms (Decision Tree, Naive Bayes, and SVM), the **Decision Tree** had the best accuracy at 98%.
- The best feature of the dataset to accurately evaluate the car is **Safety**.
- The worst feature of the dataset to accurately evaluate the car is the **Number of Doors**.
- This algorithm can be used to help evaluate more than just cars; it can be used to evaluate other motor vehicles like **boats**, **motorcycles**, etc.

## Performance Metrics

For each model, the following performance metrics were used:
- Accuracy Score: Measures how many predictions are correct.
- F1 Score: Balances precision and recall for multi-class classification.
- Confusion Matrix: Helps visualize classification errors.
- Feature Importance: Determines which features contribute most to predictions.

## Visualizations

The project includes multiple visualizations:
- Decision trees to interpret model decisions.
- Feature importance graphs to understand significant attributes.
- Decision boundaries to analyze classification regions of models.

## Next Steps
- Incorporate additional feature engineering techniques to improve accuracy.
- Experiment with ensemble learning techniques (e.g., Random Forest, Gradient Boosting).
- Compare with deep learning approaches for potential improvements.

## Link to Presentation
- https://docs.google.com/presentation/d/1ughmdCZlLxyhQJu84OzfYdtRL832bp8nhw0mnA6P28c/edit?usp=sharing 
