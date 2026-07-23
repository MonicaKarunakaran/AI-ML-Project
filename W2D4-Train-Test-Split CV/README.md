# W2D4 - Train/Test Split & Cross Validation

## Objective
Implemented train-test splitting and cross-validation techniques in a machine learning workflow.

## Concepts Learned

### Train/Test Split
Divided dataset into training and testing data to evaluate model performance on unseen data.

### StandardScaler
Used to transform features into zero mean and unit variance.
Suitable for algorithms like Logistic Regression, SVM, and KNN.

### MinMaxScaler
Scaled values between 0 and 1.
Useful when features have different ranges.

### RobustScaler
Uses median and IQR, making it less affected by outliers.

### Cross Validation
Used 5-fold cross-validation to check model performance across multiple splits.

## Model Used
Logistic Regression

## Dataset
Breast Cancer Dataset from Scikit-Learn

## Result
Compared different scaling methods and evaluated model performance using accuracy score and cross-validation.