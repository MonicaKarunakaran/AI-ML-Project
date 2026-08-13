# W2D2 Feature Scaling & Selection

## Concepts Covered

### Encoding Techniques
- LabelEncoder
- OneHotEncoder
- OrdinalEncoder

### Scaling Techniques
- StandardScaler
- MinMaxScaler
- RobustScaler

### Feature Selection
Used SelectKBest with ANOVA F-test to identify important features.

## Observations

- OneHotEncoder is suitable for categorical variables without order.
- OrdinalEncoder is used when categories have ranking.
- StandardScaler is affected by outliers.
- RobustScaler performs better with outliers.
- SelectKBest helps remove less important features.

## Output

Generated before and after scaling distribution plots.