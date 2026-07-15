# idx-data

##  1. Overview
df.info()
df.describe()

##  2. Missing
df.isnull().sum()

## 3. Duplicates
df.duplicated().sum()

## 4. Data types
df.dtypes

## 5. Unique categories
df.nunique()

## 6. Outliers
df.describe()

## 7. Clean
drop duplicates
fix types
impute missing
remove impossible values

Fill NA model: 
Summary table → "Which columns have missing values?"
Missing bar plot → "How bad is it?"
Missing heatmap → "Do missing values happen together?"
Distribution plots → "How should I fill them?"

## 8. Train Test Split & Column Transformation(normalization, encoding)
- transform separately on training and testing data