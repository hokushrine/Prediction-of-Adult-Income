# Predicting Adult Income
## Analysis of income and related columns to identify factors that determine the income of adults 

**Author**: Matt S.

### Business problem:
Determine the income level for adults based off of several factors, such as education, job, ethnicity, sex, and more. Using this data,
what areas can be improved to increase income for adults within the related sectors.


### Data:
Data dictionary:
| Variable Name | Description |
|:-------------:|:-----------:|
|      age      | continuous  |
| workclass | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked|
| fnlwgt | continuous|
| education| Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool|
| education-num| continuous|
| marital-status| Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse|
| occupation| Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces|
| relationship| Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried|
| race| White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black|
| sex| Female, Male|
| capital-gain| continuous|
| capital-loss| continuous|
| hours-per-week| continuous|
| native-country| United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands|
class| >50K, <=50K|



## Methods
Muliple steps were taken to prepare and review the data including:
- Initial data inspection and cleaning
- Exploratory Data Analysis
- Explanatory Data Analysis
- Data pre-processing for ML after exploratory and explanatory analysis

## EDA
Individuals over 35 typically make more than 50k.
![](https://github.com/hokushrine/Prediction-of-Adult-Income/blob/main/images/income_vs_age.png?raw=true)


The native-country column ended up not being very useful due to most people being from the United States. Data is either lacking or biased.
![](https://github.com/hokushrine/Prediction-of-Adult-Income/blob/main/images/income_by_native-country.png?raw=true)

## Model
Models used:
```
- Dummy Classifier (baseline)
- Logistic Regression (untuned)
- Logistic Regression (tuned)
- Logistic Regression (tuned w/ PCA)
- KNN (untuned)
- KNN (tuned)
- KNN (tuned w/ PCA)
```
### Final Model Output
Tuned Logistic Regression Classifier:
```
Accuracy: 0.86
Classification Report:
              precision    recall  f1-score   support

       <=50K       0.89      0.93      0.91      9354
        >50K       0.74      0.61      0.67      2857

    accuracy                           0.86     12211
   macro avg       0.81      0.77      0.79     12211
weighted avg       0.85      0.86      0.85     12211
```

Tuned KNN Classifier:
```
Accuracy: 0.83
Classification Report:
              precision    recall  f1-score   support

       <=50K       0.88      0.90      0.89      9354
        >50K       0.64      0.60      0.62      2857

    accuracy                           0.83     12211
   macro avg       0.76      0.75      0.76     12211
weighted avg       0.82      0.83      0.83     12211
weighted avg       0.82      0.83      0.83     12211
``````

The TLR model outperforms the Tuned KNN model for predicting income levels in the given dataset due to the following reasons:

- Higher Accuracy: The Logistic Regression model achieves a higher overall accuracy (0.86) compared to the KNN model (0.83), indicating that it makes more correct predictions.

- Precision and Recall: The Logistic Regression model shows better  precision, recall, and F1-scores for both income classes (<=50K and >50K).
  - It achieves higher precision, indicating fewer false positives, and higher recall, implying better identification of true positives.

- F1-Scores: The Logistic Regression model achieves higher macro and weighted average F1-scores (0.79 and 0.85, respectively) compared to the KNN model (0.76 and 0.83), indicating better overall balance between precision and recall for both classes.



## Recommendations:
The tuned Logistic Regression with PCA model runs very well and outperforms the KNN model variants in all areas.

## Limitations & Next Steps
- Refactor code to be more modular (move code into functions etc.)
- Consider using SMOTE or othere methods to tackle class unbalance