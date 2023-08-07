# Module 12 Report Template

## Solution 

[credit_risk_classification.ipynb](Credit_Risk/credit_risk_classification.ipynb)

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

The purpose of this analysis was to analyze loans and based on the features of these loans, determine if they are healthy loans or high-risk loans.  The features of the loans that were used were the loan size, the interest rate of the loan, the borrower's income, the debt to income ratio, the number of accounts, derogatory marks, and the total debt.  To do this, an unsupervised learning approach was taken, where data that contained previous loans' features and their outcomes were used to train a logistic regression machine learning model.  Before training the model, the data was split into two main subsets - a training subset and a testing subset.  With this, the training subset was used for training the model and the testing subset was used to evaluate the models performance.  After the model was trained and evaluted, the balanced accuracy score, a confusion matrix, and the classification report were obtained.  Because it was discovered that the data was lop-sided due to the fact that there were an overwhelming amount of healthy loans versus high-risk loans, a random over sampler method was used to intelligently oversample the data and create an even amount of healthy loans and high-risk loans with appropriate features.  These new examples were then re-trained and evaluated with another logistic regression model, in which better results were seen.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  - Accuracy
    - The model's accuracy was 92.18\%.  This means that around 92\% of the test examples were classified correctly.
  - Precision
    - For healthy loans, the precision score was 1.00.  This means that all of the healthy loan examples were classified correctly, with no false positives.
    - For high-risk loans, the precision score was 0.85.  This means that there were false positives with high-risk loans about 15\% of the time.
  - Recall scores
    - For healthy loans, the recall score was 0.99.  This means that there were a very small amount (1\%) of false negatives for healthy loans.
    - For high-risk loans, the recall score was 0.91.  This means that around 9\% of high-risk loan examples were identified as false negatives by the model.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  - Accuracy
    - The model's accuracy was 99.37\%.  This means that around 99\% of the test examples were classified correctly.  This was much better than model 1.
  - Precision
    - For healthy loans, the precision score was 1.00.  This means that all of the healthy loan examples were classified correctly, with no false positives.  This result is the same as model 1.
    - For high-risk loans, the precision score was 0.84.  This means that there were false positives with high-risk loans about 16\% of the time.  This number was lower than model 1, meaning that model 2 has a worse precision score for high-risk loans.
  - Recall scores
    - For healthy loans, the recall score was 0.99.  This means that there were a very small amount (1\%) of false negatives for healthy loans.  This result is the same as model 1.
    - For high-risk loans, the recall score was 0.99.  This means that around 1\% of high-risk loan examples were identified as false negatives by the model.  This was a great improvement from model 1, as the recall score for high-risk loans is the same as the recall score for healthy loans.
  

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

Overall, the second model performed better than the first model.  This is primarily shown in the increase in f1 and recall scores for high-risk loans with the second model.  It seems that it is more important to identify high-risk loan examples ("1"s), as these are more rare than healthy loans and they are the type of loans that a lender would want to stay away from.  To recommend a model for predicting healthy loans, both model 1 and model 2 would do this well.  To recommend a model for predicting high-risk loans, the better model to go with would be model 2, as this model did a better job of classifying high-risk loans with a lower amount of false negatives.
