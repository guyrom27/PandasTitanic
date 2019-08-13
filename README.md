# Titanic
Analyzing the Titanic data from kaggle with python sklearn.
I started with Random Forests, and performed some parameter tuning, i.e. amount of trees
![rf_roc]
(https://github.com/guyrom27/PandasTitanic/blob/master/rf_roc_and_acc.png)

I then analyzed the errors
![RF_age](https://github.com/guyrom27/PandasTitanic/blob/master/RF_age_errors.png)
![RF_class](https://github.com/guyrom27/PandasTitanic/blob/master/RF_class_dist.png)

Then I tried to use ADABoost, thinking maybe the boosting process will focus on the subset that has higher error rates, e.g. men in 3rd class, and perform better than the uniformly weighted random forests.

So, I started with tuning the amount of estimators ADABoost uses, with a decision stump as base estimator
![ADA_rounds](https://github.com/guyrom27/PandasTitanic/blob/master/ADA_rounds.png)
Then I tried to use a richer weak learner, as the training accuracy was not reaching saturation after 300 estimators/rounds. I used Decision Trees with leaf_sample_size fractional regularization. Tuning regularization and number of estimators
![ADA_lead_norm](https://github.com/guyrom27/PandasTitanic/blob/master/ADA_leaf_norm.png)

In addition, I checked if a very low regularization and just a few estimators will be good, averaging over more samples to get a less noisy estimate of the accuracy, but it was still noisy.

# The Final Results
Random forest: 63 trees Accuracy: 0.822
ADABoost:
32 estimators, decision stumps- Accuracy: 0.833
0.25 leaf regularization and 138 rounds Accuracy: 0.822
0.05 leaf regularization and 2 rounds Accuracy: 0.833
As the training set is quite small, the difference in accuracy is due to +-1 error, and probably doesn't mean anything, hence they are all equally good