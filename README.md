# Documentation

## Project Overview <br>
Enron was one of the largest companies in the United States about 20 years ago until it went bankrupt in 2002 as a result of corporate fraud. Since then, much of the previously confidential data has become publically available - including emails and employee financial data. The purpose of this project is to utilize the data to identify "Persons of Interest (POIs).  <br>
 <br>
Much of the data used in this project was previously engineered for the purpose of this exercise. It includes information for almost 150 individuals. Some of the data includes the following:

`poi` - Whether the individual was a POI <br>
`to_messages` - Total messages received <br>
`from_messages` - Total messages sent <br>
`from_poi_to_this_person` - Messages received by this individual by a POI <br>
`from_this_person_to_poi` - Messages sent from this individual to a POI <br>
`email_address` - Email address of the individual <br>

Along with the above data, there is also information regarding each individual's finances (i.e. salary, bonuses, stock information, etc). Although the majority of the data is useful, there are still many missing values. Although it is never specified, it is inferred that each of these null values should equal zero (this is done in the data cleaning process). There are also a couple of outlier data entries, including "Total" and "The Travel Agency in the Park". Because of the nature of the project, neither will be helpful in detecting POIs, and were thus removed altogether. Although there were some individuals with extreme data values that might also be considered "outliers" in a statistical sense, they were not removed from the dataset. This is because the outlying data might be a determining factor in finding POIs. <br>

As previously stated, the purpose of the project is to identify POIs. This is done by "training" a machine learning algorithm on samples of the data, which can then be applied generally accross the dataset to identify the correct persons. 

### Project File Descriptions <br>

**poi_id.py** - Complete code including data cleaning/EDA, algorithm testing/tuning, and saving the final files <br>
**poi_id.ipynb** - Notebook version of the above file <br>
**enron61702insiderpay.pdf** - Complete description of the financial data <br>
**poi_names.txt** - Names of all POIs in the dataset <br>
**poi_email_addresses.py** - All email addresses in the dataset <br>
**feature_format.py** - Includes some of the data reading/saving functions utilized throughout the project <br>
**tester.py** - Includes functions used to save/read/test the final algorithm <br>
**dos_to_unix.py** - Used convert dos formatting in the original dataset to unix  <br>
**final_project_dataset.pkl** - Dataset used in project <br>
**my_classifier.pkl** - Final algorithm used <br>
**my_dataset.pkl** - Final dataset after data cleaning, standardization, feature selection, etc <br>
**my_feature_list.pkl** - List of features used in final algorithm 

### Feature Selection

In my intitial analysis of the data after removing outliers and filling the null values, I engineered the following new features in an effort to summarize relationships in the data: <br>

`total_restricted_stock_diff`: total stock value - restricted stock <br>
`to_poi_ratio`: ratio of emails to POI <br>
`from_poi_ratio`: ratio of emails from POI <br>
`payments_to_stock_ratio`: ratio of total payments to total stock value <br>

Of these features, only `total_restricted_stock_diff` was included in the final dataset. In fact, only four features were used in the final algorithm (selected during the parameter tuning process):

`bonus` (score ~ 21.06) <br>
`total_stock_value` (score ~ 24.47) <br>
`exercised_stock_options` (score ~ 25.10) <br>
`total_restricted_stock_diff` (score ~ 25.66)

These features were selected using the sklearn `SelectKBest` algorithm. <br>
*https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html* <br>

The features of the final algorithm were also scaled using the sklearn `MinMaxScaler` method. Although scaling is not strictly necessary in the gradient boosting (the final algorithm that was used - see the next section for more details), it speeds up the training process. <br>
*https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html* <br>
*https://math.stackexchange.com/questions/2341704/feature-scalings-effect-on-gradient-descent*

### Machine Learning Algorithms

The following algorithms were all attempted during this project:

 1. Gaussian Naive Bayes Classifier
 2. Random Forest Classifier
 3. Support Vector Machine
 4. Gradient Boosting Classifier <br>
 
 *https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html* <br>
 *https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html* <br>
 *https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC* <br>
 *https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html*

As stated previously, the final selection was Gradient Boosting. <br>

In simple terms, Gradient boosting involves optimization using gradient descent to minimize the loss when additively combining Desision Trees. <br>
*https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/*

Model performance varied while I tested all of the algorithms, but they were generally pretty close. Principal Component Analysis (PCA) improved the accuracy for all four of them. However, prior to tuning parameters, NONE of the algorithms were reaching recall and precision scores of at least 0.3, which was the objective.

### Tuning Algorithm Parameters and Validation

As stated previously, not tuning the parameters resulted in unsatisfactory results. Especially because the majority of the individuals in the dataset were not POIs, the algorithms tended to classify the majority of individuals as non-POIs, and still get high-levels of accuracy. <br>

To combat this problem, as well as have a more generalized model, I utilized the sklearn `GridSearchCV` (cross-validation) method. This enabled me to iterate through parameters, as well as set my default scoring function to F1 Scores so that precision and recall could both be weighed heavier in the training process. The final Gradient Boosting algorithm was determined by tuning the `n_estimators`, `subsample`, and `max_depth` parameters. <br>

Prior to training the data, it was also split into testing/training samples in order to reduce bias (a classic mistake in machine learning - especially when using gradient descent - is overfitting the data). The data was trained on the training set, and tested on the test (validation) set. This was further enforced by using cross-validation. <br>

*https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html*

### Final Algorithm Evaluation
 
The final algorithm produced decent results considering the size of the dataset. When iteratively tested/shuffled 1000 times using **tester.py**, it achieved the following results:

Accuracy (Total Correct / Overall Total): **0.91840** <br>
Precision (True Positives / Actual Results): **0.71532** <br> 
Recall (True Positives / Predicted Results): **0.64450** <br> 
F1 (Harmonic mean of Precision and Recall): **0.67806**

For a better idea of what this looks like, of the **15000** total predictions, there were: <br>

True positives (Total correctly predicted as a POI): **1289** <br>
False positives (Total incorrectly predicted as a POI): **513** <br>
False negatives (Total incorrectly predicted as a non-POI): **711** <br>
True negatives (Total correctly predicted as a non-POI): **12487**
