Model Details
--------------------

Objective:
The objective of the prediction task is to classify whether an individual's annual income exceeds $50,000. For this purpose, we employed a Random Forest Classifier from the scikit-learn library version 0.24.0, utilizing optimized hyperparameters.

Model Information:
- Model: Random Forest Classifier
- Library: scikit-learn 0.24.0

Model File:
The trained model has been saved as a pickle file named "model.pkl".

Intended Use:
--------------------
The intended application of this model is to predict the income level of an individual using a limited set of attributes. It is primarily designed for students, academics, or research purposes.

Training Data:
--------------------
The Census Income Dataset was obtained from the UCI Machine Learning Repository as a CSV file. You can access the dataset at the following link: [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income). The original dataset consists of 32,561 rows and 15 columns. It includes a target label called "salary," 8 categorical features, and 6 numerical features. For detailed information about each feature, please refer to the UCI link provided.

The target label, "salary," has two classes ('<=50K' and '>50K'), and there is a class imbalance issue with a ratio of approximately 75% for the '<=50K' class and 25% for the '>50K' class.

To ensure data quality, a simple data cleansing process was performed on the original dataset. This involved removing leading and trailing whitespaces from the data. For more insights into the data exploration and cleansing steps, please refer to the "data_cleaning.py" file.

The dataset was split into a training set and a test set using an 80-20 split. Stratification was applied based on the target label, "salary," to maintain the class distribution in both sets.

Evaluation Data:
--------------------
A portion of 20% of the dataset was reserved specifically for evaluating the model's performance.

To prepare the categorical features and the target label for training, transformation techniques were applied. The One Hot Encoder was fitted on the training set to encode the categorical features, while the label binarizer was used to transform the target label. These transformations ensure compatibility with the machine learning model.

Metrics:
--------------------
The classification performance of the model is assessed using metrics such as precision, recall, and F-beta score. These metrics provide insights into the model's accuracy, completeness, and overall performance in predicting the target labels.

Additionally, the confusion matrix is calculated to further analyze the model's performance. The confusion matrix provides a tabular representation of predicted labels versus actual labels, enabling a more detailed examination of true positives, true negatives, false positives, and false negatives.

- Precision: 0.7455
- Recall: 0.633
- F-beta Score: 0.685

Ethical Considerations:
--------------------
It is important to note that the dataset should not be regarded as a fair representation of the salary distribution. Therefore, it is not appropriate to utilize this dataset to make assumptions about the salary levels of specific population categories.

Caveats and Recommendations:
--------------------
The dataset was extracted from the 1994 Census database. However, it is crucial to acknowledge that this dataset is an outdated sample and should not be considered as a reliable statistical representation of the population. It is recommended to utilize this dataset primarily for training purposes in machine learning classification or similar problems.