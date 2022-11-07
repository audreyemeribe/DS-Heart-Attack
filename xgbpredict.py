import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Boosting model
from xgboost.sklearn import XGBClassifier
import time

# Metrics (Computation)
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score


# Read in the file to be used
heart_df = pd.read_csv("Dataset for DS Challenge.csv")

# Looking at the first five rows of the dataset
heart_df.head()

heart_df = heart_df.drop(columns=['o2Saturation'])

# Drop row if it does not have at least two values that are not NaN
heart_df = heart_df.dropna(thresh=2) 
heart_df.info()

# Set the features 
X = heart_df.drop('output', axis=1)

# Set the target
y = heart_df['output']

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)

# Scale the data
# Define and fit the scaler
scaler = StandardScaler().fit(X_train)

# Transform the train and test data
X_train_ss = scaler.transform(X_train)
X_test_ss = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_ss, columns=X.columns)

X_test = pd.DataFrame(X_test_ss, columns=X.columns)

# Using XGBoostClassifier model
XGB = XGBClassifier(objective='binary:logistic', colsample_bytree=1, learning_rate=0.3, max_depth=1, n_estimators=41)

# fitting the XGBoost model
XGB.fit(X_train, y_train)

# Model Predictions for train and validation set
y_train_pred = XGB.predict(X_train)
y_pred_XGB = XGB.predict(X_test)

print(f"XGB Train score: {XGB.score(X_train, y_train)}")
print(f"XGB Test score: {XGB.score(X_test, y_test)}")

# Test confusion matrix
print("Test:")
print(confusion_matrix(y_test, y_pred_XGB))

# Saving model for deployment
with open("pickled_best_XGB.pkl", "wb") as pickle_out:
    pickle.dump(XGB, pickle_out)
