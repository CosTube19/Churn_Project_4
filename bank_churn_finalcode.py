# BINARY CLASSIFICATION

# Importing Libraries

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

# Setting the numpy random seed

np.random.seed(37)

# Loading Dataset

df = pd.read_csv(r'C:\Users\KAUSTUBH\Python Scripts\train.csv')
print('Dataframe shape: ', df.shape)

# DATA CLEANUP

df.head()

df.describe()

df.columns.tolist()

# Checking for missing values
df.isnull().sum()

df.dtypes

# Dropping irrelevant columns
df.drop(['id', 'CustomerId', 'Surname', 'Gender', 'Age', 'Geography' ], axis=1, inplace=True)

df.head()

df['Exited'] = df['Exited'].astype(float)

# VISUALIZATIONS

sns.set(style="darkgrid")
ax = sns.countplot(x="Exited", data=df, palette=sns.xkcd_palette(["azure", "light red"]))
plt.xlabel('Exited')
plt.ylabel('Count')
# plt.savefig('./plots/Exited_count.png')
plt.show()

sns.set(style="darkgrid")
ax = sns.countplot(x="CreditScore", data=df, palette='viridis')
plt.xlabel('CreditScore')
plt.ylabel('Count')
plt.title('count_plot of credit scores')
# plt.savefig('./plots/CreditScore_count.png')
plt.show()

sns.countplot(x='NumOfProducts', data=df, hue='Exited', palette=sns.xkcd_palette(["azure", "light red"]))
plt.title("NumOfProducts Count Plot")
plt.xlabel('NumOfProducts')
plt.ylabel('Count')
# plt.savefig('./plots/performance_count.png')
plt.show()

######################################################################################################
#############################         FEATURE ENGINEERING            #################################
######################################################################################################
# Correlation Heatmap

fig, ax = plt.subplots(figsize=(15,10))
correlation_matrix = df.corr() * 100
sns.heatmap(correlation_matrix, annot = True, ax=ax)
# plt.savefig('./plots/correlation_heatmap.png')
plt.show()

##### We see that HRLY_RATE and ANNUAL_RATE are highly correlated with correlation of 1, so we can take the ANNUAL_RATE and discard HRLY_RATE

# X = features & y = Target class

X = df.drop(['Exited'], axis=1)
y = df['Exited']

# Normalizing the all the features

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting dataset into training and testing split with 80-20% ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-fold splits

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)


############################################################################################
#############################         MODELLING            #################################
############################################################################################
# ### -----------------------------------------------------------------------------------------------------------------------Logistic Regression

# Building our model with K-fold validation and GridSearch to find the best parameters

# Defining all the parameters
params = {
    'penalty': ['l1','l2'],
    'C': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
}

# Building model
logreg = LogisticRegression(solver='liblinear')

# Parameter estimating using GridSearch
grid = GridSearchCV(logreg, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

# Fitting the model
grid.fit(X_train, y_train)

logreg_grid_val_score = grid.best_score_
print('Best Score:', logreg_grid_val_score)
print('Best Params:', grid.best_params_)
print('Best Estimator:', grid.best_estimator_)

# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)

logreg_grid = grid.best_estimator_
y_pred = logreg_grid.predict(X_test)

# Confusion matrix

pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )

# Calculating metrics

accuracy1 = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

print('Model Accuracy:', accuracy1 * 100)
print('Precision:', precision* 100)
print('Recall:', recall* 100)
print('AUC:', auc_score* 100)
print('Classification Report:\n', classification_report(y_test, y_pred))

# Assuming y_pred is the decision function or predicted probabilities for positive class
y_pred = logreg_grid.decision_function(X_test)  # Replace with your model and test data

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot AUC curve
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.3)  # Fill area under the curve
plt.show()


# ### -----------------------------------------------------------------------------------------------------------------------------------------K-Nearest Neighbor Classifier(KNN)

# Building our model with K-fold validation and GridSearch to find the best parameters

# Defining all the parameters
params = {
    'n_neighbors': [3,5,11,19],
    'weights': ['uniform','distance']
}

# Building model
knn = KNeighborsClassifier()

# Parameter estimating using GridSearch
grid = GridSearchCV(knn, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

# Fitting the model
grid.fit(X_train, y_train)

knn_grid_val_score = grid.best_score_
print('Best Score:', knn_grid_val_score)
print('Best Params:', grid.best_params_)
print('Best Estimator:', grid.best_estimator_)

# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)

knn_grid= grid.best_estimator_
y_pred = knn_grid.predict(X_test)

# Confusion matrix

pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )

# Calculating metrics
accuracy2 = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

print('Model Accuracy:', accuracy2 * 100)
print('Precision:', precision* 100)
print('Recall:', recall* 100)
print('AUC:', auc_score* 100)
print('Classification Report:\n', classification_report(y_test, y_pred))


# ### ------------------------------------------------------------------------------------------------------------------------Gaussian Naive Bayes

# Building our model with K-fold validation and GridSearch to find the best parameters

# No such parameters for Gaussian Naive Bayes
params = {}

# Building model
gb = GaussianNB()

# Parameter estimating using GridSearch
grid = GridSearchCV(gb, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

# Fitting the model
grid.fit(X_train, y_train)

gb_grid_val_score = grid.best_score_
print('Best Score:', gb_grid_val_score)
print('Best Estimator:', grid.best_estimator_)

# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)

gb_grid= grid.best_estimator_
y_pred = gb_grid.predict(X_test)

# Confusion matrix

pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted A", "Predicted T"], index=["Actual A","Actual T"] )

# Calculating metrics

accuracy3 = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

print('Model Accuracy:', accuracy3 * 100)
print('Precision:', precision* 100)
print('Recall:', recall* 100)
print('AUC:', auc_score* 100)
print('Classification Report:\n', classification_report(y_test, y_pred))

# ## RESULTS

score_df = pd.DataFrame(
    [
        ['Logistic Regression', accuracy1, logreg_grid_val_score],
        ['K-Nearest Neighbors', accuracy2, knn_grid_val_score],
        ['Gaussian Naïve Bayes', accuracy3, gb_grid_val_score],
    ],
    columns= ['Model', 'Test Score', 'Validation Score']
)
score_df['Test Score'] = score_df['Test Score']*100
score_df['Validation Score'] = score_df['Validation Score']*100

score_df

fig, ax1 = plt.subplots(figsize=(10, 5))
tidy = score_df.melt(id_vars='Model').rename(columns=str.title)
sns.barplot(x='Model', y='Value', hue='Variable', data=tidy, ax=ax1, palette=sns.xkcd_palette(["azure", "light red"]))
plt.ylim(20, 90)
plt.xticks(rotation=45, horizontalalignment="right")
# plt.savefig('./plots/result.png')
sns.despine(fig)

time_df = pd.DataFrame(
    [
        ['Logistic Regression', 1.2],
        ['K-Nearest Neighbors', 1.0],
        ['Gaussian Naïve Bayes', 0.0034], 
    ],
    columns= ['Model', 'Training Time']
)

fig, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=time_df, x='Model', y='Training Time', palette=sns.color_palette('husl'))
plt.xticks(rotation=45, horizontalalignment="right")
plt.ylabel('Training Time(in mins)')
# plt.savefig('./plots/training_time.png')
sns.despine(fig)