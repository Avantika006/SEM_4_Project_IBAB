#!/usr/bin/env python
# coding: utf-8

from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact


##-- 2way classifier using basic medical screening dataset: ADHD only v/s ADHD+ASD --##

# Loading the different datasets
df = pd.read_csv("basic_medical_screening-2023-07-21.csv")

both = 0
asd = 0
adhd = 0
none = 0
# Create a new column 'prediction' based on ASD and ADHD statuses
prediction = []
for i in range(len(df)):
    asd_status = str(df['asd'][i]).lower() in ['true', 'true.']
    adhd_status = not pd.isna(df['behav_adhd'][i])

    if asd_status and adhd_status:
        prediction.append(1)  # Both ASD and ADHD
        both = both + 1
    elif asd_status:
        prediction.append(2)  # ASD only
        asd = asd + 1
    elif adhd_status:
        prediction.append(0)  # ADHD only
        adhd = adhd + 1
    else:
        prediction.append(3)  # Neither ASD nor ADHD
        none = none + 1

df['prediction'] = prediction

# Filter rows
df = df[df['prediction'] != 3].copy()
df = df[df['prediction'] != 2].copy()

print("The number of individuals affected are:")
print("ASD only: ", asd)
print("ADHD only: ", adhd)
print("Both: ", both)
print("Neither: ", none)
all = 0
all = asd+adhd+both+none
print("Sum: ", all)

#Implementing class-balance
# Separate the dataset into ADHD-affected and unaffected individuals
adhd_only = df[df['prediction'] == 0]
asd_adhd = df[df['prediction'] == 1]

# Randomly sample an equal number of rows from the adhd only individuals subset
asd_adhd_sampled = asd_adhd.sample(n=adhd, random_state=42)

# Concatenate the ASD-affected subset and the randomly sampled unaffected individuals subset
merged_df_1 = pd.concat([adhd_only, asd_adhd_sampled])

# Shuffle the rows of the final dataset
merged_df = merged_df_1.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop irrelevant columns
features = merged_df[[
    'sex', 'gest_age', 'eating_probs', 'feeding_dx',
    'med_cond_birth', 'birth_oth_calc',
    'med_cond_birth_def',
    'med_cond_growth', 'growth_oth_calc',
    'med_cond_neuro', 'med_cond_visaud',
]]

# Resetting indices after dropping irrelevant columns
features.reset_index(drop=True, inplace=True)

# Replace null values with 0 in all columns of features
features = features.replace(np.float64('nan'), 0)
features["gest_age"] = features["gest_age"].replace(0,40)   #Replacing null values of gest age with 40

# Categorical columns
cat_col = ['sex', 'eating_probs', 'feeding_dx',
    'med_cond_birth', 'birth_oth_calc',
    'med_cond_birth_def',
    'med_cond_growth', 'growth_oth_calc',
    'med_cond_neuro', 'med_cond_visaud',
    ]
print('Categorical columns :',cat_col)

# Numerical columns
num_col = ['gest_age']
print('Numerical columns :',num_col)

for col in cat_col:
    features[col] = features[col].astype(str)

for col in num_col:
    features[col] = features[col].astype(float)

#################################################################
###-----MODEL-----###\

# Separate features (X) and target variable (y)
X = features
y = merged_df['prediction']

print("Sample size = %d, Features = %d" %  (len(X), X.shape[1]))

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_col),
        ('cat', categorical_transformer, cat_col)])

# Encode categorical columns (assuming 'sex' is categorical)
le = LabelEncoder()
features['sex'] = le.fit_transform(features['sex'])

# Calculate correlation matrix
corr_matrix = features.corr()

# Create a new figure for the heatmap
plt.figure(figsize=(10, 6))

# Generate the heatmap using Seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  # Adjust cmap for color scheme

# Customize the heatmap
plt.title('Correlation Heatmap for Features')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal

# Display the heatmap
plt.show()

#mod = XGBClassifier()
#mod = RandomForestClassifier(n_estimators=10)
#mod = LogisticRegression(max_iter=5000)
mod = tree.DecisionTreeClassifier()

accuracy = []
sensitivity = []
specificity = []
f1 = []
precision = []
roc_auc_scores = []
all_fpr = []
all_tpr = []
all_auc = []
all_recall = []
all_precision = []

# K fold cross-validation and metric calculation
print("Performing 10 fold cross-validation and printing metrics for each fold...")
print("\n")
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterate over folds
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print("-----------------------", fold, "------------------------")
    print('CROSS VALIDATION NO-> ', fold)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fitting and transforming the training set
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transforming the testing set
    X_test_transformed = preprocessor.transform(X_test)

    mod.fit(X_train_transformed, y_train)

    y_pred = mod.predict(X_test_transformed)

    # Calculating the accuracy_score for each fold
    accuracy.append(accuracy_score(y_test, y_pred))

    # Calculating the sensitivity and specificity
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Adjust indices for 4 classes (0, 1, 2, 3)
    TP = conf_matrix[0, 0].sum()  # True Positives across all classes
    FN = conf_matrix[1:, 0].sum()  # False Positives
    FP = conf_matrix[0, 1:].sum()  # False Negatives
    TN = conf_matrix[1:, 1:].sum()  # True Negatives

    sens = TP / (TP + FN)
    sensitivity.append(sens)
    spec = TN / (TN + FP)
    specificity.append(spec)

    # Calculating the F1 score
    f1.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculating the Precision
    precision.append(precision_score(y_test, y_pred, average='weighted'))

    # Calculate ROC-AUC score
    # Get the predicted probabilities for the positive class
    y_pred_prob = mod.predict_proba(X_test_transformed)[:, 1]

    # Calculate ROC-AUC score using predicted probabilities
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    roc_auc_scores.append(roc_auc)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Append fpr, tpr, and auc to the lists
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

    # Calculate precision and recall for each fold
    precision_all, recall_all, _ = precision_recall_curve(y_test, y_pred_prob)

    # Store precision and recall for all folds
    all_precision.append(precision_all)
    all_recall.append(recall_all)

    ##TABLE PRINTING
    metrics_data = [
        ("Accuracy", accuracy[fold-1]),
        ("Sensitivity", sensitivity[fold-1]),
        ("Specificity", specificity[fold-1]),
        ("F1 score", f1[fold-1]),
        ("Precision", precision[fold-1]),
        ("ROC-AUC Score", roc_auc_scores[fold-1])
    ]

    # Print the table
    table_headers = ["Metric", "Value"]
    table = tabulate(metrics_data, headers=table_headers, tablefmt="grid")

    print(table)
    print("----------------------------------------------------------------------------------------")


##Plotting heatmap for confusion matrix of last iteration
sns.heatmap(conf_matrix,
            annot=True,
            fmt='g',
            xticklabels=['AuDHD','ADHD'],
            yticklabels=['AuDHD','ADHD'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

#Statistical analysis of the Confusion matrix of last iteration
# Perform the chi-squared test
chi2, p_value, _, _ = chi2_contingency(conf_matrix)
# Print the results of the chi-squared test
print("Chi-Squared Test:")
print("Chi-Squared Statistic:", chi2)
print("P-value:", p_value)

# Perform Fisher's exact test
odds_ratio, p_value = fisher_exact(conf_matrix)
# Print the results of Fisher's exact test
print("\nFisher's Exact Test:")
print("Odds Ratio:", odds_ratio)
print("P-value:", p_value)


##Calculation of mean and standard deviation of various metrics

# Accuracy
m_accuracy = np.mean(accuracy)
std_accuracy = std(accuracy)

# Sensitivity
m_sensitivity = mean(sensitivity)
std_sensitivity = std(sensitivity)

# Specificity
m_specificity = mean(specificity)
std_specificity = std(specificity)

# F1 score
m_f1 = mean(f1)
std_f1 = std(f1)

# Precision
m_precision = mean(precision)
std_precision = std(precision)

mean_roc_auc = np.mean(roc_auc_scores)
std_roc_auc = np.std(roc_auc_scores)

##TABLE PRINTING
metrics_data = [
    ("Accuracy", m_accuracy, std_accuracy),
    ("Sensitivity", m_sensitivity, std_sensitivity),
    ("Specificity", m_specificity, std_specificity),
    ("F1 score", m_f1, std_f1),
    ("Precision", m_precision, std_precision),
    ("ROC-AUC Score:", mean_roc_auc,std_roc_auc),
]

# Print the table
table_headers = ["Metric", "Mean", "Standard Deviation"]
table = tabulate(metrics_data, headers=table_headers, tablefmt="grid")

print("MEAN AND STANDARD DEVIATION OF VARIOUS METRICS")
print(table)

# Plotting changes in each metric across the 10 folds
plt.figure(figsize=(10, 6))

# Accuracy
plt.plot(range(1, 11), accuracy, marker='o', label='Accuracy')
# Sensitivity
plt.plot(range(1, 11), sensitivity, marker='o', label='Sensitivity')
# Specificity
plt.plot(range(1, 11), specificity, marker='o', label='Specificity')
# F1 score
plt.plot(range(1, 11), f1, marker='o', label='F1 Score')
# Precision
plt.plot(range(1, 11), precision, marker='o', label='Precision')
# ROC-AUC Score
plt.plot(range(1, 11), roc_auc_scores, marker='o', label='ROC-AUC Score')

plt.title('Metric Changes Across Folds')
plt.xlabel('Fold')
plt.ylabel('Metric Value')
plt.xticks(range(1, 11))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot ROC curve for all folds in a single plot
plt.figure()

# Plot ROC curve for each fold
for i in range(len(all_fpr)):
    plt.plot(all_fpr[i], all_tpr[i], lw=2, label='ROC curve (fold {}) (AUC = {:.2f})'.format(i+1, all_auc[i]))

# Plot random guessing line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot precision-recall curves for all folds
plt.figure(figsize=(10, 6))

for i in range(len(all_recall)):
    plt.plot(all_recall[i], all_precision[i], marker='None', label='PR curve (fold {})'.format(i+1))

plt.title('Precision-Recall Curves Across Folds')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()