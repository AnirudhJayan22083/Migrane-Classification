import os
import numpy as np
import librosa
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import csv

dataset=open('migraine_data.csv','r')
# Initialize lists for features and labels
TempFeatures = []
labels = []

rows=csv.reader(dataset)
for row in rows:
    TempList=row[0:23]
    TempFeatures.append(TempList)
    if row[23]=="Typical aura with migraine":
        labels.append(1)
    elif row[23]=="Migraine without aura":
        labels.append(2)
    elif row[23]=="Typical aura without migraine":
        labels.append(3)
    elif row[23]=="Familial hemiplegic migraine":
        labels.append(4)
    elif row[23]=="Sporadic hemiplegic migraine":
        labels.append(5)
    elif row[23]=="Basilar-type aura":
        labels.append(6)
    elif row[23]=="Other":
        labels.append(7)
print(TempFeatures[1])
print(len(labels))   
features=[]
l=TempFeatures[1:]
for data in l:
    TempData=[]
    for i in data:
        TempData.append(int(i))
    features.append(TempData)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

le=23
s=0

for i in features:
    if len(i)-le!=0:
        s+=1
print(s)

print(features[1])
# Normalize features
print('Normalizing features...')
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)

# Avoid division by zero: if std is 0, leave the feature as is (no normalization)
features_normalized = np.where(std != 0, (features - mean) / std, features)

# Split into training and testing sets (70% train, 30% test)
print('Splitting data into training and testing sets...')
X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels, test_size=0.3, random_state=42)

print(len(X_train), len(X_train[1]))
c=0
for i in X_train:
    if len(i) != 23:  # Check if the number of features is correct
        print(i)
    for element in i:
        if np.isnan(element):
            print("NaN found!")
            c += 1
print(c)

# 1. Support Vector Machine (SVM) Classifier using SVC
print('\nTraining Support Vector Machine (SVM) classifier...')

C_values = [0.1, 1, 10, 100]
svm_results = {}

print(f'\nSVM with rbf kernel:')
best_accuracy = 0
best_model = None
best_C = None

for C in C_values:
    # Train SVM model
    clf = svm.SVC(kernel='rbf', C=C, gamma='scale', decision_function_shape='ovr')
    clf.fit(X_train, y_train)

    # Cross-validation
    cv_accuracy = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"C = {C}, Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%")

    if cv_accuracy > best_accuracy:
        best_accuracy = cv_accuracy
        best_model = clf
        best_C = C


# Predict on test set
y_pred_svm = best_model.predict(X_test)

# Calculate test accuracy
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Best Test Accuracy for rbf SVM: {test_accuracy_svm * 100:.2f}%")

# Print classification report for SVM
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=["Typical aura with migraine", "Migraine without aura", 
                                                           "Typical aura without migraine", "Familial hemiplegic migraine", 
                                                           "Sporadic hemiplegic migraine", "Basilar-type aura", "Other"]))


# 2. Random Forest Classifier
print('\nTraining Random Forest Classifier...')
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate test accuracy
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {test_accuracy_rf * 100:.2f}%")

# Print classification report for Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Typical aura with migraine", "Migraine without aura", 
                                                           "Typical aura without migraine", "Familial hemiplegic migraine", 
                                                           "Sporadic hemiplegic migraine", "Basilar-type aura", "Other"]))

# Confusion matrix for SVM
conf_mat_svm = confusion_matrix(y_test, y_pred_svm)

# Display confusion matrix for SVM
disp = ConfusionMatrixDisplay(conf_mat_svm, display_labels=["Typical aura with migraine", "Migraine without aura", 
                                                           "Typical aura without migraine", "Familial hemiplegic migraine", 
                                                           "Sporadic hemiplegic migraine", "Basilar-type aura", "Other"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SVM (rbf Kernel)')
plt.show()


# Confusion matrix for RandomForest
conf_mat_rf = confusion_matrix(y_test, y_pred_rf)

# Display confusion matrix for Random Forest
disp = ConfusionMatrixDisplay(conf_mat_rf, display_labels=["Typical aura with migraine", "Migraine without aura", 
                                                           "Typical aura without migraine", "Familial hemiplegic migraine", 
                                                           "Sporadic hemiplegic migraine", "Basilar-type aura", "Other"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest')
plt.show()
