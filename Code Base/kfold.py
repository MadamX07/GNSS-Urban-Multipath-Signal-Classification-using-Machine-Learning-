# -------------------------------
# Step 1: Import Required Libraries
# -------------------------------
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

# -------------------------------
# Step 2: Load CSV Files for LOS and NLOS Data
# -------------------------------
los_file_path = input("Enter the path to your LOS GNSS CSV file: ")
nlos_file_path = input("Enter the path to your NLOS GNSS CSV file: ")

try:
    los_df = pd.read_csv(los_file_path)
    print("LOS data loaded successfully.\n")
except Exception as e:
    print("Error loading LOS file:", e)
    exit()

try:
    nlos_df = pd.read_csv(nlos_file_path)
    print("NLOS data loaded successfully.\n")
except Exception as e:
    print("Error loading NLOS file:", e)
    exit()

# -------------------------------
# Step 3: Preprocessing the  Data and  Features Extraction
# -------------------------------
los_df.columns = los_df.columns.str.strip()
nlos_df.columns = nlos_df.columns.str.strip()

expected_columns = {'Azimuth', 'Elevation', 'SNR'}
if not expected_columns.issubset(los_df.columns) or not expected_columns.issubset(nlos_df.columns):
    print(f"CSV must contain columns: {expected_columns}")
    exit()

los_df['Label'] = 1
nlos_df['Label'] = 0

df = pd.concat([los_df, nlos_df], ignore_index=True)

X = df[['Azimuth', 'Elevation', 'SNR']]
y = df['Label']

X = X.fillna(X.mean(numeric_only=True))
y = y.loc[X.index]

# -------------------------------
# Step 4: K-Fold Cross-Validation
# -------------------------------
k = 5  # number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
svm_model = SVC(kernel='rbf', random_state=42)

# Accuracy across folds
scores = cross_val_score(svm_model, X, y, cv=skf, scoring='accuracy')
print(f"\n K-Fold Cross-Validation (k={k}) Accuracy Scores: {scores}")
print(f" Average Accuracy: {scores.mean():.4f}")

# -------------------------------
# Step 5: Train on One Fold to Evaluate and Save
# -------------------------------
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    break  # Use only the first fold for reporting and confusion matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the final SVM model on the test fold: {accuracy:.4f}")

# -------------------------------
# Step 6: Evaluate Model on One Fold
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, target_names=["NLOS", "LOS"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# -------------------------------
# Step 7: Visualize Confusion Matrix
# -------------------------------
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NLOS", "LOS"], yticklabels=["NLOS", "LOS"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (1st Fold)")
plt.show()

# -------------------------------
# Step 8: Save Outputs to Excel 
# -------------------------------
cm_df = pd.DataFrame(cm, index=["Actual NLOS", "Actual LOS"], columns=["Predicted NLOS", "Predicted LOS"])

with pd.ExcelWriter("svm_model_output.xlsx", engine="openpyxl") as writer:
    report_df.to_excel(writer, sheet_name="Classification_Report")
    cm_df.to_excel(writer, sheet_name="Confusion_Matrix")

print("\nExcel file with results saved as svm_model_output.xlsx")
