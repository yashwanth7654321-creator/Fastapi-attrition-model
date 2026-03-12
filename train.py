import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv(r"C:\Users\candy\Documents\Data_science_TRaining\IBM_HR_attrition\archive\WA_Fn-UseC_-HR-Employee-Attrition.csv")

actual_categorical = [col for col in data.columns if data[col].nunique() < 7 ]
print(actual_categorical)
print("No of columns is : {}".format(len(actual_categorical)))

obj_cols = data.select_dtypes(include = "object").columns.tolist()
print(obj_cols)
print("No of columns is :{}".format(len(obj_cols)))

cat_cols = list(set(obj_cols)|set(actual_categorical))
print(cat_cols)
print("No of categorical columns is :{}".format(len(cat_cols)))

Futile = [ col for col in data.columns if data[col].nunique() == 1]
print(Futile)

norm_df = data.copy()

norm_df[cat_cols] = norm_df[cat_cols].astype('object')

norm_df = norm_df.drop(Futile, axis = 1)

norm_df['MonthlyRate'].skew()

skew_limit = 0.75
skew_vals = norm_df.select_dtypes(include = ["int64", "float64"]).skew()
skew_cols_df = skew_vals.sort_values(ascending = False).to_frame().rename(columns = { 0 : 'Skew'})
skew_cols = skew_cols_df[skew_cols_df['Skew'].abs() > skew_limit ]
#display(skew_cols.style.set_caption("columns_to_be_transformed"))

norm_df[skew_cols.index.values.tolist()] = norm_df[skew_cols.index.values.tolist()].apply(np.log1p)

y = norm_df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0 )
X = norm_df.loc[:, norm_df.columns != 'Attrition']

cat_cols_in_x = [col for col in cat_cols if col in X.columns]
X = pd.get_dummies(X, columns = cat_cols_in_x, drop_first = True, dtype = int)
X.columns = X.columns.str.replace(" ", "_")
X

X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size = 0.75, random_state = 21)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5, random_state=21)

print("Train_shape:", X_train.shape,  y_train.shape)
print("Test_shape:", X_test.shape,  y_test.shape)
print("Validation_shape:", X_val.shape,  y_val.shape)

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_val = pd.DataFrame(scaler.fit_transform(X_val), columns = X_val.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_val), columns = X_test.columns)

param_grid = [
    {
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs"],
        "max_iter": [300, 500]
    },
    {
        "penalty": ["l1"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear"],
        "max_iter": [300, 500]
    },
    {
        "penalty": ["elasticnet"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["saga"],
        "l1_ratio": [0.2, 0.5, 0.8],
        "max_iter": [500]
    }
]

grid = GridSearchCV(
    estimator = LogisticRegression(max_iter = 100),
    param_grid = param_grid,
    cv =5,
    scoring = "roc_auc")

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

bundle = {
    "model" : best_model,
    "scaler": scaler,
    "features": X_train.columns.tolist()
}
joblib.dump(bundle, "model/model_bundle.pkl")
