import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool

# Better to use kwargs, since I dont really know what/how many args catboost can take. The docs from catboost is way too simplified 
# and there is no valuable resources avaliable for a complete list as far as I found
def catboost_fit(data, target, **kwargs):
    X = data.drop(target, axis=1).values
    y = data[target].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    # very strong setup but it is fast
    cb_params = {'iterations': 40000,
                'learning_rate': 0.01,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bagging_temperature': 0,
                'border_count': 128,
                'loss_function': 'Logloss',
                'random_seed': 42,
                'verbose': 500,
                'task_type': 'CPU',
                'leaf_estimation_backtracking': 'AnyImprovement',
                'bootstrap_type': 'Bayesian',
                'use_best_model': True,
                }
    cb_params.update(kwargs)
    model = CatBoostClassifier(**cb_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    y_pred = model.predict(X_val)
    print("\n\nAccuracy Score: ", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    model.save_model("catboost_model/catboost_model.cbm")
# Simple gridsearch
def grid_search_catboost(data, target, **kwargs):
    X = data.drop(target, axis=1).values
    y = data[target].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    model = CatBoostClassifier(**kwargs)

    param_grid = {'learning_rate': [0.01, 0.05, 0.1],
                  'depth': [4, 6, 8],
                  'l2_leaf_reg': [1, 3, 5]}

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_macro', cv=5, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("Best parameters found: ", best_params)

    best_cb = grid.best_estimator_
    y_pred_best = best_cb.predict(X_val)
    print("\n\nAccuracy Score (After Grid Search): ", accuracy_score(y_val, y_pred_best))
    print("\nClassification Report (After Grid Search):\n", classification_report(y_val, y_pred_best))
    best_cb.save_model("catboost_model/catboost_best_model.cbm")
#Predict Fucntion
def catboost_predict(data, target, test, **kwargs):
    X = data.drop(target, axis=1).values
    y = data[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    model = CatBoostClassifier()
    model.load_model("catboost_model/catboost_best_model.cbm")

    y_pred = model.predict(X_test)
    print("\n\nAccuracy Score: ", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main(data_location, target, action, **kwargs):
    col_names = [target]
    for i in range(1, 45):
        col_names.append('F' + str(i))
    data = pd.read_csv(data_location, names=col_names, header=None)

    if action == "train":
        catboost_fit(data, target=target, **kwargs)
    elif action == "search":
        grid_search_catboost(data, target=target, **kwargs)
    elif action == "predict":
        test_location = input("Please input the test set location!")
        test = pd.read_csv(test_location, names=col_names, header=None)
        catboost_predict(data, target=target, test=test, **kwargs)
    else:
        print("Invalid action. Please choose either 'train', 'search' or 'predict'.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python final_cat.py <data_location> <target> <action> [<key=value>...]")
    else:
        kwargs = {}
        for arg in sys.argv[4:]:
            key, value = arg.split('=')
            kwargs[key] = value
        main(sys.argv[1], sys.argv[2], sys.argv[3], **kwargs)




