import sys


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import cvxopt
import random
import warnings
warnings.filterwarnings('ignore')




class SVM:
    def __init__(self, kernel='linear', C=1.0, normalize=True, balance_data=True, max_iter=10000, tol=1e-5, gamma = 0.1):
        self.kernel = kernel
        self.C = C
        self.normalize = normalize
        self.balance_data = balance_data
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma

    def _linear_kernel(self, x, y):
        return np.dot(x, y)

    def _quadratic_kernel(self, x, y):
        return (np.dot(x, y) + 1) ** 2

    def _rbf_kernel(self, x, y, gamma = 0.1):
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

    def _kernel_function(self, x, y):
        if self.kernel == 'linear':
            return self._linear_kernel(x, y)
        elif self.kernel == 'quadratic':
            return self._quadratic_kernel(x, y)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(x, y, gamma=self.gamma)
        else:
            raise ValueError("Invalid kernel choice")
# Could've use SMOTE but it should be better to do it my own, hopefully it can work 
# ideally(even though I tested a lot). However, this data is balanced(40:40) in training 
# set which means it needs no balaancing, but the test set is fairly imbalanced, so I 
# implemented accuracy metrics calculations after so that it can cover that gap.
    def _normalize(self, X):
        return (X - self.mean) / self.std

    def _balance_classes(self, X, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)

        new_X, new_y = [], []

        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            np.random.shuffle(class_indices)
            repeat_count = max_count // counts[cls]
            remaining = max_count % counts[cls]

            repeated_indices = np.repeat(class_indices, repeat_count)
            remaining_indices = class_indices[:remaining]

            indices = np.concatenate([repeated_indices, remaining_indices])
            new_X.append(X[indices])
            new_y.append(y[indices])

        return np.vstack(new_X), np.hstack(new_y)
# fit method:
    def fit(self, X, y):
        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = self._normalize(X)

        if self.balance_data:
            X, y = self._balance_classes(X, y)

        n_samples, n_features = X.shape

        y[y == 0] = -1
        y = y.astype(np.float64)

        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])

        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        A = y.reshape(1, -1)
        b = np.zeros(1)

        G = np.vstack((np.eye(n_samples) * -1, np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b) #needs simplify?
        # Solve the quadratic problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Get the Lagrange multipliers
        alphas = np.ravel(solution['x'])

        # Support vectors have non-zero Lagrange multipliers
        sv_indices = alphas > self.tol
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]

        # Calculate the bias term
        self.b = np.mean(
            [y[i] - np.sum(self.alphas * self.support_vector_labels * K[sv_indices, i]) for i in range(n_samples) if
            sv_indices[i]])
        return self # Might need this

    def predict(self, X):
        if self.normalize:
            X = self._normalize(X)

        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            y_pred[i] = np.sum(
                [self.alphas[j] * self.support_vector_labels[j] * self._kernel_function(X[i], self.support_vectors[j])
                for j in range(len(self.support_vectors))]) + self.b

        y_pred = np.sign(y_pred).astype(int)
        y_pred[y_pred == -1] = 0 #make it back to normal

        return y_pred



# Calculating the accuracy scores:
def calculate_performance_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return [float(precision), float(recall), float(f1), float(auc)] #forbidding warnings and other outputs Used warning packges as well to mute it


# Graphing the accuracy aspects:
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()



def SVM_fit(df, target, fold = 10, hold=0.2, C=1.0, kernel = 'rbf',normalize=True, balance_data=False, graph_op = False, tol = 1e-5, iter = 1e10, gamma = 0.1, report = True):
    #Cleaning Inputs from IO
    fold = int(fold)
    hold = float(hold)
    C = float(C)
    iter = int(iter)
    if kernel == 'rbf':
        gamma = float(gamma)
    tol = float(tol)
    normalize = True if (normalize == "True" or normalize == True) else False
    balance_data = True if (balance_data == "True"  or balance_data == True) else False
    graph_op = True if graph_op == "True" else False
    report = True if report == "True" else False

    X = df.drop(target, axis=1).values
    y = df[target].values
    
    random.seed(314)

    # Calculating the number of rows to allocate for the testing set
    test_size = int(len(X) * hold)
    # Checking dimensionality
    if report == True:
        print("Shape of the data: ", X.shape)
        print("Shape of the label: ", y.shape)
    # Shuffling index list for all the rows in the data
    index_list = list(range(len(X)))
    folds = []
    
    for i in range(fold):
        bundle = []
        random.shuffle(index_list)

        # Split the index list into training and testing indices
        train_indices = index_list[test_size:]
        test_indices = index_list[:test_size]
        
        # Create the training and testing sets 
        X_train = np.array([X[i] for i in train_indices])
        X_test = np.array([X[i] for i in test_indices])
        y_train = np.array([y[i] for i in train_indices])
        y_test = np.array([y[i] for i in test_indices])
        bundle = [X_train, X_test, y_train, y_test]  #IMPORTANT!!!
        folds.append(bundle)

    acc_list = []

    svm = None
    for i in range(fold):
        # Create a SVM class
        svm = SVM(C=C, kernel=kernel,normalize=normalize, balance_data=balance_data, tol = tol, max_iter = iter, gamma = gamma)
        # Fit the model
        svm = svm.fit(folds[i][0], folds[i][2])
        # Make predictions on the test set
        y_pred = svm.predict(folds[i][1])

        # Calculate the accuracy
        accuracy = np.mean(folds[i][3] == y_pred)

        acc = calculate_performance_metrics(folds[i][3], y_pred)
        acc.append(accuracy)
        if report == True:
            print('At Fold #', i+1, '-> Accuracy: ', acc[4], 'precision: ', acc[0],'recall: ', acc[1],'f1: ', acc[2],'auc: ', acc[3] )
        acc_list.append(acc)
        if graph_op == True:    
            plot_confusion_matrix(folds[i][3], y_pred)
            plot_roc_curve(folds[i][3], y_pred)
            
    averages = [sum(column) / len(column) for column in zip(*acc_list)]
    if kernel == 'rbf':
        print('The Performance of this model after ', fold, 'folds. \n Scores are:  \n Accuracy: ', averages[4], 'precision: ', averages[0],'recall: ', \
            averages[1],'f1: ', averages[2],'auc: ', averages[3], '\n', 'Setup: C = ', C, ', kernel used:', kernel, ', gamma = ', gamma)
    else:
        print('The Performance of this model after ', fold, 'folds. \n Scores are:  \n Accuracy: ', averages[4], 'precision: ', averages[0],'recall: ', \
            averages[1],'f1: ', averages[2],'auc: ', averages[3], '\n', 'Setup: C = ', C, ', kernel used:', kernel)
    return (averages)
        

def grid_search_svm(df, target, kernel='rbf', fold=10, hold=0.2, normalize=True, balance_data=False, graph_op=False, tol=1e-5, iter=1e10, strong=False):
    fold = int(fold)
    hold = float(hold)

    iter = int(iter)

    tol = float(tol) 
    normalize = True if (normalize == "True" or normalize == True) else False
    balance_data = True if (balance_data == "True"  or balance_data == True) else False
    graph_op = True if graph_op == "True" else False

    if strong == "True":
        C_values = np.arange(0.1, 100.1, 0.1)
        gamma_values = [None] if kernel != 'rbf' else np.arange(0.0001, 10.0001, 0.0001)
    else:
        C_values_small = np.arange(0.1, 1, 0.1)
        C_values_big = np.arange(1, 100.1, 1)
        C_values = np.append(C_values_small, C_values_big)
        gamma_values_small = [None] if kernel != 'rbf' else np.arange(0.1, 1, 0.1)
        gamma_values_big = [None] if kernel != 'rbf' else np.arange(1, 11, 1)
        gamma_values = np.append(gamma_values_small, gamma_values_big)
    best_metrics = {
        'accuracy': (None, 0),
        'precision': (None, 0),
        'recall': (None, 0),
        'f1': (None, 0),
        'auc': (None, 0)
    }

    for c_try in C_values:
        for gamma in gamma_values:
            averages = SVM_fit(df, target, fold=fold, hold=hold, C=c_try, kernel=kernel, normalize=normalize,
                                  balance_data=balance_data, graph_op=graph_op, tol=tol, iter=iter, gamma=gamma, report=False)


            for metric_idx, metric_name in enumerate(('precision', 'recall', 'f1', 'auc', 'accuracy')):
                if averages[metric_idx] > best_metrics[metric_name][1]:
                    best_metrics[metric_name] = ((c_try, gamma) if kernel == 'rbf' else (c_try,), averages[metric_idx])
    print(best_metrics)
    return best_metrics

def finalize (df, target, test, C=1.0, gamma=0.1, kernel = 'rbf', normalize=True, balance_data = False, tol = 1e-5, iter = 1e10, save = False):
    normalize = True if (normalize == "True" or normalize == True) else False
    balance_data = True if (balance_data == "True"  or balance_data == True) else False
    save = True if save == "True" else False
    C = float(C)
    iter = int(iter)
    if kernel == 'rbf':
        gamma = float(gamma)
    tol = float(tol)


    X = df.drop(target, axis=1).values
    y = df[target].values
    # Create an instance of the SVM class
    svm = SVM(C=C, kernel=kernel,normalize=normalize, balance_data=balance_data, tol = tol, max_iter = iter, gamma = gamma)
    svm = svm.fit(X, y)

    # Make predictions on the input test set
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    y_pred = svm.predict(X_test)

    # Calculate the accuracy
    accuracy = np.mean(y_test == y_pred)

    acc = calculate_performance_metrics(y_test, y_pred)
    acc.append(accuracy)
    print( '-> Accuracy: ', acc[4], 'precision: ', acc[0],'recall: ', acc[1],'f1: ', acc[2],'auc: ', acc[3] )
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)
    if save:
        test['prediction'] = y_pred
        test.to_csv('predictions.csv', index=False)
    return svm





def main(data_location, target, action, kernel, **kwargs):

    col_names = [target]
    for i in range(1,45):
        col_names.append('F'+str(i))
    data = pd.read_csv(data_location, names=col_names, header=None)



    if action == "train":
        SVM_fit(data, target=target, kernel=kernel, **kwargs)
    elif action == "search":
        grid_search_svm(data, target=target, kernel=kernel, **kwargs)
    
    elif action == "predict":
        test_location = input("Please input the test set location!")
        test = pd.read_csv(test_location, names=col_names, header=None)
        finalize(data, target=target, test = test, **kwargs)
        
    else:
        print("Invalid action. Please choose either 'train', 'search' or 'predict'.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python svm_script.py <data_location> <target> <action> <kernel> [<key=value>...]")
    else:
        kwargs = {}
        for arg in sys.argv[5:]:
            key, value = arg.split('=')
            kwargs[key] = value
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], **kwargs)


