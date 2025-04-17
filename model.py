from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import argparse
import time
import os



if __name__ == '__main__':
    
    Parameter = argparse.ArgumentParser()
    Parameter.add_argument('--integration_data', help="omics data")
    Parameter.add_argument('--label', help="sample label")
    args = Parameter.parse_args()
    
    ###Load feature matrix and label file###
    fn_omics = pd.read_csv(args.integration_data, index_col = 0).values
    label = pd.read_csv(args.label, header=0, index_col=0)['Label'].values
    
    ###Setting the output directory###
    output = os.getcwd() + '/result'
    os.makedirs(output)
    
    ###Model training and evaluation###

    model_params = {
        "penalty": 'l1',  # elasticnet, l1, l2
        "solver": 'saga',
        "C": 1,
        "max_iter": 100000,
        #"l1_ratio": 0.5,
    }

    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    all_results = []
    start_time = time.time()


    run_results = []
    base_model = LogisticRegression(random_state = 42, **model_params)
    model = OneVsRestClassifier(base_model) 

    for fold, (train_index, test_index) in enumerate(kf.split(fn_omics, label), start = 1):
        X_train, X_test = fn_omics[train_index], fn_omics[test_index]
        y_train, y_test = label[train_index], label[test_index]

     
        model.fit(X_train, y_train)

      
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='weighted')

        run_results.append({
            'Fold': fold,
            'Accuracy': accuracy,
            'F1_weighted': f1_weighted,
            'F1_macro': f1_macro,
            'AUC': auc,
        })

    
    results_df = pd.DataFrame(run_results)  
    results_df.to_csv("./result/L1-performance_results.csv", index=False)

    # 统计运行时间
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds.")
    print("Results saved to 'performance_results.csv' and 'important_features.txt'.")
