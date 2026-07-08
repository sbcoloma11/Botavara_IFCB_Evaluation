# This script performs model training and evaluation across several data splits and approaches
# It is designed to pick-up where it left off in case it is interrupted
from ml_funcs import load_data, select_features, train_ml_model, get_evals, get_impurity, compute_shap, check_results_directory
import warnings
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('agg') # more efficient since script only saves images and does not display them
warnings.filterwarnings('ignore') # SHAP will sometimes display warning messages for large datasets

approaches = ['random_forest']
oversamplings = ['ros', 'smote']
tree_grid = {'max_depth' : [10, 20, None], 'ccp_alpha' : [0., 0.01]}
forest_grid = {'n_estimators' : [50, 100, 150], 'max_depth' : [10, 20, None], 'ccp_alpha' : [0., 0.01]}
boost_grid = {'learning_rate' : [0.01, 0.1], 'n_estimators' : [100, 125]}
grids = [tree_grid, forest_grid, boost_grid]

#source_dir = '../data/combined_dataset/'
features_file = '../data/open_source_v2/features.csv'
index_file = '../indices/indices.csv'

if __name__ == '__main__':
    ml_split = 0
    with open('finished_splits/ml_finished_splits.txt') as f:
        ml_split = f.readline()
    while int(ml_split) < 5:
        ml_split = int(ml_split) + 1
        check_results_directory(ml_split)
        print(f'Running split {ml_split}............................................')
        X_train, y_train, X_test, y_test = load_data(features_file, index_file, ml_split) 
        classes = np.unique(y_train)
        # Allows for selected features to be loaded in case feature selection has been finished already
        if 'selected_features.txt' in os.listdir(f'../results/selected_features/split_{ml_split}'):
            with open(f'../results/selected_features/split_{ml_split}/selected_features.txt') as f:
                features = f.readline().split(' ')
            X_train, X_test = X_train[features], X_test[features]
        else:
            X_train, X_test = select_features(X_train, y_train, X_test, ml_split) 
        for oversampling in oversamplings:
            for approach, grid in zip(approaches, grids):
                # Allows for trained model to be loaded in case model training has been finished already
                if f'{approach}_{oversampling}.pkl' in os.listdir(f'../results/models/split_{ml_split}'):
                    model = pickle.load(open(f'../results/models/split_{ml_split}/{approach}_{oversampling}.pkl', 'rb'))
                else:
                    model = train_ml_model(approach, oversampling, grid, X_train, y_train, ml_split)
                # Evaluation and feature importance are skipped when they have finished already
                if f'{approach}_{oversampling}.png' not in os.listdir(f'../results/confusion_matrices/split_{ml_split}'):
                    get_evals(model, approach, oversampling, X_test, y_test, ml_split)
                if f'{approach}_{oversampling}.png' not in os.listdir(f'../results/impurity_importances/split_{ml_split}'):
                    get_impurity(model, approach, oversampling, X_test, ml_split)
                if f'{approach}_{oversampling}.png' not in os.listdir(f'../results/shap_plots/split_{ml_split}'):
                    compute_shap(model, approach, oversampling, X_test, ml_split, classes)
        # Finished splits are tracked and updated on an external text file
        with open('finished_splits/ml_finished_splits.txt', 'w') as f:
            f.writelines(str(ml_split))
        print(f'Split {ml_split} finished............................................')