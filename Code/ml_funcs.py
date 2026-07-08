# This script contains all necessary functions needed for the machine learning pipeline
# Assumes feature extraction and data splitting have been finished
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import shap
import matplotlib
import os
matplotlib.use('agg') # more efficient since script only saves images and does not display them

def check_results_directory(split):
    """Function for recreating the folder structure of the results if not already present"""
    if not os.path.isdir(os.path.join(f'../results')):
        os.mkdir(os.path.join(f'../results'))
    if not os.path.isdir(os.path.join(f'../results/selected_features/')):
        os.mkdir(os.path.join(f'../results/selected_features/'))
    if not os.path.isdir(os.path.join(f'../results/selected_features/split_{split}/')):
        os.mkdir(os.path.join(f'../results/selected_features/split_{split}/'))
    if not os.path.isdir(os.path.join(f'../results/models/')):
        os.mkdir(os.path.join(f'../results/models/'))
    if not os.path.isdir(os.path.join(f'../results/models/split_{split}/')):
        os.mkdir(os.path.join(f'../results/models/split_{split}/'))
    if not os.path.isdir(os.path.join(f'../results/confusion_matrices/')):
        os.mkdir(os.path.join(f'../results/confusion_matrices/'))
    if not os.path.isdir(os.path.join(f'../results/confusion_matrices/split_{split}')):
        os.mkdir(os.path.join(f'../results/confusion_matrices/split_{split}'))
    if not os.path.isdir(os.path.join(f'../results/impurity_importances/')):
        os.mkdir(os.path.join(f'../results/impurity_importances/'))
    if not os.path.isdir(os.path.join(f'../results/impurity_importances/split_{split}')):
        os.mkdir(os.path.join(f'../results/impurity_importances/split_{split}'))
    if not os.path.isdir(os.path.join(f'../results/shap_plots/')):
        os.mkdir(os.path.join(f'../results/shap_plots/'))
    if not os.path.isdir(os.path.join(f'../results/shap_plots/split_{split}')):
        os.mkdir(os.path.join(f'../results/shap_plots/split_{split}'))
    if not os.path.isdir(os.path.join(f'../results/classification_reports/')):
        os.mkdir(os.path.join(f'../results/classification_reports/'))
    if not os.path.isdir(os.path.join(f'../results/classification_reports/split_{split}/')):
        os.mkdir(os.path.join(f'../results/classification_reports/split_{split}/'))
        
def load_data(split):
    """Function for loading the dataset and performing train-test split based on predetermined indices"""
    print(f'Loading dataset for split {split}............................................')
    df = pd.read_csv('../data/features.csv', index_col = 0)
    indices = pd.read_csv('../indices/indices.csv', index_col = 0)
    # Instances within the split are put on the test set
    train_indices = indices[indices['split'] != split]
    test_indices = indices[indices['split'] == split]
    train = pd.merge(df, train_indices, 'inner')
    test = pd.merge(df, test_indices, 'inner')
    X_train = train[[column for column in train.columns if column not in ['filename', 'label', 'split', 'roi_number']]]
    y_train = train['label']
    X_test = test[[column for column in test.columns if column not in ['filename', 'label', 'split', 'roi_number']]]
    y_test = test['label']
    y_train, y_test = y_train.values.reshape(-1), y_test.values.reshape(-1)
    return X_train, y_train, X_test, y_test

def detect_load_data(source_dir, folder):
    """Function for loading the dataset and performing train-test split based on predetermined indices"""
    df = pd.read_csv(os.path.join(source_dir,f"{folder}_features.csv"), index_col = 0)
    df = df.dropna(inplace = False)
    df.to_csv(os.path.join(source_dir,f"{folder}_analyzed_features.csv"))
    X_detect = df[[column for column in df.columns if column not in ['filename', 'label', 'roi_number']]]
    
    return X_detect, df[['source','filename','roi_number']]

def select_features(X_train, y_train, X_test, split):
    """Function for performing feature selection"""
    print('Starting feature selection............................................')
    classes = np.unique(y_train)
    # Class weighting is performed to handle data imbalance
    class_weights = compute_class_weight(class_weight = "balanced", classes = classes, y = y_train)
    sample_weights = np.zeros_like(y_train)
    for i, specie in enumerate(classes):
        specie_indices = y_train == specie
        sample_weights[specie_indices] = class_weights[i]
    # Literature suggests picking 25 features
    selector = RFE(estimator = RandomForestClassifier(random_state = 1), n_features_to_select = 25, step = 0.05)
    selector.fit(X_train, y_train, sample_weight = sample_weights)
    features = selector.feature_names_in_[selector.support_]
    X_train = X_train[features]
    X_test = X_test[features]
    with open(f'../results/selected_features/split_{split}/selected_features.txt', 'w') as f:
        f.writelines(' '.join(list(features)))
    print('Feature selection finished............................................')
    return X_train, X_test

def train_ml_model(approach, oversampling, grid, X_train, y_train, split):
    """Function for model training and hyperparameter tuning"""
    print(f'Starting training of {approach} model using {oversampling} oversampling............................................')
    if approach == 'decision_tree':
        model = DecisionTreeClassifier()
    elif approach == 'random_forest':
        model = RandomForestClassifier()
    elif approach == 'gradient_boosting':
        model = GradientBoostingClassifier()
    else:
        return None
    if oversampling == 'ros':
        oversampler = RandomOverSampler(random_state = 1)
    elif oversampling == 'smote':
        oversampler = SMOTE(random_state = 1)
    else:
        return None
    new_X_train, new_y_train = oversampler.fit_resample(X_train, y_train)
    grid_search = GridSearchCV(model, grid, cv = 5, scoring = 'accuracy', return_train_score = True, n_jobs = -1)
    best = grid_search.fit(new_X_train, new_y_train).best_estimator_
    pickle.dump(best, open(f'../results/models/split_{split}/{approach}_{oversampling}.pkl', 'wb')) 
    print(f'Model saved at results/models/split_{split}/{approach}_{oversampling}.pkl............................................')
    return best

def get_evals(model, approach, oversampling, X_test, y_test, split):
    """Function for getting classification report and confusion matrix"""
    y_pred = model.predict(X_test)
    label_names = model.classes_
    class_report = classification_report(y_test, y_pred, labels = label_names, digits = 4, output_dict = True, zero_division = 0)
    class_report = pd.DataFrame(class_report).transpose()
    class_report.to_csv(f'../results/classification_reports/split_{split}/{approach}_{oversampling}.csv')
    print(f'Classification report saved at results/classification_reports/split_{split}/{approach}_{oversampling}.csv............................................')
    # Figure size and xticks rotation should be adjusted based on number of classes
    # Manual checking of resulting figures is suggested
    _, ax = plt.subplots(figsize = (5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels = label_names, ax = ax, normalize = 'true')
    plt.xticks(rotation = 60)
    plt.savefig(f'../results/confusion_matrices/split_{split}/{approach}_{oversampling}.png', bbox_inches = "tight", dpi = 100)
    print(f'Confusion matrix saved at results/confusion_matrices/split_{split}/{approach}_{oversampling}.png............................................')

def detect_get_evals(model, X_test):
    """Function for getting classification report and confusion matrix"""
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    return y_pred, y_probs

def get_impurity(model, approach, oversampling, X_test, split):
    """Function for getting impurity-based feature importance"""
    impurity_based_imp = model.feature_importances_
    ranking = np.argsort(impurity_based_imp)
    fig = plt.figure()
    fig.suptitle('Impurity-Based Feature Importance')
    pd.Series(impurity_based_imp[ranking], index = X_test.columns[ranking]).plot.barh()
    plt.savefig(f'../results/impurity_importances/split_{split}/{approach}_{oversampling}.png', bbox_inches = "tight", dpi = 100)
    print(f'Impurity importance plot saved at results/impurity_importances/split_{split}/{approach}_{oversampling}.png............................................')

def compute_shap(model, approach, oversampling, X_test, split, classes):
    """Function for getting SHAP values"""
    print(f'Starting SHAP computation............................................')
    fig = plt.figure()
    fig.suptitle('Mean of Absolute SHAP Values', fontsize = 'xx-large', fontweight = 'extra bold', y = 0.9)
    # The amount of instances to sample should be adjusted based on available computational resources
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 50)) 
    shap_values = explainer.shap_values(shap.sample(X_test, 1000), check_additivity = False)
    for i, specie in enumerate(classes):
        mean_shap = np.abs(shap_values[i]).mean(axis = 0)
        arrangement = np.argsort(mean_shap)[::-1]
        mean_shap = mean_shap[arrangement].tolist()
        feature_names = X_test.columns[arrangement].tolist()[:9]
        feature_names.append('Sum of 16 other features')
        mean_shap[9] = sum(mean_shap[9:])
        mean_shap = mean_shap[:10]
        # Number of rows and columns should be adjusted based on number of classes 
        ax = fig.add_subplot(5, 4, i + 1)
        y_pos = np.arange(len(feature_names))
        hbars = ax.barh(y_pos, mean_shap, align='center', color = '#ff0051')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize = 'large')
        ax.invert_yaxis()  
        ax.set_xlabel('mean(|SHAP value|)')
        ax.set_title(specie)
        ax.tick_params(left = False)
        ax.xaxis.label.set_fontsize('large')
        ax.spines[['right', 'top']].set_visible(False)
        ax.bar_label(hbars, fmt=' + %.2f', color = '#ff0051', fontsize = 'large')
    # Figure size and spacing should be adjusted based on the number of classes
    fig.set_size_inches(32, 18)
    plt.subplots_adjust(wspace = 0.7, hspace = 0.35)
    plt.savefig(f'../results/shap_plots/split_{split}/{approach}_{oversampling}.png', bbox_inches = "tight", dpi = 100)
    print(f'SHAP plot saved at results/shap_plots/split_{split}/{approach}_{oversampling}.png............................................')

