import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def prepare_features(slope, aspect, curvature, twi, spi, distance, elevation=None, plan_curvature=None):
    """Prepare feature matrix from various indices."""
    feature_list = [slope.flatten(), aspect.flatten(), curvature.flatten(),
                    twi.flatten(), spi.flatten(), distance.flatten()]
    
    if elevation is not None:
        feature_list.append(elevation.flatten())
    if plan_curvature is not None:
        feature_list.append(plan_curvature.flatten())
    
    # Stack features into a 2D array
    features = np.stack(feature_list, axis=1)
    # Remove NaN values
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    return features, valid_mask

def prepare_labels(glacier_mask, valid_mask):
    """Prepare labels from glacier mask."""
    labels = glacier_mask.flatten()[valid_mask]
    labels = labels.astype(int)
    return labels

def train_decision_tree(features, labels):
    """Train Decision Tree Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_clf = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_clf.predict(X_test)
    
    # Evaluation
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # ROC AUC if binary classification
    if len(np.unique(labels)) == 2:
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    
    # Visualize the tree
    plt.figure(figsize=(20,10))
    feature_names = ['slope', 'aspect', 'curvature', 'twi', 'spi', 'distance', 'elevation', 'plan_curvature'][:features.shape[1]]
    tree.plot_tree(best_clf, filled=True, feature_names=feature_names, max_depth=3)
    plt.show()
    
    return best_clf

def train_random_forest(features, labels):
    """Train Random Forest Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Random Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return clf

def train_svm(features, labels):
    """Train SVM Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    
    print("SVM Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return clf, scaler

def cross_validate_model(features, labels, model='dt'):
    """Perform cross-validation."""
    if model == 'dt':
        clf = DecisionTreeClassifier(random_state=42)
    elif model == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = SVC(random_state=42)
    
    scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    return scores

def predict_glacier_probability(model, features):
    """Predict glacier probability."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[:, 1]
    else:
        # For SVM without probability
        proba = model.decision_function(features)
        proba = 1 / (1 + np.exp(-proba))  # Sigmoid
    return proba

# Complete the truncated section assuming features and labels are prepared
# Assuming features and labels are defined elsewhere
# clf = train_decision_tree(features, labels)




