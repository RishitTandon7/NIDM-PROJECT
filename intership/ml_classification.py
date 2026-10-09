import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def prepare_data(features, labels):
    """
    Prepare data for training by removing NaNs and aligning features with labels.
    
    Args:
        features (numpy.ndarray): Flattened feature matrix.
        labels (numpy.ndarray): Flattened label array (same length as features).
        
    Returns:
        X (numpy.ndarray): Cleaned feature matrix.
        y (numpy.ndarray): Cleaned labels.
    """
    # Create a mask for valid data (no NaNs in features and valid label)
    # Assuming labels are 0, 1, 2. If label is masked/nodata, it might be -1 or similar.
    # We'll assume valid labels are >= 0.
    
    valid_mask = ~np.isnan(features).any(axis=1) & (labels >= 0)
    
    X = features[valid_mask]
    y = labels[valid_mask]
    
    return X, y

def train_model(X_train, y_train, model_type='rf'):
    """
    Train a specified model.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        model_type (str): 'rf' (Random Forest), 'svm' (Support Vector Machine), 'dt' (Decision Tree).
        
    Returns:
        model: Trained model object.
    """
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'svm':
        # SVM can be slow on large datasets, consider using LinearSVC or SGDClassifier for very large data
        clf = SVC(kernel='rbf', random_state=42)
    elif model_type == 'dt':
        clf = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        class_names (list): List of class names for the report.
        
    Returns:
        dict: Dictionary containing accuracy and report.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': conf_matrix
    }

def compare_models(X, y, model_types=['rf', 'svm', 'dt']):
    """
    Train and evaluate multiple models.
    
    Args:
        X: Features.
        y: Labels.
        model_types: List of model types to compare.
        
    Returns:
        pd.DataFrame: Comparison results.
    """
    # Split data once for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data for SVM (and generally good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    for m_type in model_types:
        print(f"Training {m_type}...")
        
        # SVM needs scaled data, trees don't strictly need it but it doesn't hurt much
        # For strict correctness, we use scaled for all or handle inside loop
        if m_type == 'svm':
            clf = train_model(X_train_scaled, y_train, m_type)
            metrics = evaluate_model(clf, X_test_scaled, y_test)
        else:
            clf = train_model(X_train, y_train, m_type)
            metrics = evaluate_model(clf, X_test, y_test)
            
        results.append({
            'Model': m_type.upper(),
            'Accuracy': metrics['accuracy'],
            'Macro F1': metrics['report']['macro avg']['f1-score'],
            'Weighted F1': metrics['report']['weighted avg']['f1-score']
        })
        
    return pd.DataFrame(results)





