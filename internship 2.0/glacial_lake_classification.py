"""
Glacial Lake Classification using Multi-Source Satellite Data
Compares Random Forest, SVM, and Decision Tree classifiers on Landsat 5 and Sentinel-2 data

Author: Generated for NIDM-SRM Drone Project
Date: 2026-01-07
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GlacialLakeClassifier:
    """
    Multi-source satellite data classifier for glacial lake detection
    
    Classes:
    - 0: Lake
    - 1: Glacial Lake (G Lake)
    - 2: Non-Glacial Lake (Non G Lake)
    """
    
    def __init__(self, output_dir="classification_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.class_names = ['Lake', 'Glacial Lake', 'Non-Glacial Lake']
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        
        self.scaler = StandardScaler()
        self.results = {}
        
    def calculate_indices(self, bands_dict):
        """
        Calculate spectral indices for classification
        
        Parameters:
        -----------
        bands_dict: dict
            Dictionary with band names as keys and numpy arrays as values
            
        Returns:
        --------
        features: numpy array
            Feature matrix
        """
        features = []
        
        # Add original bands
        for band_name in sorted(bands_dict.keys()):
            features.append(bands_dict[band_name].flatten())
        
        # Calculate NDWI (Normalized Difference Water Index)
        # For Landsat 5: NDWI = (Green - NIR) / (Green + NIR) = (Band2 - Band4) / (Band2 + Band4)
        # For Sentinel-2: NDWI = (Green - NIR) / (Green + NIR) = (Band3 - Band8) / (Band3 + Band8)
        if 'band_2' in bands_dict and 'band_4' in bands_dict:  # Landsat 5
            green = bands_dict['band_2']
            nir = bands_dict['band_4']
            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = (green - nir) / (green + nir)
                ndwi = np.nan_to_num(ndwi, 0)
            features.append(ndwi.flatten())
            
        elif 'band_3' in bands_dict and 'band_8' in bands_dict:  # Sentinel-2
            green = bands_dict['band_3']
            nir = bands_dict['band_8']
            with np.errstate(divide='ignore', invalid='ignore'):
                ndwi = (green - nir) / (green + nir)
                ndwi = np.nan_to_num(ndwi, 0)
            features.append(ndwi.flatten())
        
        # Calculate NDVI (Normalized Difference Vegetation Index)
        if 'band_3' in bands_dict and 'band_4' in bands_dict:  # Landsat 5
            red = bands_dict['band_3']
            nir = bands_dict['band_4']
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red)
                ndvi = np.nan_to_num(ndvi, 0)
            features.append(ndvi.flatten())
            
        elif 'band_4' in bands_dict and 'band_8' in bands_dict:  # Sentinel-2
            red = bands_dict['band_4']
            nir = bands_dict['band_8']
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red)
                ndvi = np.nan_to_num(ndvi, 0)
            features.append(ndvi.flatten())
        
        # Stack all features
        features = np.column_stack(features)
        
        return features
    
    def load_vrt_data(self, vrt_path):
        """Load data from VRT file"""
        try:
            with rasterio.open(vrt_path) as src:
                bands_dict = {}
                for i in range(1, min(src.count + 1, 11)):  # Read up to 10 bands
                    band_data = src.read(i).astype(float)
                    nodata = src.nodatavals[i-1]
                    if nodata is not None:
                        band_data[band_data == nodata] = 0
                    bands_dict[f'band_{i}'] = band_data
                
                return bands_dict, src.meta
        except Exception as e:
            print(f"Error loading {vrt_path}: {str(e)}")
            return None, None
    
    def generate_synthetic_training_data(self, features, n_samples=10000):
        """
        Generate synthetic training labels based on spectral characteristics
        This is a placeholder - replace with actual ground truth data
        
        Parameters:
        -----------
        features: numpy array
            Feature matrix
        n_samples: int
            Number of samples to generate
            
        Returns:
        --------
        X_train, y_train: Training features and labels
        """
        print("\n[INFO] Generating synthetic training data...")
        print("[WARNING] This uses simulated labels. Replace with actual ground truth data!")
        
        # Randomly sample pixels
        n_pixels = features.shape[0]
        indices = np.random.choice(n_pixels, min(n_samples, n_pixels), replace=False)
        
        X_train = features[indices]
        
        # Simple heuristic for demonstration (REPLACE WITH ACTUAL LABELS!)
        # Assuming last column is NDWI
        ndwi_col = -2 if features.shape[1] > 8 else -1
        ndwi = X_train[:, ndwi_col]
        
        y_train = np.zeros(len(X_train), dtype=int)
        
        # High NDWI = Water/Lake (class 0)
        y_train[ndwi > 0.3] = 0
        
        # Medium-high NDWI + specific spectral signature = Glacial Lake (class 1)
        glacial_mask = (ndwi > 0.2) & (ndwi <= 0.3)
        y_train[glacial_mask] = 1
        
        # Lower NDWI but still water = Non-Glacial Lake (class 2)
        non_glacial_mask = (ndwi > 0.1) & (ndwi <= 0.2)
        y_train[non_glacial_mask] = 2
        
        # Add some randomness to simulate real-world variability
        noise_indices = np.random.choice(len(y_train), len(y_train) // 10, replace=False)
        y_train[noise_indices] = np.random.randint(0, 3, len(noise_indices))
        
        print(f"Training samples: {len(y_train)}")
        print(f"  - Lake: {np.sum(y_train == 0)}")
        print(f"  - Glacial Lake: {np.sum(y_train == 1)}")
        print(f"  - Non-Glacial Lake: {np.sum(y_train == 2)}")
        
        return X_train, y_train
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, dataset_name):
        """
        Train all models and evaluate performance
        
        Parameters:
        -----------
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        dataset_name: Name of the dataset (e.g., 'Landsat 5', 'Sentinel-2')
        """
        print(f"\n{'='*80}")
        print(f"Training and Evaluating Models on {dataset_name}")
        print(f"{'='*80}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n[{model_name}]")
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.results[dataset_name] = results
        
        return results
    
    def plot_comparison(self):
        """Create comprehensive comparison plots"""
        print(f"\nGenerating comparison visualizations...")
        
        # Prepare data for plotting
        datasets = list(self.results.keys())
        models = list(self.models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data
            data = []
            for dataset in datasets:
                for model in models:
                    data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'Score': self.results[dataset][model][metric]
                    })
            
            df = pd.DataFrame(data)
            
            # Create grouped bar plot
            x = np.arange(len(models))
            width = 0.35
            
            for i, dataset in enumerate(datasets):
                dataset_data = df[df['Dataset'] == dataset]
                offset = width * (i - len(datasets)/2 + 0.5)
                ax.bar(x + offset, dataset_data['Score'], width, 
                      label=dataset, alpha=0.8)
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=150)
        print(f"Saved: {os.path.join(self.output_dir, 'model_comparison.png')}")
        plt.show()
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models and datasets"""
        datasets = list(self.results.keys())
        models = list(self.models.keys())
        
        fig, axes = plt.subplots(len(datasets), len(models), 
                                figsize=(5*len(models), 5*len(datasets)))
        
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        if len(models) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Confusion Matrices - All Models and Datasets', 
                    fontsize=16, fontweight='bold')
        
        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                ax = axes[i, j]
                cm = self.results[dataset][model]['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=self.class_names,
                           yticklabels=self.class_names)
                ax.set_title(f'{dataset}\n{model}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=150)
        print(f"Saved: {os.path.join(self.output_dir, 'confusion_matrices.png')}")
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive text report"""
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GLACIAL LAKE CLASSIFICATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("CLASSIFICATION TASK:\n")
            f.write("  - Classes: Lake, Glacial Lake, Non-Glacial Lake\n")
            f.write("  - Datasets: Landsat 5, Sentinel-2\n")
            f.write("  - Models: Random Forest, SVM, Decision Tree\n\n")
            
            for dataset in self.results.keys():
                f.write(f"\n{'-'*80}\n")
                f.write(f"DATASET: {dataset}\n")
                f.write(f"{'-'*80}\n\n")
                
                for model_name, metrics in self.results[dataset].items():
                    f.write(f"\n[{model_name}]\n")
                    f.write(f"  Accuracy:         {metrics['accuracy']:.4f}\n")
                    f.write(f"  Precision:        {metrics['precision']:.4f}\n")
                    f.write(f"  Recall:           {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score:         {metrics['f1_score']:.4f}\n")
                    f.write(f"  CV Score (mean):  {metrics['cv_mean']:.4f}\n")
                    f.write(f"  CV Score (std):   {metrics['cv_std']:.4f}\n")
            
            # Find best performer
            f.write(f"\n\n{'='*80}\n")
            f.write("BEST PERFORMERS\n")
            f.write(f"{'='*80}\n\n")
            
            for metric in ['accuracy', 'f1_score']:
                best_score = 0
                best_combo = None
                
                for dataset in self.results.keys():
                    for model_name, metrics in self.results[dataset].items():
                        score = metrics[metric]
                        if score > best_score:
                            best_score = score
                            best_combo = (dataset, model_name)
                
                f.write(f"Best {metric.upper()}:\n")
                f.write(f"  Dataset: {best_combo[0]}\n")
                f.write(f"  Model: {best_combo[1]}\n")
                f.write(f"  Score: {best_score:.4f}\n\n")
        
        print(f"\nSaved comprehensive report: {report_path}")


def main():
    """Main execution function"""
    print("="*80)
    print("GLACIAL LAKE CLASSIFICATION - MULTI-SOURCE SATELLITE DATA")
    print("="*80)
    
    # Initialize classifier
    classifier = GlacialLakeClassifier(output_dir="classification_results")
    
    # Dataset paths
    datasets = {
        'Landsat 5': r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Landsat 5\L5_VRT",
        'Sentinel-2': r"f:\NIDM-SRM Drone Project Docs\internship 2.0\Sentinel 2\S2_VRT"
    }
    
    # Process each dataset
    for dataset_name, vrt_dir in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} Dataset")
        print(f"{'='*80}")
        
        # Get VRT files (use first available for demonstration)
        vrt_files = sorted(glob.glob(os.path.join(vrt_dir, "*.vrt")))
        
        if not vrt_files:
            print(f"No VRT files found in {vrt_dir}")
            continue
        
        # Use first VRT file
        vrt_path = vrt_files[0]
        print(f"\nUsing: {os.path.basename(vrt_path)}")
        
        # Load data
        bands_dict, meta = classifier.load_vrt_data(vrt_path)
        
        if bands_dict is None:
            continue
        
        # Calculate features
        print("Calculating spectral indices...")
        features = classifier.calculate_indices(bands_dict)
        print(f"Feature shape: {features.shape}")
        
        # Generate training data (REPLACE WITH ACTUAL GROUND TRUTH!)
        X_train, y_train = classifier.generate_synthetic_training_data(features)
        
        # Split into train/test
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        
        # Train and evaluate
        classifier.train_and_evaluate(
            X_train_split, X_test, y_train_split, y_test, dataset_name
        )
    
    # Generate comparison plots
    print(f"\n{'='*80}")
    print("Generating Visualizations and Reports")
    print(f"{'='*80}")
    
    classifier.plot_comparison()
    classifier.plot_confusion_matrices()
    classifier.generate_report()
    
    print(f"\n{'='*80}")
    print("[SUCCESS] Classification Complete!")
    print(f"Results saved in: {classifier.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()



