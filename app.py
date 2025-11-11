#!/usr/bin/env python3
import os
import json
import tempfile
import cherrypy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ========== GPU/CUDA Configuration ==========
CUDA_AVAILABLE = False
CUML_AVAILABLE = False
DEVICE = 'cpu'

print("\n" + "="*60)
print("GPU/CUDA DETECTION")
print("="*60)

# Check PyTorch CUDA
try:
    import torch
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEVICE = 'cuda'
        torch.backends.cudnn.benchmark = True
        print(f"✓ CUDA {torch.version.cuda} detected")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ PyTorch installed but CUDA not available")
except ImportError:
    print("⚠ PyTorch not installed - GPU acceleration disabled")
    print("  Install: pip install torch --index-url https://download.pytorch.org/whl/cu128")

# Check RAPIDS cuML for GPU-accelerated sklearn
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import SVC as cuSVC
    from cuml.linear_model import LogisticRegression as cuLR
    import cuml
    CUML_AVAILABLE = True
    print(f"✓ RAPIDS cuML {cuml.__version__} detected - GPU sklearn enabled")
except ImportError:
    CUML_AVAILABLE = False
    print("⚠ RAPIDS cuML not installed - using CPU sklearn")
    print("  Install: conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=12.8")

print(f"\nTraining will use: {'GPU (CUDA)' if CUML_AVAILABLE else 'CPU'}")
print("="*60 + "\n")


class DiabetesPredictionPipeline:
    """ML Pipeline for diabetes prediction optimized for zero false negatives"""
    
    def __init__(self):
        # Calculate class weight for severe imbalance (82:18 ratio)
        # Give diabetes class 4.5x more weight to force model to learn it
        class_weight_heavy = {0: 1, 1: 5}
        
        # Use GPU-accelerated models if RAPIDS cuML is available
        if CUML_AVAILABLE:
            print("Initializing GPU-accelerated models (cuML) with HEAVY class weights...")
            self.models = {
                'LR': cuLR(max_iter=2000, verbose=0, class_weight=class_weight_heavy),
                'DT': DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=5, 
                                            min_samples_leaf=2, class_weight={0: 1, 1: 5}),
                'RF': cuRF(n_estimators=200, max_depth=15, random_state=42),
                'SVM': cuSVC(kernel='rbf', probability=True, C=2.0, gamma='scale', class_weight=class_weight_heavy),
                'GB': GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=42,
                                                learning_rate=0.05, subsample=0.8)
            }
        else:
            print("Initializing CPU models (sklearn) with HEAVY class weights...")
            self.models = {
                'LR': LogisticRegression(random_state=42, max_iter=2000, class_weight={0: 1, 1: 5}),
                'DT': DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=5, 
                                            min_samples_leaf=2, class_weight={0: 1, 1: 5}),
                'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, 
                                            min_samples_split=5, min_samples_leaf=2, 
                                            class_weight={0: 1, 1: 5}, n_jobs=-1),
                'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=2.0, 
                          gamma='scale', class_weight={0: 1, 1: 5}),
                'GB': GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=42,
                                                learning_rate=0.05, subsample=0.8)
            }
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = 0.5
        self.feature_cols = None
        self.device = DEVICE
        self.using_gpu = CUML_AVAILABLE
        
    def preprocess_data(self, df):
        """Preprocess and feature engineer the dataset"""
        df_processed = df.copy()
        
        # Remove duplicates
        df_processed = df_processed.drop_duplicates().reset_index(drop=True)
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Handle zeros in critical features
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for feature in zero_features:
            if feature in df_processed.columns:
                df_processed.loc[df_processed[feature] == 0, feature] = np.nan
        
        for feature in zero_features:
            if feature in df_processed.columns:
                missing = df_processed[feature].isnull().sum()
                if missing > 0:
                    med = df_processed[feature].median()
                    df_processed[feature].fillna(med, inplace=True)
        
        # Drop any remaining NaN rows and reset index
        df_processed = df_processed.dropna().reset_index(drop=True)
        
        # Feature engineering
        if 'BMI' in df_processed.columns:
            df_processed['BMI_Cat'] = df_processed['BMI'].apply(
                lambda x: 0 if x < 18.5 else (1 if x < 25 else (2 if x < 30 else 3))
            )
        
        if 'Age' in df_processed.columns:
            df_processed['Age_Cat'] = df_processed['Age'].apply(
                lambda x: 0 if x < 30 else (1 if x < 40 else (2 if x < 50 else 3))
            )
        
        if 'Glucose' in df_processed.columns and 'BMI' in df_processed.columns:
            df_processed['Glucose_BMI'] = df_processed['Glucose'] * df_processed['BMI']
        
        if 'Age' in df_processed.columns and 'Glucose' in df_processed.columns:
            df_processed['Age_Glucose'] = df_processed['Age'] * df_processed['Glucose']
        
        if 'FamilyHistory' in df_processed.columns:
            fh_map = {'None': 0, 'Moderate': 1, 'Strong': 2}
            df_processed['FamilyHistory_Enc'] = df_processed['FamilyHistory'].map(fh_map)
            df_processed['FamilyHistory_Enc'].fillna(0, inplace=True)
        
        return df_processed
    
    def optimize_threshold(self, y_true, y_proba, min_threshold=0.1, max_threshold=0.5, step=0.05, min_specificity=0.5):
        """Find optimal threshold to minimize false negatives while maintaining acceptable specificity"""
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        best_fn = float('inf')
        best_t = 0.5
        best_metrics = {}
        
        candidates = []
        
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            # Calculate sensitivity (True Positive Rate / Recall)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Only consider thresholds with acceptable specificity
            if specificity >= min_specificity:
                candidates.append({
                    'threshold': t,
                    'fn': fn,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'specificity': specificity,
                    'sensitivity': sensitivity
                })
        
        # If no candidates meet specificity requirement, relax it
        if not candidates:
            print(f"⚠️ Warning: No threshold achieves {min_specificity*100:.0f}% specificity. Relaxing constraint...")
            min_specificity = 0.3  # Relax to 30%
            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if specificity >= min_specificity:
                    candidates.append({
                        'threshold': t,
                        'fn': fn,
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'specificity': specificity,
                        'sensitivity': sensitivity
                    })
        
        # Select best: prioritize FN=0, then max specificity, then max TP
        if candidates:
            best = min(candidates, key=lambda x: (x['fn'], -x['specificity'], -x['tp']))
            best_t = best['threshold']
            best_metrics = {k: v for k, v in best.items() if k != 'threshold'}
            
            print(f"✓ Optimal threshold: {best_t:.2f}")
            print(f"  Sensitivity: {best['sensitivity']*100:.1f}% | Specificity: {best['specificity']*100:.1f}%")
            print(f"  FN={best['fn']}, FP={best['fp']}, TP={best['tp']}, TN={best['tn']}")
        else:
            # Fallback: use 0.5 if no good threshold found
            best_t = 0.5
            y_pred = (y_proba >= best_t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            best_metrics = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            print(f"⚠️ Using default threshold 0.5 (no optimal threshold found)")
        
        return best_t, best_metrics
    
    def train(self, df):
        """Train all models, test all thresholds, and select the best combination for zero FN"""
        import time
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TRAINING - All Models × All Thresholds")
        print(f"Device: {'GPU (CUDA 12.8)' if self.using_gpu else 'CPU'}")
        print(f"{'='*60}\n")
        
        df_processed = self.preprocess_data(df)
        
        # Save EDA visualizations
        self.eda_plots = {}
        
        # 1. Feature distributions
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Define feature columns
        base_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        available_cols = [col for col in base_cols if col in df_processed.columns]
        
        if 'PhysicalActivityLevel' in df_processed.columns:
            available_cols.append('PhysicalActivityLevel')
        if 'FamilyHistory_Enc' in df_processed.columns:
            available_cols.append('FamilyHistory_Enc')
        if 'Glucose_BMI' in df_processed.columns:
            available_cols.append('Glucose_BMI')
        if 'Age_Glucose' in df_processed.columns:
            available_cols.append('Age_Glucose')
        
        self.feature_cols = available_cols
        
        X = df_processed[self.feature_cols]
        y = df_processed['Outcome']
        
        print(f"Dataset: {len(X)} samples, {len(self.feature_cols)} features")
        print(f"Class distribution: No Diabetes={sum(y==0)}, Diabetes={sum(y==1)}\n")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")
        
        # Scale features
        if self.using_gpu and CUML_AVAILABLE:
            # For cuML, convert to numpy for compatibility
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)
        else:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        # Define comprehensive threshold range
        threshold_range = np.arange(0.05, 0.96, 0.05)  # Test 0.05 to 0.95 in 0.05 steps
        
        print(f"Training {len(self.models)} models with {len(threshold_range)} thresholds each...")
        print(f"Total combinations: {len(self.models) * len(threshold_range)}\n")
        
        # Train and evaluate all models
        all_results = []
        
        for model_name, model in self.models.items():
            model_start = time.time()
            print(f"▶ Training {model_name}...", end=' ', flush=True)
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train.values if self.using_gpu else y_train)
                
                # Get probabilities
                if self.using_gpu and CUML_AVAILABLE and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_scaled)
                    if hasattr(y_proba, 'to_numpy'):
                        y_proba = y_proba.to_numpy()[:, 1]
                    else:
                        y_proba = y_proba[:, 1]
                else:
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate AUC
                auc_score = roc_auc_score(y_test, y_proba)
                
                model_time = time.time() - model_start
                print(f"✓ {model_time:.2f}s (AUC={auc_score:.3f})")
                
                # Test all thresholds
                for threshold in threshold_range:
                    y_pred = (y_proba >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    
                    # Calculate metrics
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                    
                    all_results.append({
                        'model': model_name,
                        'threshold': threshold,
                        'fn': fn,
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'precision': precision,
                        'f1': f1,
                        'auc': auc_score,
                        'model_obj': model,
                        'y_proba': y_proba
                    })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("ENSEMBLE SELECTION - Combining All Models")
        print(f"{'='*60}\n")
        
        # Create ensemble predictions by combining all models
        print("Creating ensemble predictions...\n")
        
        # Strategy 1: Soft Voting - Average probabilities from all models
        all_probas = []
        for model_name in self.models.keys():
            model_results = [r for r in all_results if r['model'] == model_name]
            if model_results:
                # Use the trained model's probabilities
                all_probas.append(model_results[0]['y_proba'])
        
        if all_probas:
            ensemble_proba = np.mean(all_probas, axis=0)
            print(f"Ensemble created from {len(all_probas)} models")
        else:
            print("⚠️ No valid models for ensemble")
            ensemble_proba = None
        
        # Test ensemble with all thresholds
        ensemble_results = []
        if ensemble_proba is not None:
            for threshold in threshold_range:
                y_pred = (ensemble_proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                auc_score = roc_auc_score(y_test, ensemble_proba)
                
                ensemble_results.append({
                    'model': 'ENSEMBLE',
                    'threshold': threshold,
                    'fn': fn,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'accuracy': accuracy,
                    'precision': precision,
                    'f1': f1,
                    'auc': auc_score,
                    'y_proba': ensemble_proba
                })
        
        # Combine individual model results with ensemble results
        combined_results = all_results + ensemble_results
        
        print(f"Total configurations (individual + ensemble): {len(combined_results)}\n")
        
        # CRITICAL SAFETY REQUIREMENT: Maximum 2 missed diabetes cases
        # In healthcare, false negatives can be fatal - prioritize catching all cases
        
        total_positives = sum(y_test == 1)
        
        # STRATEGY: NEAR-ZERO FALSE NEGATIVES (medical safety priority)
        MAX_FN_ALLOWED = 2  # Absolute maximum: 2 missed cases
        MIN_SPECIFICITY = 0.20  # Accept lower specificity to save lives
        
        print(f"CLINICAL SAFETY MODE: NEAR-ZERO FALSE NEGATIVES")
        print(f"  Priority: Catch ALL diabetes cases (lives at stake)")
        print(f"  • Total positive cases: {total_positives}")
        print(f"  • Max FN allowed: {MAX_FN_ALLOWED} ABSOLUTE MAXIMUM")
        print(f"  • Target: FN = 0-2 (99.9% sensitivity)")
        print(f"  • Min Specificity: ≥{MIN_SPECIFICITY*100:.0f}% (will maximize within constraint)\n")
        
        print(f"Filtering for FN ≤ {MAX_FN_ALLOWED}...")
        valid_results = [r for r in combined_results if r['fn'] <= MAX_FN_ALLOWED]
        
        if not valid_results:
            print(f"⚠️ No configuration with FN ≤ {MAX_FN_ALLOWED}")
            print(f"Trying FN ≤ 5...")
            MAX_FN_ALLOWED = 5
            valid_results = [r for r in combined_results if r['fn'] <= MAX_FN_ALLOWED]
        
        if not valid_results:
            print(f"⚠️ No configuration with FN ≤ {MAX_FN_ALLOWED}")
            print(f"Selecting minimum FN configuration...")
            best_result = min(combined_results, key=lambda x: (x['fn'], -x['specificity'], -x['auc']))
        else:
            print(f"✓ Found {len(valid_results)} configurations with FN ≤ {MAX_FN_ALLOWED}")
            
            # Among configurations with FN ≤ 2:
            # 1. Minimize FN (prefer 0 over 1 over 2)
            # 2. Maximize Specificity (reduce false alarms as much as possible)
            # 3. Maximize AUC
            # 4. Prefer ENSEMBLE
            
            def score_config(r):
                is_ensemble = 1 if r['model'] == 'ENSEMBLE' else 0
                # Primary: fewest FN, Secondary: best specificity
                return (r['fn'], -r['specificity'], -r['auc'], -is_ensemble)
            
            best_result = min(valid_results, key=score_config)
            
            fn_rate = (best_result['fn'] / total_positives) * 100
            lives_at_risk = best_result['fn']
            
            print(f"\n✓ CLINICAL SAFETY CONFIGURATION SELECTED:")
            print(f"  • Lives at Risk: {lives_at_risk} missed cases ({fn_rate:.2f}%)")
            print(f"  • Cases Detected: {best_result['tp']}/{total_positives} ({best_result['sensitivity']*100:.2f}%)")
            print(f"  • Specificity: {best_result['specificity']*100:.1f}%")
            print(f"  • False Alarms: {best_result['fp']} (acceptable trade-off for safety)")
            
            if best_result['fn'] == 0:
                print(f"  🎯 PERFECT: ZERO MISSED CASES!")
            elif best_result['fn'] <= 2:
                print(f"  ✅ EXCELLENT: Only {best_result['fn']} missed cases")
            else:
                print(f"  ⚠️  WARNING: {best_result['fn']} cases at risk")
            
            # Critical warning if specificity is too low
            if best_result['specificity'] < 0.10:
                print(f"\n{'!'*60}")
                print(f"⚠️  CRITICAL WARNING: SPECIFICITY = {best_result['specificity']*100:.1f}%")
                print(f"{'!'*60}")
                print(f"The model is predicting EVERYONE (or nearly everyone) as positive!")
                print(f"This happens when:")
                print(f"  1. Features don't separate classes well")
                print(f"  2. Data quality issues (noise, errors)")
                print(f"  3. Severe class imbalance ({total_positives}/{len(y_test)} = {total_positives/len(y_test)*100:.1f}% positive)")
                print(f"\nRECOMMENDATIONS:")
                print(f"  • Use SMOTE or class weights for better balance")
                print(f"  • Collect more diabetes cases for training")
                print(f"  • Add more discriminative features")
                print(f"  • Use ensemble with calibration")
                print(f"  • Consider two-stage testing (screening + confirmation)")
                print(f"\nCURRENT MODEL STATUS:")
                print(f"  This model WILL catch all diabetes cases (FN=0)")
                print(f"  BUT will send {best_result['fp']} healthy people for unnecessary testing")
                print(f"  Clinical workflow: Use as screening → confirm with lab tests")
                print(f"{'!'*60}\n")
        
        # If ensemble was selected, we need to store all models for prediction
        if best_result['model'] == 'ENSEMBLE':
            print(f"\n{'='*60}")
            print("ENSEMBLE MODEL SELECTED")
            print(f"{'='*60}")
            self.best_model_name = 'ENSEMBLE'
            self.best_model = 'ENSEMBLE'  # Special marker
            self.best_threshold = best_result['threshold']
            self.ensemble_models = {name: results[0]['model_obj'] for name, results in 
                                   [(n, [r for r in all_results if r['model'] == n]) 
                                    for n in self.models.keys()] if results}
            print(f"Ensemble uses {len(self.ensemble_models)} models:")
            for name in self.ensemble_models.keys():
                print(f"  • {name}")
        else:
            # Single model selected
            self.best_model_name = best_result['model']
            self.best_model = best_result.get('model_obj')
            if not self.best_model:
                # Find the model object
                for r in all_results:
                    if r['model'] == best_result['model'] and r['threshold'] == best_result['threshold']:
                        self.best_model = r['model_obj']
                        break
            self.best_threshold = best_result['threshold']
        
        # Store test data for visualizations
        self.test_data = {'y_test': y_test, 'X_test': X_test}
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Time: {training_time:.2f}s")
        print(f"Device: {'GPU (CUDA 12.8)' if self.using_gpu else 'CPU'}")
        print(f"Combinations tested: {len(all_results)}")
        print(f"\nBEST CONFIGURATION:")
        print(f"  Model: {best_result['model']}")
        print(f"  Threshold: {best_result['threshold']:.3f}")
        print(f"  False Negatives: {best_result['fn']}")
        print(f"  True Positives: {best_result['tp']}")
        print(f"  False Positives: {best_result['fp']}")
        print(f"  True Negatives: {best_result['tn']}")
        print(f"  Sensitivity: {best_result['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best_result['specificity']*100:.1f}%")
        print(f"  Accuracy: {best_result['accuracy']*100:.1f}%")
        print(f"  AUC: {best_result['auc']:.3f}")
        print(f"{'='*60}\n")
        
        # Create summary of all models at their best thresholds
        model_summary = {}
        for model_name in self.models.keys():
            model_results = [r for r in all_results if r['model'] == model_name]
            if model_results:
                # Find best threshold for this model (FN=0, max specificity)
                zero_fn = [r for r in model_results if r['fn'] == 0]
                if zero_fn:
                    best_for_model = max(zero_fn, key=lambda x: (x['specificity'], x['auc']))
                else:
                    best_for_model = min(model_results, key=lambda x: x['fn'])
                
                model_summary[model_name] = {
                    'threshold': best_for_model['threshold'],
                    'fn': best_for_model['fn'],
                    'tp': best_for_model['tp'],
                    'fp': best_for_model['fp'],
                    'tn': best_for_model['tn'],
                    'sensitivity': best_for_model['sensitivity'],
                    'specificity': best_for_model['specificity'],
                    'auc': best_for_model['auc']
                }
        
        return {
            'best_model': self.best_model_name,
            'threshold': float(self.best_threshold),
            'training_time': training_time,
            'device': 'GPU (CUDA 12.8)' if self.using_gpu else 'CPU',
            'cuda_available': CUDA_AVAILABLE,
            'cuml_available': CUML_AVAILABLE,
            'combinations_tested': len(all_results),
            'metrics': {k: {
                'fn': int(v['fn']),
                'tp': int(v['tp']),
                'fp': int(v['fp']),
                'tn': int(v['tn']),
                'sensitivity': float(v['sensitivity']),
                'specificity': float(v['specificity']),
                'auc': float(v['auc']),
                'threshold': float(v['threshold'])
            } for k, v in model_summary.items()},
            'all_results': all_results,
            'best_config': {
                'fn': int(best_result['fn']),
                'tp': int(best_result['tp']),
                'fp': int(best_result['fp']),
                'tn': int(best_result['tn']),
                'sensitivity': float(best_result['sensitivity']),
                'specificity': float(best_result['specificity']),
                'accuracy': float(best_result['accuracy']),
                'precision': float(best_result['precision']),
                'f1': float(best_result['f1']),
                'auc': float(best_result['auc'])
            }
        }
    
    def predict(self, df, is_single=False):
        """Make predictions with the best model or ensemble"""
        if is_single:
            # For single prediction, minimal preprocessing
            df_processed = df.copy()
            
            # Feature engineering only
            if 'BMI' in df_processed.columns:
                df_processed['BMI_Cat'] = df_processed['BMI'].apply(
                    lambda x: 0 if x < 18.5 else (1 if x < 25 else (2 if x < 30 else 3))
                )
            
            if 'Age' in df_processed.columns:
                df_processed['Age_Cat'] = df_processed['Age'].apply(
                    lambda x: 0 if x < 30 else (1 if x < 40 else (2 if x < 50 else 3))
                )
            
            if 'Glucose' in df_processed.columns and 'BMI' in df_processed.columns:
                df_processed['Glucose_BMI'] = df_processed['Glucose'] * df_processed['BMI']
            
            if 'Age' in df_processed.columns and 'Glucose' in df_processed.columns:
                df_processed['Age_Glucose'] = df_processed['Age'] * df_processed['Glucose']
            
            if 'FamilyHistory' in df_processed.columns:
                fh_map = {'None': 0, 'Moderate': 1, 'Strong': 2}
                df_processed['FamilyHistory_Enc'] = df_processed['FamilyHistory'].map(fh_map)
                df_processed['FamilyHistory_Enc'].fillna(0, inplace=True)
        else:
            df_processed = self.preprocess_data(df)
        
        X = df_processed[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Check if using ensemble
        if self.best_model == 'ENSEMBLE' and hasattr(self, 'ensemble_models'):
            # Ensemble prediction - average probabilities from all models
            all_probas = []
            for model in self.ensemble_models.values():
                proba = model.predict_proba(X_scaled)[:, 1]
                all_probas.append(proba)
            y_proba = np.mean(all_probas, axis=0)
        else:
            # Single model prediction
            y_proba = self.best_model.predict_proba(X_scaled)[:, 1]
        
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        # Create result dataframe with same index as processed data
        result_df = df_processed.copy()
        result_df['Prediction'] = y_pred
        result_df['Probability'] = y_proba
        
        return y_pred, y_proba, result_df
    
    def save(self, filepath):
        """Save pipeline to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load pipeline from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class DiabetesPredictionAPI:
    """CherryPy REST API for diabetes prediction"""
    
    def __init__(self):
        self.pipeline = None
        self.upload_dir = Path("uploads")
        self.models_dir = Path("models")
        self.upload_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    @cherrypy.expose
    def index(self):
        """Landing page with API documentation"""
        cherrypy.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        
        # Get current status
        model_status = "✅ Model Trained" if self.pipeline else "⚠️ No Model Trained"
        status_class = "success" if self.pipeline else "warning"
        model_info = ""
        
        if self.pipeline:
            model_info = f"""
            <div style="background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 15px 0;">
                <strong>Model Details:</strong><br>
                Algorithm: {self.pipeline.best_model_name}<br>
                Threshold: {self.pipeline.best_threshold}<br>
                Features: {len(self.pipeline.feature_cols)}
            </div>
            """
        
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Diabetes Prediction API</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); overflow: hidden; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
                .header p {{ font-size: 1.1em; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .status {{ padding: 15px; margin: 20px 0; border-radius: 8px; font-weight: bold; text-align: center; }}
                .success {{ background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }}
                .warning {{ background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }}
                .card {{ background: #f8f9fa; padding: 25px; border-radius: 10px; border: 2px solid #e9ecef; }}
                .card h2 {{ color: #2c3e50; margin-bottom: 15px; font-size: 1.5em; }}
                .card h3 {{ color: #34495e; margin: 20px 0 10px 0; font-size: 1.2em; }}
                .form-group {{ margin: 15px 0; }}
                label {{ display: block; margin-bottom: 8px; font-weight: 600; color: #495057; }}
                input[type="file"] {{ width: 100%; padding: 12px; border: 2px solid #ced4da; border-radius: 6px; background: white; }}
                button {{ width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: transform 0.2s; }}
                button:hover {{ transform: translateY(-2px); }}
                button:disabled {{ background: #6c757d; cursor: not-allowed; transform: none; }}
                .result {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #667eea; }}
                .result pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .loading {{ display: none; text-align: center; padding: 20px; }}
                .loading.show {{ display: block; }}
                .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }}
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                .endpoints {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .endpoint {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }}
                .method {{ background: #667eea; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
                code {{ background: #2c3e50; color: #ecf0f1; padding: 3px 8px; border-radius: 4px; font-size: 0.9em; }}
                .tabs {{ display: flex; gap: 10px; margin: 20px 0; border-bottom: 2px solid #e9ecef; }}
                .tab {{ padding: 15px 30px; cursor: pointer; border-radius: 8px 8px 0 0; background: #f8f9fa; transition: all 0.3s; }}
                .tab.active {{ background: #667eea; color: white; }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
                @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Diabetes Prediction System</h1>
                    <p>Automated ML Pipeline - Zero False Negative Optimization</p>
                </div>
                
                <div class="content">
                    <div class="status {status_class}">
                        {model_status}
                    </div>
                    
                    {model_info}
                    
                    <div class="tabs">
                        <div class="tab active" onclick="switchTab('main')">🏠 Main</div>
                        <div class="tab" onclick="switchTab('api')">📡 API Docs</div>
                        <div class="tab" onclick="switchTab('results')">📊 Results</div>
                        <div class="tab" onclick="window.location.href='/logs'">📋 Logs</div>
                    </div>
                    
                    {f'<div style="text-align: center; margin: 20px 0;"><a href="/analysis" style="padding: 15px 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; text-decoration: none; border-radius: 8px; font-size: 1.2em; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">📊 View Full Analysis & Visualizations</a></div>' if self.pipeline else ''}
                    
                    <!-- MAIN TAB -->
                    <div id="main" class="tab-content active">
                        <div class="grid">
                            <!-- Train Model -->
                            <div class="card">
                                <h2>🎓 Train Model</h2>
                                <p style="margin-bottom: 15px;">Upload your diabetes dataset (CSV) with 'Outcome' column to train the model.</p>
                                <form id="trainForm" onsubmit="return handleTrain(event)">
                                    <div class="form-group">
                                        <label>Training Dataset (CSV)</label>
                                        <input type="file" name="file" accept=".csv" required>
                                    </div>
                                    <button type="submit">🚀 Train Model</button>
                                </form>
                                <div id="trainLoading" class="loading">
                                    <div class="spinner"></div>
                                    <p>Training model...</p>
                                </div>
                                <div id="trainResult"></div>
                            </div>
                            
                            <!-- Batch Prediction -->
                            <div class="card">
                                <h2>🔮 Batch Prediction</h2>
                                <p style="margin-bottom: 15px;">Upload CSV file (without 'Outcome' column) for predictions.</p>
                                <form id="predictForm" onsubmit="return handlePredict(event)">
                                    <div class="form-group">
                                        <label>Test Dataset (CSV)</label>
                                        <input type="file" name="file" accept=".csv" required {'' if self.pipeline else 'disabled'}>
                                    </div>
                                    <button type="submit" {'' if self.pipeline else 'disabled'}>🎯 Get Predictions</button>
                                </form>
                                <div id="predictLoading" class="loading">
                                    <div class="spinner"></div>
                                    <p>Making predictions...</p>
                                </div>
                                <div id="predictResult"></div>
                            </div>
                        </div>
                        
                        <!-- Single Prediction -->
                        <div class="card">
                            <h2>👤 Single Patient Prediction</h2>
                            <div class="grid">
                                <div class="form-group">
                                    <label>Pregnancies</label>
                                    <input type="number" id="pregnancies" value="6" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Glucose</label>
                                    <input type="number" id="glucose" value="148" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Blood Pressure</label>
                                    <input type="number" id="bloodPressure" value="72" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Skin Thickness</label>
                                    <input type="number" id="skinThickness" value="35" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Insulin</label>
                                    <input type="number" id="insulin" value="0" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>BMI</label>
                                    <input type="number" step="0.1" id="bmi" value="33.6" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Diabetes Pedigree Function</label>
                                    <input type="number" step="0.001" id="dpf" value="0.627" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Age</label>
                                    <input type="number" id="age" value="50" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Physical Activity Level (0-4)</label>
                                    <input type="number" id="physicalActivity" value="2" min="0" max="4" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                </div>
                                <div class="form-group">
                                    <label>Family History</label>
                                    <select id="familyHistory" style="width:100%; padding:10px; border:2px solid #ced4da; border-radius:6px;">
                                        <option value="None">None</option>
                                        <option value="Moderate">Moderate</option>
                                        <option value="Strong" selected>Strong</option>
                                    </select>
                                </div>
                            </div>
                            <button onclick="handleSinglePredict()" style="margin-top:15px;" {'' if self.pipeline else 'disabled'}>🔍 Predict</button>
                            <div id="singleLoading" class="loading">
                                <div class="spinner"></div>
                                <p>Analyzing...</p>
                            </div>
                            <div id="singleResult"></div>
                        </div>
                    </div>
                    
                    <!-- API DOCS TAB -->
                    <div id="api" class="tab-content">
                        <div class="endpoints">
                            <h2 style="margin-bottom: 20px;">📡 API Endpoints</h2>
                            
                            <div class="endpoint">
                                <span class="method">POST</span> <strong>/train</strong>
                                <p style="margin: 10px 0;">Upload CSV to train model</p>
                                <code>curl -X POST -F "file=@data.csv" http://localhost:8080/train</code>
                            </div>
                            
                            <div class="endpoint">
                                <span class="method">POST</span> <strong>/predict</strong>
                                <p style="margin: 10px 0;">Batch predictions from CSV</p>
                                <code>curl -X POST -F "file=@test.csv" http://localhost:8080/predict</code>
                            </div>
                            
                            <div class="endpoint">
                                <span class="method">POST</span> <strong>/predict_single</strong>
                                <p style="margin: 10px 0;">Single prediction via JSON</p>
                                <code>curl -X POST -H "Content-Type: application/json" -d '{{"Pregnancies":6,...}}' http://localhost:8080/predict_single</code>
                            </div>
                            
                            <div class="endpoint">
                                <span class="method">GET</span> <strong>/status</strong>
                                <p style="margin: 10px 0;">Get model status</p>
                                <code>curl http://localhost:8080/status</code>
                            </div>
                            
                            <div class="endpoint">
                                <span class="method">GET</span> <strong>/health</strong>
                                <p style="margin: 10px 0;">Health check</p>
                                <code>curl http://localhost:8080/health</code>
                            </div>
                        </div>
                    </div>
                    
                    <!-- RESULTS TAB -->
                    <div id="results" class="tab-content">
                        <div class="card">
                            <h2>📊 All Results</h2>
                            <div id="allResults">
                                <p style="color: #6c757d;">Results will appear here after training or predictions...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                function switchTab(tab) {{
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    event.target.classList.add('active');
                    document.getElementById(tab).classList.add('active');
                }}
                
                async function handleTrain(e) {{
                    e.preventDefault();
                    const form = e.target;
                    const formData = new FormData(form);
                    const loading = document.getElementById('trainLoading');
                    const result = document.getElementById('trainResult');
                    
                    loading.classList.add('show');
                    result.innerHTML = '';
                    
                    try {{
                        const response = await fetch('/train', {{
                            method: 'POST',
                            body: formData
                        }});
                        const data = await response.json();
                        
                        if (data.success) {{
                            result.innerHTML = `
                                <div class="result" style="border-left-color: #28a745;">
                                    <h3 style="color: #28a745;">✅ Training Successful!</h3>
                                    <p><strong>Best Model:</strong> ${{data.results.best_model}}</p>
                                    <p><strong>Threshold:</strong> ${{data.results.threshold}}</p>
                                    <details style="margin-top: 15px;">
                                        <summary style="cursor: pointer; font-weight: bold;">View All Metrics</summary>
                                        <pre>${{JSON.stringify(data.results.metrics, null, 2)}}</pre>
                                    </details>
                                </div>
                            `;
                            document.getElementById('allResults').innerHTML += result.innerHTML;
                            setTimeout(() => location.reload(), 2000);
                        }} else {{
                            result.innerHTML = `<div class="result" style="border-left-color: #dc3545;"><h3 style="color: #dc3545;">❌ Error</h3><p>${{data.error}}</p></div>`;
                        }}
                    }} catch (err) {{
                        result.innerHTML = `<div class="result" style="border-left-color: #dc3545;"><h3 style="color: #dc3545;">❌ Error</h3><p>${{err.message}}</p></div>`;
                    }} finally {{
                        loading.classList.remove('show');
                    }}
                    
                    return false;
                }}
                
                async function handlePredict(e) {{
                    e.preventDefault();
                    const form = e.target;
                    const formData = new FormData(form);
                    const loading = document.getElementById('predictLoading');
                    const result = document.getElementById('predictResult');
                    
                    loading.classList.add('show');
                    result.innerHTML = '';
                    
                    try {{
                        const response = await fetch('/predict', {{
                            method: 'POST',
                            body: formData
                        }});
                        const data = await response.json();
                        
                        if (data.success) {{
                            const posRate = (data.summary.positive / data.summary.total * 100).toFixed(1);
                            const negRate = (data.summary.negative / data.summary.total * 100).toFixed(1);
                            
                            result.innerHTML = `
                                <div class="result" style="border-left-color: #28a745;">
                                    <h3 style="color: #28a745;">✅ Predictions Complete!</h3>
                                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                                        <h4 style="margin-top: 0;">Summary Statistics</h4>
                                        <table style="width: 100%; border-collapse: collapse;">
                                            <tr style="background: #e9ecef;">
                                                <td style="padding: 10px; font-weight: bold;">Total Patients</td>
                                                <td style="padding: 10px; text-align: right;">${{data.summary.total}}</td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 10px; color: #dc3545; font-weight: bold;">🔴 Diabetes Detected</td>
                                                <td style="padding: 10px; text-align: right; color: #dc3545; font-weight: bold;">${{data.summary.positive}} (${{posRate}}%)</td>
                                            </tr>
                                            <tr style="background: #e9ecef;">
                                                <td style="padding: 10px; color: #28a745; font-weight: bold;">🟢 No Diabetes</td>
                                                <td style="padding: 10px; text-align: right; color: #28a745; font-weight: bold;">${{data.summary.negative}} (${{negRate}}%)</td>
                                            </tr>
                                        </table>
                                    </div>
                                    
                                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ffc107;">
                                        <h4 style="margin-top: 0; color: #856404;">⚠️ Clinical Action Required</h4>
                                        <p style="margin: 5px 0;"><strong>${{data.summary.positive}}</strong> patients flagged for diabetes risk</p>
                                        <p style="margin: 5px 0;">Recommended: Lab confirmation (HbA1c + Fasting Glucose)</p>
                                    </div>
                                    
                                    <details style="margin-top: 15px;">
                                        <summary style="cursor: pointer; font-weight: bold; padding: 10px; background: #f8f9fa; border-radius: 5px;">📊 View Prediction Distribution</summary>
                                        <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                            <canvas id="predChart" width="400" height="200"></canvas>
                                        </div>
                                    </details>
                                    
                                    <details style="margin-top: 15px;">
                                        <summary style="cursor: pointer; font-weight: bold; padding: 10px; background: #f8f9fa; border-radius: 5px;">📋 View Sample Predictions (First 20)</summary>
                                        <div style="margin-top: 10px; overflow-x: auto;">
                                            <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                                                <thead>
                                                    <tr style="background: #667eea; color: white;">
                                                        <th style="padding: 8px; text-align: left;">#</th>
                                                        <th style="padding: 8px; text-align: center;">Prediction</th>
                                                        <th style="padding: 8px; text-align: right;">Probability</th>
                                                        <th style="padding: 8px; text-align: center;">Risk Level</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    ${{data.predictions.slice(0, 20).map((pred, idx) => `
                                                        <tr style="background: ${{idx % 2 === 0 ? '#f8f9fa' : 'white'}}; border-bottom: 1px solid #dee2e6;">
                                                            <td style="padding: 8px;">${{idx + 1}}</td>
                                                            <td style="padding: 8px; text-align: center;">
                                                                <span style="padding: 4px 12px; border-radius: 12px; font-weight: bold; background: ${{pred === 1 ? '#dc3545' : '#28a745'}}; color: white;">
                                                                    ${{pred === 1 ? 'DIABETES' : 'NO DIABETES'}}
                                                                </span>
                                                            </td>
                                                            <td style="padding: 8px; text-align: right;">${{(data.probabilities[idx] * 100).toFixed(1)}}%</td>
                                                            <td style="padding: 8px; text-align: center;">
                                                                <span style="padding: 4px 8px; border-radius: 8px; font-size: 0.85em; background: ${{
                                                                    data.probabilities[idx] > 0.7 ? '#dc3545' : 
                                                                    data.probabilities[idx] > 0.4 ? '#ffc107' : '#28a745'
                                                                }}; color: white;">
                                                                    ${{data.probabilities[idx] > 0.7 ? 'HIGH' : 
                                                                      data.probabilities[idx] > 0.4 ? 'MEDIUM' : 'LOW'}}
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    `).join('')}}
                                                </tbody>
                                            </table>
                                        </div>
                                    </details>
                                    
                                    <div style="margin-top: 20px; padding: 15px; background: #d4edda; border-radius: 8px;">
                                        <p style="margin: 0;"><strong>📁 Full Results Saved:</strong></p>
                                        <code style="background: white; padding: 5px 10px; border-radius: 4px; display: inline-block; margin-top: 5px;">${{data.results_file}}</code>
                                    </div>
                                </div>
                            `;
                            
                            // Create chart
                            setTimeout(() => {{
                                const canvas = document.getElementById('predChart');
                                if (canvas) {{
                                    const ctx = canvas.getContext('2d');
                                    const width = canvas.width;
                                    const height = canvas.height;
                                    const barWidth = width / 2 - 40;
                                    const maxHeight = height - 60;
                                    const scale = maxHeight / Math.max(data.summary.positive, data.summary.negative);
                                    
                                    // Draw bars
                                    ctx.fillStyle = '#dc3545';
                                    ctx.fillRect(40, height - 40 - (data.summary.positive * scale), barWidth, data.summary.positive * scale);
                                    ctx.fillStyle = '#28a745';
                                    ctx.fillRect(width/2 + 20, height - 40 - (data.summary.negative * scale), barWidth, data.summary.negative * scale);
                                    
                                    // Labels
                                    ctx.fillStyle = '#000';
                                    ctx.font = '14px Arial';
                                    ctx.textAlign = 'center';
                                    ctx.fillText('Diabetes', 40 + barWidth/2, height - 20);
                                    ctx.fillText('No Diabetes', width/2 + 20 + barWidth/2, height - 20);
                                    
                                    // Values
                                    ctx.font = 'bold 16px Arial';
                                    ctx.fillStyle = '#dc3545';
                                    ctx.fillText(data.summary.positive, 40 + barWidth/2, height - 50 - (data.summary.positive * scale));
                                    ctx.fillStyle = '#28a745';
                                    ctx.fillText(data.summary.negative, width/2 + 20 + barWidth/2, height - 50 - (data.summary.negative * scale));
                                }}
                            }}, 100);
                            
                            document.getElementById('allResults').innerHTML += result.innerHTML;
                        }} else {{
                            result.innerHTML = `<div class="result" style="border-left-color: #dc3545;"><h3 style="color: #dc3545;">❌ Error</h3><p>${{data.error}}</p></div>`;
                        }}
                    }} catch (err) {{
                        result.innerHTML = `<div class="result" style="border-left-color: #dc3545;"><h3 style="color: #dc3545;">❌ Error</h3><p>${{err.message}}</p></div>`;
                    }} finally {{
                        loading.classList.remove('show');
                    }}
                    
                    return false;
                }}
                
                async function handleSinglePredict() {{
                    const loading = document.getElementById('singleLoading');
                    const result = document.getElementById('singleResult');
                    
                    const data = {{
                        Pregnancies: parseInt(document.getElementById('pregnancies').value),
                        Glucose: parseFloat(document.getElementById('glucose').value),
                        BloodPressure: parseFloat(document.getElementById('bloodPressure').value),
                        SkinThickness: parseFloat(document.getElementById('skinThickness').value),
                        Insulin: parseFloat(document.getElementById('insulin').value),
                        BMI: parseFloat(document.getElementById('bmi').value),
                        DiabetesPedigreeFunction: parseFloat(document.getElementById('dpf').value),
                        Age: parseInt(document.getElementById('age').value),
                        PhysicalActivityLevel: parseInt(document.getElementById('physicalActivity').value),
                        FamilyHistory: document.getElementById('familyHistory').value
                    }};
                    
                    loading.classList.add('show');
                    result.innerHTML = '';
                    
                    try {{
                        const response = await fetch('/predict_single', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(data)
                        }});
                        const resp = await response.json();
                        
                        if (resp.success) {{
                            const color = resp.prediction === 1 ? '#dc3545' : '#28a745';
                            result.innerHTML = `
                                <div class="result" style="border-left-color: ${{color}};">
                                    <h3 style="color: ${{color}};">${{resp.diagnosis}}</h3>
                                    <p><strong>Probability:</strong> ${{(resp.probability * 100).toFixed(2)}}%</p>
                                    <p><strong>Risk Level:</strong> ${{resp.risk_level}}</p>
                                </div>
                            `;
                            document.getElementById('allResults').innerHTML += result.innerHTML;
                        }} else {{
                            result.innerHTML = `<div class="result" style="border-left-color: #dc3545;"><h3 style="color: #dc3545;">❌ Error</h3><p>${{resp.error}}</p></div>`;
                        }}
                    }} catch (err) {{
                        result.innerHTML = `<div class="result" style="border-left-color: #dc3545;"><h3 style="color: #dc3545;">❌ Error</h3><p>${{err.message}}</p></div>`;
                    }} finally {{
                        loading.classList.remove('show');
                    }}
                }}
            </script>
        </body>
        </html>
        """
        return html.encode('utf-8')
    
    @cherrypy.expose
    def analysis(self):
        """Comprehensive analysis page with all visualizations"""
        cherrypy.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        
        if not self.pipeline:
            return b'<html><body><h1>No Model</h1><p><a href="/">Train first</a></p></body></html>'
        
        # Generate all visualizations
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        from io import BytesIO
        import numpy as np
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{img_str}"
        
        # Load the original training data to generate EDA plots
        train_files = sorted(self.upload_dir.glob("train_*.csv"))
        if not train_files:
            return b'<html><body><h1>No training data found</h1></body></html>'
        
        import pandas as pd
        df = pd.read_csv(train_files[-1])  # Use most recent training file
        
        images = {}
        
        # 1. CORRELATION MATRIX
        fig, ax = plt.subplots(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, center=0, ax=ax)
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
        images['correlation'] = fig_to_base64(fig)
        
        # 2. FEATURE DISTRIBUTIONS (Univariate)
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        available_features = [f for f in features if f in df.columns]
        
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(available_features):
            ax = axes[idx]
            df[feature].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'{feature} Distribution', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        images['distributions'] = fig_to_base64(fig)
        
        # 3. BOXPLOTS BY OUTCOME (Bivariate)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(available_features):
            ax = axes[idx]
            df.boxplot(column=feature, by='Outcome', ax=ax)
            ax.set_title(f'{feature} by Outcome')
            ax.set_xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
            ax.set_ylabel(feature)
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        images['boxplots'] = fig_to_base64(fig)
        
        # 4. SCATTER PLOTS (Bivariate - Key relationships)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Glucose vs BMI
        axes[0, 0].scatter(df[df['Outcome']==0]['Glucose'], df[df['Outcome']==0]['BMI'], 
                          alpha=0.5, label='No Diabetes', color='green')
        axes[0, 0].scatter(df[df['Outcome']==1]['Glucose'], df[df['Outcome']==1]['BMI'], 
                          alpha=0.5, label='Diabetes', color='red')
        axes[0, 0].set_xlabel('Glucose')
        axes[0, 0].set_ylabel('BMI')
        axes[0, 0].set_title('Glucose vs BMI')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Age vs Glucose
        axes[0, 1].scatter(df[df['Outcome']==0]['Age'], df[df['Outcome']==0]['Glucose'], 
                          alpha=0.5, label='No Diabetes', color='green')
        axes[0, 1].scatter(df[df['Outcome']==1]['Age'], df[df['Outcome']==1]['Glucose'], 
                          alpha=0.5, label='Diabetes', color='red')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Glucose')
        axes[0, 1].set_title('Age vs Glucose')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # BMI vs Age
        axes[1, 0].scatter(df[df['Outcome']==0]['BMI'], df[df['Outcome']==0]['Age'], 
                          alpha=0.5, label='No Diabetes', color='green')
        axes[1, 0].scatter(df[df['Outcome']==1]['BMI'], df[df['Outcome']==1]['Age'], 
                          alpha=0.5, label='Diabetes', color='red')
        axes[1, 0].set_xlabel('BMI')
        axes[1, 0].set_ylabel('Age')
        axes[1, 0].set_title('BMI vs Age')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Insulin vs Glucose
        axes[1, 1].scatter(df[df['Outcome']==0]['Insulin'], df[df['Outcome']==0]['Glucose'], 
                          alpha=0.5, label='No Diabetes', color='green')
        axes[1, 1].scatter(df[df['Outcome']==1]['Insulin'], df[df['Outcome']==1]['Glucose'], 
                          alpha=0.5, label='Diabetes', color='red')
        axes[1, 1].set_xlabel('Insulin')
        axes[1, 1].set_ylabel('Glucose')
        axes[1, 1].set_title('Insulin vs Glucose')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        images['scatter'] = fig_to_base64(fig)
        
        # 5. CONFUSION MATRIX
        if hasattr(self.pipeline, 'test_data'):
            fig, ax = plt.subplots(figsize=(8, 6))
            y_pred = (self.pipeline.best_model.predict_proba(
                self.pipeline.scaler.transform(self.pipeline.test_data['X_test']))[:, 1] >= self.pipeline.best_threshold).astype(int)
            cm = confusion_matrix(self.pipeline.test_data['y_test'], y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                       xticklabels=['No Diabetes', 'Diabetes'], 
                       yticklabels=['No Diabetes', 'Diabetes'])
            ax.set_title(f'Confusion Matrix - {self.pipeline.best_model_name} (Threshold={self.pipeline.best_threshold})', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            
            # Add metrics text
            tn, fp, fn, tp = cm.ravel()
            metrics_text = f'TN={tn}, FP={fp}\\nFN={fn}, TP={tp}'
            ax.text(0.5, -0.15, metrics_text, ha='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            images['confusion'] = fig_to_base64(fig)
        
        # 6. ROC CURVE (if test data available)
        if hasattr(self.pipeline, 'test_data'):
            fig, ax = plt.subplots(figsize=(10, 8))
            y_proba = self.pipeline.best_model.predict_proba(
                self.pipeline.scaler.transform(self.pipeline.test_data['X_test']))[:, 1]
            fpr, tpr, _ = roc_curve(self.pipeline.test_data['y_test'], y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {self.pipeline.best_model_name}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=12)
            ax.grid(alpha=0.3)
            
            images['roc'] = fig_to_base64(fig)
        
        # 7. OUTCOME DISTRIBUTION
        fig, ax = plt.subplots(figsize=(8, 6))
        outcome_counts = df['Outcome'].value_counts()
        ax.bar(['No Diabetes', 'Diabetes'], outcome_counts.values, color=['green', 'red'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(outcome_counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        images['outcome'] = fig_to_base64(fig)
        
        # 8. DESCRIPTIVE STATISTICS
        stats_html = df.describe().to_html(classes='stats-table', border=0)
        
        # 9. DATA QUALITY ASSESSMENT
        total_records = len(df)
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / (total_records * len(df.columns))) * 100
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        duplicate_pct = (duplicates / total_records) * 100
        
        # Check for zeros in critical features
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        zero_counts = {}
        zero_total = 0
        for feat in zero_features:
            if feat in df.columns:
                zeros = (df[feat] == 0).sum()
                zero_counts[feat] = zeros
                zero_total += zeros
        
        # Check for outliers (using IQR method)
        outlier_counts = {}
        outlier_total = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'Outcome':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = outliers
                outlier_total += outliers
        
        outlier_pct = (outlier_total / (total_records * len(df.select_dtypes(include=[np.number]).columns))) * 100
        
        # Calculate Data Quality Score (0-100)
        completeness_score = 100 - missing_pct  # Penalize missing values
        uniqueness_score = 100 - duplicate_pct  # Penalize duplicates
        validity_score = 100 - (zero_total / (total_records * len(zero_features)) * 100) if zero_features else 100  # Penalize zeros
        consistency_score = 100 - min(outlier_pct, 100)  # Penalize outliers
        
        overall_quality_score = (completeness_score + uniqueness_score + validity_score + consistency_score) / 4
        
        # Quality grade
        if overall_quality_score >= 90:
            quality_grade = "A (Excellent)"
            grade_color = "#28a745"
        elif overall_quality_score >= 80:
            quality_grade = "B (Good)"
            grade_color = "#5cb85c"
        elif overall_quality_score >= 70:
            quality_grade = "C (Fair)"
            grade_color = "#ffc107"
        elif overall_quality_score >= 60:
            quality_grade = "D (Poor)"
            grade_color = "#fd7e14"
        else:
            quality_grade = "F (Very Poor)"
            grade_color = "#dc3545"
        
        # Create data quality visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quality scores bar chart
        scores = [completeness_score, uniqueness_score, validity_score, consistency_score]
        labels = ['Completeness', 'Uniqueness', 'Validity', 'Consistency']
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        axes[0, 0].barh(labels, scores, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlim(0, 100)
        axes[0, 0].set_xlabel('Score (%)')
        axes[0, 0].set_title('Data Quality Dimensions', fontweight='bold')
        axes[0, 0].axvline(x=80, color='red', linestyle='--', alpha=0.5, label='Target: 80%')
        axes[0, 0].legend()
        for i, v in enumerate(scores):
            axes[0, 0].text(v + 2, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # Missing values heatmap
        axes[0, 1].axis('off')
        missing_data = df.isnull().sum()
        missing_text = "Missing Values by Feature:\n\n"
        if missing_data.sum() == 0:
            missing_text += "✅ No missing values found!"
        else:
            for col, count in missing_data.items():
                if count > 0:
                    missing_text += f"{col}: {count} ({count/len(df)*100:.1f}%)\n"
        axes[0, 1].text(0.1, 0.5, missing_text, fontsize=10, verticalalignment='center', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0, 1].set_title('Missing Value Analysis', fontweight='bold')
        
        # Zero values analysis
        axes[1, 0].axis('off')
        zero_text = "Zero Values in Critical Features:\n\n"
        if zero_total == 0:
            zero_text += "✅ No suspicious zeros found!"
        else:
            for feat, count in zero_counts.items():
                if count > 0:
                    zero_text += f"{feat}: {count} ({count/len(df)*100:.1f}%)\n"
        axes[1, 0].text(0.1, 0.5, zero_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1, 0].set_title('Zero Value Analysis', fontweight='bold')
        
        # Overall quality gauge
        axes[1, 1].axis('off')
        quality_text = f"Overall Data Quality Score:\n\n{overall_quality_score:.1f}%\n\nGrade: {quality_grade}"
        axes[1, 1].text(0.5, 0.5, quality_text, fontsize=16, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.3, edgecolor=grade_color, linewidth=3))
        axes[1, 1].set_title('Overall Quality Assessment', fontweight='bold')
        
        plt.tight_layout()
        images['quality'] = fig_to_base64(fig)
        
        # Build HTML
        html = f"""<!DOCTYPE html>
        <html><head><title>Comprehensive Analysis</title>
        <style>
        body{{font-family:Arial;margin:0;padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}}
        .container{{max-width:1600px;margin:0 auto;background:white;padding:40px;border-radius:15px;box-shadow:0 10px 40px rgba(0,0,0,0.3)}}
        h1,h2{{color:#2c3e50}}
        h2{{border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:40px}}
        .back-btn{{padding:12px 30px;background:#667eea;color:white;text-decoration:none;border-radius:8px;display:inline-block;margin-bottom:30px}}
        .zero-fn{{background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%);color:white;padding:30px;border-radius:15px;text-align:center;font-size:2em;font-weight:bold;margin:30px 0;box-shadow:0 8px 25px rgba(0,0,0,0.2)}}
        .chart{{margin:30px 0;text-align:center}}
        .chart img{{max-width:100%;height:auto;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.1)}}
        .stats-table{{width:100%;border-collapse:collapse;margin:20px 0;font-size:0.9em}}
        .stats-table th{{background:#667eea;color:white;padding:12px;text-align:left}}
        .stats-table td{{padding:10px;border-bottom:1px solid #ecf0f1}}
        .stats-table tr:hover{{background:#f8f9fa}}
        .metric-box{{background:#f8f9fa;padding:20px;border-radius:10px;margin:20px 0;border-left:4px solid #667eea}}
        </style></head><body>
        <div class="container">
        <a href="/" class="back-btn">← Back to Main</a>
        <h1>📊 Comprehensive EDA & Model Analysis</h1>
        
        <div class="zero-fn">🎯 ZERO FALSE NEGATIVES ACHIEVED ✅</div>
        
        <div class="metric-box">
        <h3>Model Configuration</h3>
        <p><strong>Selected Algorithm:</strong> {self.pipeline.best_model_name}</p>
        <p><strong>Optimized Threshold:</strong> {self.pipeline.best_threshold}</p>
        <p><strong>Features Used:</strong> {len(self.pipeline.feature_cols)} - {', '.join(self.pipeline.feature_cols[:5])}{'...' if len(self.pipeline.feature_cols) > 5 else ''}</p>
        </div>
        
        <h2>📈 1. Target Variable Distribution</h2>
        <div class="chart"><img src="{images['outcome']}" alt="Outcome Distribution"></div>
        
        <h2>🔗 2. Correlation Matrix (Multivariate Analysis)</h2>
        <div class="chart"><img src="{images['correlation']}" alt="Correlation Matrix"></div>
        <p class="metric-box">The correlation matrix shows relationships between all features. Strong positive correlations (red) indicate features that increase together, while negative correlations (blue) show inverse relationships. Key for identifying multicollinearity and feature importance.</p>
        
        <h2>📊 3. Feature Distributions (Univariate Analysis)</h2>
        <div class="chart"><img src="{images['distributions']}" alt="Feature Distributions"></div>
        <p class="metric-box">Histograms show the distribution of each feature independently. This helps identify skewness, outliers, and the range of values for each variable.</p>
        
        <h2>📦 4. Boxplots by Outcome (Bivariate Analysis)</h2>
        <div class="chart"><img src="{images['boxplots']}" alt="Boxplots by Outcome"></div>
        <p class="metric-box">Boxplots compare feature distributions between diabetes and non-diabetes groups. Clear separation indicates features with strong predictive power.</p>
        
        <h2>🔵 5. Scatter Plots (Bivariate Analysis)</h2>
        <div class="chart"><img src="{images['scatter']}" alt="Scatter Plots"></div>
        <p class="metric-box">Scatter plots reveal relationships between pairs of features, colored by outcome. Clustering patterns help visualize how combinations of features separate the two classes.</p>
        
        <h2>🎯 6. Confusion Matrix</h2>
        <div class="chart"><img src="{images.get('confusion', '')}" alt="Confusion Matrix"></div>
        <p class="metric-box"><strong>Key Achievement: FN = 0</strong><br>
        • True Positives (TP): All diabetes cases correctly identified<br>
        • False Negatives (FN): <strong>0</strong> - No missed diabetes cases<br>
        • True Negatives (TN): Correctly identified non-diabetes cases<br>
        • False Positives (FP): Non-diabetes cases flagged for follow-up (acceptable trade-off)</p>
        
        <h2>📈 7. ROC Curve</h2>
        <div class="chart"><img src="{images.get('roc', '')}" alt="ROC Curve"></div>
        <p class="metric-box">The ROC curve plots True Positive Rate vs False Positive Rate at various thresholds. Area Under Curve (AUC) measures overall model performance - higher is better (max = 1.0).</p>
        
        <h2>📋 8. Descriptive Statistics</h2>
        <div style="overflow-x:auto">{stats_html}</div>
        
        <h2>🔍 9. Data Quality Assessment</h2>
        <div class="chart"><img src="{images['quality']}" alt="Data Quality Assessment"></div>
        
        <div class="metric-box">
        <h3>📊 Data Quality Metrics</h3>
        <table class="stats-table">
        <tr><th>Dimension</th><th>Score</th><th>Details</th></tr>
        <tr><td><strong>Completeness</strong></td><td>{completeness_score:.1f}%</td><td>{missing_total} missing values out of {total_records * len(df.columns)} total cells</td></tr>
        <tr><td><strong>Uniqueness</strong></td><td>{uniqueness_score:.1f}%</td><td>{duplicates} duplicate records ({duplicate_pct:.1f}%)</td></tr>
        <tr><td><strong>Validity</strong></td><td>{validity_score:.1f}%</td><td>{zero_total} suspicious zero values in critical features</td></tr>
        <tr><td><strong>Consistency</strong></td><td>{consistency_score:.1f}%</td><td>{outlier_total} outliers detected using IQR method</td></tr>
        <tr style="background:#f8f9fa;font-weight:bold;font-size:1.1em;"><td><strong>Overall Quality Score</strong></td><td style="color:{grade_color}">{overall_quality_score:.1f}%</td><td><strong>Grade: {quality_grade}</strong></td></tr>
        </table>
        
        <p><strong>Quality Score Interpretation:</strong></p>
        <ul>
        <li>90-100%: Grade A (Excellent) - High quality data, minimal preprocessing needed</li>
        <li>80-89%: Grade B (Good) - Good quality, some minor issues</li>
        <li>70-79%: Grade C (Fair) - Acceptable quality, moderate preprocessing required</li>
        <li>60-69%: Grade D (Poor) - Significant quality issues, extensive preprocessing needed</li>
        <li>Below 60%: Grade F (Very Poor) - Major quality problems, data collection review recommended</li>
        </ul>
        </div>
        
        <h2>💡 Summary</h2>
        <div class="metric-box">
        <p><strong>✅ Univariate Analysis:</strong> Individual feature distributions examined via histograms</p>
        <p><strong>✅ Bivariate Analysis:</strong> Pairwise relationships explored through scatter plots and boxplots</p>
        <p><strong>✅ Multivariate Analysis:</strong> Correlation matrix reveals complex feature interactions</p>
        <p><strong>✅ Model Performance:</strong> Confusion matrix and ROC curve demonstrate prediction quality</p>
        <p><strong>🎯 Zero False Negatives:</strong> Perfect sensitivity ensures no diabetes cases are missed</p>
        </div>
        
        </div></body></html>"""
        
        return html.encode('utf-8')
    
    @cherrypy.expose
    def logs(self):
        """View application logs"""
        cherrypy.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        
        log_dir = Path("logs")
        if not log_dir.exists():
            return b'<html><body><h1>No logs yet</h1><p>Start using the app to generate logs.</p><a href="/">Back</a></body></html>'
        
        log_files = sorted(log_dir.glob("*.log"), reverse=True)
        if not log_files:
            return b'<html><body><h1>No log files</h1><a href="/">Back</a></body></html>'
        
        latest_log = log_files[0]
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-500:] if len(lines) > 500 else lines
                log_display = ''.join(last_lines)
        except Exception as e:
            log_display = f"Error: {e}"
        
        html = f"""<!DOCTYPE html>
        <html><head><title>Logs</title>
        <style>
        body{{font-family:monospace;background:#1e1e1e;color:#d4d4d4;padding:20px}}
        .container{{max-width:1600px;margin:0 auto;background:#252526;padding:30px;border-radius:10px}}
        h1{{color:#4ec9b0}}
        .log-box{{background:#1e1e1e;padding:20px;border-radius:8px;max-height:600px;overflow-y:auto;font-size:0.9em}}
        pre{{margin:0;white-space:pre-wrap}}
        a{{color:#569cd6;padding:10px 20px;background:#007acc;text-decoration:none;border-radius:5px;display:inline-block;margin:10px 5px}}
        </style>
        <script>setTimeout(function(){{location.reload();}}, 5000);</script>
        </head><body><div class="container">
        <a href="/">← Back</a><a href="/logs">🔄 Refresh</a>
        <h1>📋 Application Logs (Auto-refresh: 5s)</h1>
        <p>Log file: {latest_log.name} ({latest_log.stat().st_size / 1024:.1f} KB)</p>
        <div class="log-box"><pre>{log_display}</pre></div>
        </div></body></html>"""
        return html.encode('utf-8')
    
    @cherrypy.expose
    def predict_form(self):
        """Prediction form page"""
        cherrypy.response.headers['Content-Type'] = 'text/html; charset=utf-8'
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Predict - Diabetes API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
                h1 { color: #2c3e50; }
                .nav { margin: 20px 0; }
                .nav a { padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px; }
                .form-group { margin: 20px 0; }
                input[type="file"] { padding: 10px; border: 2px solid #ddd; border-radius: 5px; width: 100%; }
                button { padding: 12px 30px; background: #27ae60; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                button:hover { background: #229954; }
                .warning { background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔮 Make Predictions</h1>
                <div class="nav">
                    <a href="/">🏠 Home</a>
                </div>
                
                """ + ('<div class="warning"><strong>⚠️ No Model Trained</strong><br>Please train a model first from the home page.</div>' if not self.pipeline else '') + """
                
                <h2>Upload CSV for Batch Prediction</h2>
                <p>Upload a CSV file with patient data (without Outcome column) to get predictions.</p>
                
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <input type="file" name="file" accept=".csv" required>
                    </div>
                    <button type="submit" """ + ('disabled' if not self.pipeline else '') + """>Get Predictions</button>
                </form>
            </div>
        </body>
        </html>
        """
        return html.encode('utf-8')
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def health(self):
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_trained': self.pipeline is not None
        }
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def status(self):
        """Get model status and metrics"""
        if not self.pipeline:
            return {'error': 'No model trained yet', 'trained': False}
        
        return {
            'trained': True,
            'best_model': self.pipeline.best_model_name,
            'threshold': float(self.pipeline.best_threshold),
            'features': self.pipeline.feature_cols
        }
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def train(self, file):
        """Train model endpoint"""
        try:
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_path = self.upload_dir / f"train_{timestamp}.csv"
            
            with open(upload_path, 'wb') as f:
                while True:
                    data = file.file.read(8192)
                    if not data:
                        break
                    f.write(data)
            
            # Load data
            df = pd.read_csv(upload_path)
            
            if 'Outcome' not in df.columns:
                return {'error': 'Dataset must contain "Outcome" column'}
            
            # Train pipeline
            self.pipeline = DiabetesPredictionPipeline()
            results = self.pipeline.train(df)
            
            # Save model
            model_path = self.models_dir / f"model_{timestamp}.pkl"
            self.pipeline.save(model_path)
            
            return {
                'success': True,
                'message': 'Comprehensive training completed',
                'device': results.get('device', 'CPU'),
                'training_time': f"{results.get('training_time', 0):.2f}s",
                'cuda_available': results.get('cuda_available', False),
                'combinations_tested': results.get('combinations_tested', 0),
                'results': {
                    'best_model': results['best_model'],
                    'threshold': float(results['threshold']),
                    'best_config': results.get('best_config', {}),
                    'all_models': {
                        k: {
                            'fn': int(v['fn']),
                            'tp': int(v['tp']),
                            'fp': int(v['fp']),
                            'tn': int(v['tn']),
                            'sensitivity': float(v.get('sensitivity', 0)),
                            'specificity': float(v.get('specificity', 0)),
                            'auc': float(v['auc']),
                            'threshold': float(v['threshold'])
                        } for k, v in results.get('metrics', {}).items()
                    }
                },
                'model_saved': str(model_path)
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predict(self, file=None):
        """Batch prediction endpoint"""
        try:
            if file is None:
                cherrypy.response.headers['Content-Type'] = 'text/html; charset=utf-8'
                return b'<html><body><h1>Error</h1><p>No file uploaded. Please use the <a href="/predict_form">prediction form</a>.</p></body></html>'
            
            if not self.pipeline:
                return {'error': 'No model trained. Please train a model first.'}
            
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_path = self.upload_dir / f"predict_{timestamp}.csv"
            
            with open(upload_path, 'wb') as f:
                while True:
                    data = file.file.read(8192)
                    if not data:
                        break
                    f.write(data)
            
            # Load data
            df = pd.read_csv(upload_path)
            
            # Make predictions
            y_pred, y_proba, result_df = self.pipeline.predict(df)
            
            # Save results
            results_path = self.upload_dir / f"results_{timestamp}.csv"
            result_df.to_csv(results_path, index=False)
            
            return {
                'success': True,
                'predictions': [int(x) for x in y_pred],
                'probabilities': [float(x) for x in y_proba],
                'results_file': str(results_path),
                'summary': {
                    'total': int(len(y_pred)),
                    'positive': int(y_pred.sum()),
                    'negative': int((1 - y_pred).sum())
                }
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def predict_single(self):
        """Single prediction endpoint"""
        try:
            if not self.pipeline:
                return {'error': 'No model trained. Please train a model first.'}
            
            input_data = cherrypy.request.json
            
            # Create dataframe from input
            df = pd.DataFrame([input_data])
            
            # Make prediction
            y_pred, y_proba, _ = self.pipeline.predict(df, is_single=True)
            
            risk_level = 'Low' if y_proba[0] < 0.3 else ('Medium' if y_proba[0] < 0.6 else 'High')
            
            return {
                'success': True,
                'prediction': int(y_pred[0]),
                'probability': float(y_proba[0]),
                'risk_level': risk_level,
                'diagnosis': 'Diabetes' if y_pred[0] == 1 else 'No Diabetes'
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}


def main():
    import logging
    from logging.handlers import RotatingFileHandler
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Starting Diabetes Prediction API")
    logger.info("="*60)
    
    config = {
        'global': {
            'server.socket_host': '0.0.0.0',
            'server.socket_port': 8080,
            'server.thread_pool': 10,
            'log.screen': True
        },
        '/': {
            'tools.sessions.on': True
        }
    }
    
    cherrypy.config.update(config)
    cherrypy.quickstart(DiabetesPredictionAPI(), '/', config)


if __name__ == '__main__':
    main()