"""
STAGE 5 CONTINUATION: EXACT REPLICATION OF ORIGINAL OPTIMIZATION
================================================================

This script exactly replicates the original optimization approach:
- Only Autoencoder (no LSTM, no Isolation Forest)
- Same hyperparameter space as the original 4-day run
- Continue from Client 1 patient-based split
- Same architecture and training approach

Based on the original terminal output showing:
- Client 4 temporal: F1=0.4438 (encoding_dim=64, lr=0.001, dropout=0.1, batch=32, epochs=150)
- Client 0 patient-based: F1=0.3686 (encoding_dim=64, lr=0.0001, dropout=0.0, batch=128, epochs=100)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import traceback
import time
import json
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import joblib

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# =============================================================================
# CONFIGURATION - EXACT SAME AS ORIGINAL
# =============================================================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Research configuration
TARGET_F1 = 0.25  # Research target
EXCELLENT_F1 = 0.30  # Excellent performance threshold
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# =============================================================================
# AUTOENCODER ARCHITECTURE - EXACT SAME AS ORIGINAL
# =============================================================================

class OptimizedAutoencoder(nn.Module):
    """
    Exact same autoencoder architecture as the original optimization
    """
    
    def __init__(self, input_dim, encoding_dim=32, dropout_rate=0.1, use_batch_norm=True):
        super(OptimizedAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Encoder layers - EXACT SAME ARCHITECTURE
        encoder_layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]
        
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(128))
        
        encoder_layers.extend([
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(64))
        
        encoder_layers.extend([
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers - EXACT SAME ARCHITECTURE
        decoder_layers = [
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]
        
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm1d(64))
        
        decoder_layers.extend([
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm1d(128))
        
        decoder_layers.extend([
            nn.Linear(128, input_dim)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# =============================================================================
# OPTIMIZATION CLASS - EXACT SAME APPROACH
# =============================================================================

class ExactContinuationOptimizer:
    """
    Exact continuation of the original optimization
    """
    
    def __init__(self):
        self.device = device
        self.results = []
        self.best_global_f1 = 0.0
        self.best_global_params = None
        self.best_global_client = None
        
        # EXACT SAME HYPERPARAMETER SPACE AS ORIGINAL
        self.autoencoder_params = {
            'encoding_dim': [16, 32, 64],  # Same as original
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],  # Same as original
            'dropout_rate': [0.0, 0.1, 0.2, 0.3],  # Same as original
            'batch_size': [32, 64, 128],  # Same as original
            'epochs': [50, 100, 150],  # Same as original
            'use_batch_norm': [True, False]  # Same as original
        }
        
        # Create results directory
        self.results_dir = "models/optimized_models_patient_based_exact_continuation"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🚀 EXACT CONTINUATION OPTIMIZER INITIALIZED")
        print(f"📁 Results will be saved to: {self.results_dir}")
        print(f"🎯 Target F1-Score: {TARGET_F1}")
        print(f"🏆 Excellent F1-Score: {EXCELLENT_F1}")
        print(f"🔧 Hyperparameter combinations: {self._calculate_total_combinations()}")
    
    def _calculate_total_combinations(self):
        """Calculate total hyperparameter combinations"""
        total = 1
        for values in self.autoencoder_params.values():
            total *= len(values)
        return total
    
    def optimize_threshold(self, y_true, anomaly_scores):
        """
        Advanced threshold optimization using multiple strategies - EXACT SAME AS ORIGINAL
        
        Returns:
            tuple: (best_threshold, best_f1_score)
        """
        # Strategy 1: Percentile-based thresholds
        percentile_thresholds = np.percentile(anomaly_scores, np.arange(85, 99, 1))
        
        # Strategy 2: Statistical-based thresholds
        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        statistical_thresholds = [mean_score + i * std_score for i in range(1, 5)]
        
        # Strategy 3: Quantile-based thresholds
        quantile_thresholds = np.quantile(anomaly_scores, [0.90, 0.92, 0.94, 0.96, 0.98])
        
        # Combine all strategies
        all_thresholds = np.unique(np.concatenate([
            percentile_thresholds, 
            statistical_thresholds, 
            quantile_thresholds
        ]))
        
        best_f1 = 0
        best_threshold = None
        
        for threshold in all_thresholds:
            y_pred = (anomaly_scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    def load_data(self, client_id, split_type="patient_based"):
        """Load data for specific client and split type - EXACT SAME AS ORIGINAL"""
        try:
            # Load train data - EXACT SAME FILE PATHS
            train_path = f"processed_data/splits/{split_type}/fold_client_{client_id}/fold_client_{client_id}_train.csv"
            test_path = f"processed_data/splits/{split_type}/fold_client_{client_id}/fold_client_{client_id}_test.csv"
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print(f"❌ Data files not found for client {client_id}")
                return None, None, None, None
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Separate features and labels - EXACT SAME APPROACH
            feature_cols = [col for col in train_data.columns if col not in ['anomaly', 'timestamp', 'patient_id']]
            
            X_train = train_data[feature_cols].values
            y_train = train_data['anomaly'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['anomaly'].values
            
            print(f"✅ Loaded data for Client {client_id} ({split_type} split)")
            print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"   Features: {len(feature_cols)}")
            print(f"   Anomaly rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"❌ Error loading data for client {client_id}: {str(e)}")
            return None, None, None, None
    
    def train_autoencoder(self, X_train, y_train, X_test, y_test, hyperparams):
        """Train autoencoder with given hyperparameters - EXACT SAME AS ORIGINAL"""
        try:
            # Normalize data - EXACT SAME APPROACH
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert to tensors - EXACT SAME APPROACH
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            
            # Create data loaders - EXACT SAME APPROACH
            train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True)
            
            # Initialize model - EXACT SAME ARCHITECTURE
            model = OptimizedAutoencoder(
                input_dim=X_train.shape[1],
                encoding_dim=hyperparams['encoding_dim'],
                dropout_rate=hyperparams['dropout_rate'],
                use_batch_norm=hyperparams['use_batch_norm']
            ).to(self.device)
            
            # Initialize optimizer - EXACT SAME APPROACH
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5, verbose=False
            )
            
            # Training loop with early stopping - EXACT SAME APPROACH
            model.train()
            best_train_loss = float('inf')
            patience_counter = 0
            patience = 15
            
            for epoch in range(hyperparams['epochs']):
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                scheduler.step(avg_loss)
                
                # Early stopping - EXACT SAME AS ORIGINAL
                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                
                # Print progress every 25 epochs - EXACT SAME AS ORIGINAL
                if (epoch + 1) % 25 == 0:
                    print(f"    Epoch [{epoch+1}/{hyperparams['epochs']}], Loss: {avg_loss:.6f}")
            
            # Evaluation - EXACT SAME APPROACH
            model.eval()
            with torch.no_grad():
                reconstructed = model(X_test_tensor)
                mse_loss = F.mse_loss(reconstructed, X_test_tensor, reduction='none')
                anomaly_scores = mse_loss.mean(dim=1).cpu().numpy()
            
            # Optimize threshold - EXACT SAME APPROACH
            threshold, f1 = self.optimize_threshold(y_test, anomaly_scores)
            
            # Calculate all metrics - EXACT SAME APPROACH
            y_pred_binary = (anomaly_scores > threshold).astype(int)
            precision = precision_score(y_test, y_pred_binary, zero_division=0)
            recall = recall_score(y_test, y_pred_binary, zero_division=0)
            roc_auc = roc_auc_score(y_test, anomaly_scores)
            pr_auc = average_precision_score(y_test, anomaly_scores)
            
            return {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'threshold': threshold,
                'model': model,
                'scaler': scaler
            }
                
        except Exception as e:
            print(f"❌ Error training autoencoder: {str(e)}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def optimize_client(self, client_id, split_type="patient_based"):
        """Optimize hyperparameters for a specific client - EXACT SAME APPROACH"""
        print(f"\n🔄 Optimizing Client {client_id} ({split_type} split)")
        
        # Load data
        X_train, y_train, X_test, y_test = self.load_data(client_id, split_type)
        if X_train is None:
            return None
        
        # Generate all parameter combinations - EXACT SAME APPROACH
        param_names = list(self.autoencoder_params.keys())
        param_values = list(self.autoencoder_params.values())
        
        total_combinations = self._calculate_total_combinations()
        print(f"   Testing {total_combinations} hyperparameter combinations...")
        
        best_f1 = 0.0
        best_params = None
        best_model = None
        best_scaler = None
        combination_count = 0
        
        # Grid search - EXACT SAME APPROACH
        from itertools import product
        for param_combo in product(*param_values):
            combination_count += 1
            
            # Create hyperparameter dictionary
            hyperparams = dict(zip(param_names, param_combo))
            
            # Print progress every 50 combinations - EXACT SAME AS ORIGINAL
            if combination_count % 50 == 0:
                print(f"     Testing combination {combination_count}/{total_combinations}...")
            
            # Train model
            result = self.train_autoencoder(X_train, y_train, X_test, y_test, hyperparams)
            
            if result is not None and result['f1'] > best_f1:
                best_f1 = result['f1']
                best_params = hyperparams.copy()
                best_model = result['model']
                best_scaler = result['scaler']
                
                print(f"     🎯 New best F1: {best_f1:.4f}")
            
            # Clear GPU memory after each training - EXACT SAME AS ORIGINAL
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"   Best F1: {best_f1:.4f}")
        print(f"   Best hyperparameters: {best_params}")
        
        # Check if this is a new global best
        if best_f1 > self.best_global_f1:
            self.best_global_f1 = best_f1
            self.best_global_params = best_params.copy()
            self.best_global_client = client_id
            print(f"   🚀 NEW GLOBAL BEST F1: {best_f1:.4f}")
        
        # Save best model for this client
        if best_model is not None:
            client_dir = os.path.join(self.results_dir, f"client_{client_id}")
            os.makedirs(client_dir, exist_ok=True)
            
            # Save model and scaler
            torch.save(best_model.state_dict(), os.path.join(client_dir, f"optimized_autoencoder_client_{client_id}.pth"))
            joblib.dump(best_scaler, os.path.join(client_dir, f"optimized_scaler_client_{client_id}.joblib"))
            
            # Save hyperparameters
            with open(os.path.join(client_dir, f"best_hyperparameters_client_{client_id}.json"), 'w') as f:
                json.dump(best_params, f, indent=2)
        
        return {
            'client_id': client_id,
            'split_type': split_type,
            'best_f1': best_f1,
            'best_params': best_params,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'anomaly_rate_train': y_train.mean(),
            'anomaly_rate_test': y_test.mean()
        }
    
    def run_continuation(self):
        """Run continuation optimization for clients 1-4 - EXACT SAME APPROACH"""
        print("🚀 STARTING EXACT CONTINUATION OPTIMIZATION")
        print("=" * 60)
        
        # Continue from client 1 (client 0 was already completed in original)
        clients_to_process = [1, 2, 3, 4]
        
        for client_id in clients_to_process:
            try:
                result = self.optimize_client(client_id, "patient_based")
                if result is not None:
                    self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Optimization interrupted at client {client_id}")
                print("💾 Saving current results...")
                self.save_results()
                break
            except Exception as e:
                print(f"❌ Error processing client {client_id}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Final results
        self.print_final_results()
        return self.results
    
    def save_results(self):
        """Save optimization results"""
        if not self.results:
            return
        
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.results_dir, "exact_continuation_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_clients_processed': len(self.results),
            'best_global_f1': self.best_global_f1,
            'best_global_client': self.best_global_client,
            'best_global_params': self.best_global_params,
            'target_f1': TARGET_F1,
            'target_achieved': self.best_global_f1 >= TARGET_F1,
            'excellent_f1': EXCELLENT_F1,
            'excellent_achieved': self.best_global_f1 >= EXCELLENT_F1
        }
        
        summary_path = os.path.join(self.results_dir, "exact_continuation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
    
    def print_final_results(self):
        """Print final optimization results"""
        print("\n" + "=" * 60)
        print("🎉 EXACT CONTINUATION COMPLETED!")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        print(f"\n📊 FINAL RESULTS SUMMARY:")
        print(f"   Total clients processed: {len(self.results)}")
        print(f"   Best F1-Score: {self.best_global_f1:.4f}")
        print(f"   Best client: {self.best_global_client}")
        print(f"   Target F1-Score: {TARGET_F1}")
        print(f"   Target achieved: {'✅ YES' if self.best_global_f1 >= TARGET_F1 else '❌ NO'}")
        print(f"   Excellent F1-Score: {EXCELLENT_F1}")
        print(f"   Excellent achieved: {'✅ YES' if self.best_global_f1 >= EXCELLENT_F1 else '❌ NO'}")
        
        print(f"\n🏆 BEST HYPERPARAMETERS:")
        if self.best_global_params:
            for param, value in self.best_global_params.items():
                print(f"   {param}: {value}")
        
        print(f"\n📈 CLIENT-BY-CLIENT RESULTS:")
        for result in self.results:
            print(f"   Client {result['client_id']}: F1={result['best_f1']:.4f}")
        
        print(f"\n📁 Results saved to: {self.results_dir}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("🚀 STAGE 5 EXACT CONTINUATION: PATIENT-BASED SPLITS OPTIMIZATION")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target F1-Score: {TARGET_F1}")
    print(f"🏆 Excellent F1-Score: {EXCELLENT_F1}")
    print(f"💻 Device: {device}")
    print(f"🔧 Approach: EXACT SAME as original 4-day optimization")
    
    try:
        # Initialize optimizer
        optimizer = ExactContinuationOptimizer()
        
        # Run continuation optimization
        results = optimizer.run_continuation()
        
        print(f"\n✅ Exact continuation optimization completed successfully!")
        print(f"📊 Processed {len(results)} clients")
        print(f"🏆 Best F1-Score achieved: {optimizer.best_global_f1:.4f}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Optimization interrupted by user")
        print("💾 Results saved before interruption")
        return []
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    results = main()
