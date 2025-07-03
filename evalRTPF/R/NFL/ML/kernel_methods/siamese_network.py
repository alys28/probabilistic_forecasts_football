import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.5):  # Smaller margin for cosine similarity
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         label = label.float()
#         # label: 1 if similar, 0 if dissimilar
        
#         # Cosine similarity (ranges from -1 to 1, where 1 = identical)
#         cosine_sim = F.cosine_similarity(output1, output2, dim=1)
        
#         # Convert to distance (0 = identical, 2 = opposite)
#         cosine_distance = 1 - cosine_sim
        
#         # Contrastive loss with cosine distance
#         loss = (label) * torch.pow(cosine_distance, 2) + \
#                (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
#         return loss.mean()


class NFLDataset(Dataset):
    def __init__(self, data_x, data_y, max_pairs_per_sample=100, device='cpu'):
        # Store device for tensor creation
        self.device = device
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.LongTensor(data_y)
        self.pairs = []
        
        # Simple win/loss classification based on final score difference
        def get_game_pattern(final_score_diff):
            # Classify based on final score difference and margin
            # Positive score_diff = win, Negative = loss
            # Margin <= 7 = close, > 7 = big
            
            if final_score_diff > 0:
                return "win"
            else:
                return "loss"
        # Group samples by game pattern
        category_groups = {}
        for i, y in enumerate(self.data_y):
            game_seq = self.data_x[i]
            cat = get_game_pattern(y.item())
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(i)
        
        # Create balanced pairs: both positive (similar) and negative (dissimilar) pairs
        random.seed(42)  # For reproducibility
        
        for i in range(len(self.data_x)):
            game_seq = self.data_x[i]
            current_cat = get_game_pattern(self.data_y[i].item())
            pairs_created = 0
            
            # Create positive pairs (same broad category only)
            positive_candidates = []
            if current_cat in category_groups:
                positive_candidates = [j for j in category_groups[current_cat] if j != i]
            
            # Sample positive pairs
            num_positive = min(max_pairs_per_sample // 2, len(positive_candidates))
            if positive_candidates:
                positive_samples = random.sample(positive_candidates, num_positive)
                for j in positive_samples:
                    self.pairs.append((i, j))
                    pairs_created += 1
            
            # Create negative pairs (different broad categories)
            negative_candidates = []
            for cat, indices in category_groups.items():
                if cat != current_cat:  # Different broad categories
                    negative_candidates.extend([j for j in indices if j != i])
            
            # Sample negative pairs
            num_negative = min(max_pairs_per_sample - pairs_created, len(negative_candidates))
            if negative_candidates:
                negative_samples = random.sample(negative_candidates, num_negative)
                for j in negative_samples:
                    self.pairs.append((i, j))
        
        # Shuffle pairs for better training
        random.shuffle(self.pairs)
        print(f"Created {len(self.pairs)} balanced pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns a pair of data points and a label. The label is 1 if the two data points have similar score differences, 0 otherwise.
        """
        i, j = self.pairs[idx]
        x1 = self.data_x[i]
        x2 = self.data_x[j]
        y1 = self.data_y[i]
        y2 = self.data_y[j]
        # Get score differences for both games (keep the sign!)
        score_diff_1 = y1.item()
        score_diff_2 = y2.item()
        
        # Simple win/loss classification based on final score difference
        def get_game_pattern(final_score_diff):
            # Classify based on final score difference and margin
            # Positive score_diff = win, Negative = loss
            # Margin <= 7 = close, > 7 = big
            
            if final_score_diff > 0:
                return "win"
            else:
                return "loss"
        
        # Pattern similarity: 1 if same game outcome pattern, 0 otherwise
        cat1 = get_game_pattern(score_diff_1)
        cat2 = get_game_pattern(score_diff_2)
        
        # Games are similar only if they follow the same outcome pattern:
        # big_win, close_win, close_loss, or big_loss
        label = 1 if cat1 == cat2 else 0
        return x1.to(self.device), x2.to(self.device), torch.FloatTensor([label]).to(self.device)
    
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, head_output_dim=64):
        super(SiameseNetwork, self).__init__()
        # Moderate complexity feature extraction head
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim // 2, head_output_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(head_output_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward_one(self, x):
        """
        x: [batch, input_dim] - 2D input expected by Linear layers
        """
        # Pass through the head network
        x = self.head(x)  # [batch, output_dim]
        return x
        
    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        product = torch.mul(x1, x2)
        out = self.cls_head(product)
        return out

class SiameseClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None):
        """
        Initializes the SiameseClassifier.
        """
        self.model = model.to(device)  # Ensure model is on correct device
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion.to(device)  # Ensure criterion is on correct device
        self.device = device
        self.scheduler = scheduler
        # Track training metrics
        self.final_train_loss = None
        self.final_train_accuracy = None
        self.final_val_loss = None
        self.final_val_accuracy = None
        self.best_epoch = None
        self.epochs_trained = None
    
    def fit(self, X, y, val_X = None, val_y = None, batch_size = 128):
        """
        Fits the SiameseClassifier to the data.
        Args:
            X: np.array of shape (n_samples, n_features)
            y: np.array of shape (n_samples,)
            val_X: np.array of shape (n_val_samples, n_features)
            val_y: np.array of shape (n_val_samples,)
            score_difference_index: int, index of the score difference feature in X
        """
        train_dataset = NFLDataset(X, y, max_pairs_per_sample=15, device=self.device)  # More pairs for better generalization

        print("Data loaded!")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_X is not None and val_y is not None:
            val_dataset = NFLDataset(val_X, val_y, max_pairs_per_sample=10, device=self.device)  # More validation pairs for better evaluation
            # Use the same batch size for validation to avoid batch norm issues
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training with scheduler and early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 5  # Aggressive early stopping to prevent overfitting
        best_epoch = 0
        
        print(f"Starting training on device: {self.device}")
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                x1, x2, y = batch
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                
                # Debug device info for first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"Input tensors device: x1={x1.device}, x2={x2.device}, y={y.device}")
                    print(f"Model parameters device: {next(self.model.parameters()).device}")
                
                # Check for NaN in inputs
                if torch.isnan(x1).any() or torch.isnan(x2).any():
                    print("NaN in inputs")
                    continue
                    
                output = self.model(x1, x2)
                
                # Check for NaN in outputs
                if torch.isnan(output).any():
                    print("NaN in outputs")
                    continue
                    
                self.optimizer.zero_grad()
                loss = self.criterion(output, y)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print("NaN in loss")
                    continue
                    
                loss.backward()
                
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                with torch.no_grad():
                   predictions = (output > 0.5).float()
                   train_correct += (predictions == y).float().sum().item()
                   train_total += len(predictions)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Validation
            if val_loader is not None:
                val_loss, val_accuracy = self._evaluate(val_loader)
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()  # Save best model
                    best_epoch = epoch + 1
                    # Store best metrics
                    self.final_val_loss = val_loss
                    self.final_val_accuracy = val_accuracy
                    self.final_train_loss = avg_train_loss
                    self.final_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)  # Restore best model
                            print(f"Restored model from best epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}")
                # Save model state even without validation for consistency
                if best_model_state is None:
                    best_model_state = self.model.state_dict().copy()
                # Store final metrics for no-validation case
                self.final_train_loss = avg_train_loss
                self.final_train_accuracy = train_accuracy
                best_epoch = epoch + 1
        
        # Store final training info
        self.best_epoch = best_epoch
        self.epochs_trained = epoch + 1
    
    def _evaluate(self, data_loader):
        """Helper method for evaluation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                x1, x2, y = batch
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                
                # Skip batch if inputs contain NaN
                if torch.isnan(x1).any() or torch.isnan(x2).any():
                    continue
                    
                output = self.model(x1, x2)
                
                # Skip batch if outputs contain NaN
                if torch.isnan(output).any():
                    print("NaN in Evaluation")
                    continue
                    
                loss = self.criterion(output, y)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    print("NaN Loss in Evaluation")
                    continue
                    
                total_loss += loss.item()
                
                predictions = (output > 0.5).float()
                correct += (predictions == y).float().sum().item()
                total_samples += len(predictions)
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def predict(self, x1, x2):
        """
        Predicts the similarity between two inputs based on cosine similarity.
        """
        self.model.eval()
        with torch.no_grad():
            # Ensure input tensors are on the same device as the model
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            output = self.model(x1, x2)
            return output
    
    def save_model(self, filepath_prefix, timesteps_range):
        """
        Save the trained model with informative filename including metrics.
        
        Args:
            filepath_prefix: Base path and prefix for the saved model
            timesteps_range: List [start_time, end_time] for the timestep range
        """
        # Create informative filename
        # Format metrics for filename (avoid decimals in filename)
        val_acc = int(self.final_val_accuracy * 10000) if self.final_val_accuracy else 0
        val_loss = int(self.final_val_loss * 10000) if self.final_val_loss else 0
        
        # Build filename with timesteps and metrics
        filename = f"{filepath_prefix}_ep{self.best_epoch}"
        filename += f"_valAcc{val_acc}_valLoss{val_loss}.pth"
        
        # Save model state dict and training info
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps_range': timesteps_range,
            'best_epoch': self.best_epoch,
            'epochs_trained': self.epochs_trained,
            'final_train_loss': self.final_train_loss,
            'final_train_accuracy': self.final_train_accuracy,
            'final_val_loss': self.final_val_loss,
            'final_val_accuracy': self.final_val_accuracy,
            'model_config': {
                'input_dim': self.model.head[0].in_features,
                'hidden_dim': self.model.head[-1].out_features,
                'output_dim': self.model.head[-1].out_features
            }
        }
        
        torch.save(checkpoint, filename)
        print(f"Model saved: {filename}")
        return filename
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved SiameseClassifier model.
        
        Args:
            filepath: Path to the saved model file
            device: Device to load the model on (default: auto-detect)
        
        Returns:
            SiameseClassifier: Loaded model instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate the model architecture
        model_config = checkpoint['model_config']
        siamese_network = SiameseNetwork(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim']
        )
        
        # Load model state
        siamese_network.load_state_dict(checkpoint['model_state_dict'])
        siamese_network.to(device)
        
        # Create classifier instance (dummy optimizer/criterion for loading)
        criterion = nn.BCELoss() 
        optimizer = torch.optim.AdamW(siamese_network.parameters(), lr=0.001)
        
        # Note: siamese_network is already moved to device above
        classifier = cls(siamese_network, 1, optimizer, criterion, device)
        
        # Restore training metrics
        classifier.final_train_loss = checkpoint.get('final_train_loss')
        classifier.final_train_accuracy = checkpoint.get('final_train_accuracy')
        classifier.final_val_loss = checkpoint.get('final_val_loss')
        classifier.final_val_accuracy = checkpoint.get('final_val_accuracy')
        classifier.best_epoch = checkpoint.get('best_epoch')
        classifier.epochs_trained = checkpoint.get('epochs_trained')
        
        print(f"Model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}")
        
        return classifier

