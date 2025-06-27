import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.float()
        # label: 1 if similar, 0 if dissimilar
        # Add small epsilon to prevent sqrt of zero/negative numbers
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


class NFLDataset(Dataset):
    def __init__(self, data_x, data_y, max_pairs_per_sample=50):
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.LongTensor(data_y)
        self.pairs = []
        
        # Simple game flow analysis using actual features
        def get_game_pattern(game_sequence, final_score_diff):
            try:
                # Extract score_difference column (assuming it's index 0)
                if torch.is_tensor(game_sequence):
                    score_progression = game_sequence[:, 0].numpy()  # score_difference over time
                else:
                    score_progression = np.array(game_sequence)[:, 0]
                
                # Analyze how the score changed throughout the game
                early_score = score_progression[:len(score_progression)//3].mean()  # First third
                late_score = score_progression[-len(score_progression)//3:].mean()   # Last third
                
                final_margin = abs(final_score_diff)
                
                # Simple patterns based on score progression + final outcome:
                
                # Pattern 1: Close games (final margin <= 7)
                if final_margin <= 7:
                    return "close_game"
                
                # Pattern 2: Early lead held (similar early/late scores, big final margin)
                elif abs(early_score - late_score) < 7 and final_margin > 14:
                    return "wire_to_wire"
                
                # Pattern 3: Momentum shift (big difference between early/late scores)  
                elif abs(early_score - late_score) > 10:
                    return "momentum_shift"
                
                # Pattern 4: Standard blowout (big final margin)
                else:
                    return "blowout"
                    
            except:
                # Fallback to simple margin-based
                final_margin = abs(final_score_diff)
                if final_margin <= 7:
                    return "close_game"
                else:
                    return "blowout"
        
        # Group samples by game pattern
        category_groups = {}
        for i, y in enumerate(self.data_y):
            game_seq = self.data_x[i]
            cat = get_game_pattern(game_seq, y.item())
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(i)
        
        # Create balanced pairs: both positive (similar) and negative (dissimilar) pairs
        random.seed(42)  # For reproducibility
        
        for i in range(len(self.data_x)):
            game_seq = self.data_x[i]
            current_cat = get_game_pattern(game_seq, self.data_y[i].item())
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
        
        # Game pattern analysis using score progression + final outcome
        def get_game_pattern(game_sequence, final_score_diff):
            try:
                # Extract score_difference column (assuming it's index 0)
                if torch.is_tensor(game_sequence):
                    score_progression = game_sequence[:, 0].numpy()  # score_difference over time
                else:
                    score_progression = np.array(game_sequence)[:, 0]
                
                # Analyze how the score changed throughout the game
                early_score = score_progression[:len(score_progression)//3].mean()  # First third
                late_score = score_progression[-len(score_progression)//3:].mean()   # Last third
                
                final_margin = abs(final_score_diff)
                
                # Simple patterns based on score progression + final outcome:
                
                # Pattern 1: Close games (final margin <= 7)
                if final_margin <= 7:
                    return "close_game"
                
                # Pattern 2: Early lead held (similar early/late scores, big final margin)
                elif abs(early_score - late_score) < 7 and final_margin > 14:
                    return "wire_to_wire"
                
                # Pattern 3: Momentum shift (big difference between early/late scores)  
                elif abs(early_score - late_score) > 10:
                    return "momentum_shift"
                
                # Pattern 4: Standard blowout (big final margin)
                else:
                    return "blowout"
                    
            except:
                # Fallback to simple margin-based
                final_margin = abs(final_score_diff)
                if final_margin <= 7:
                    return "close_game"
                else:
                    return "blowout"
        
        # Pattern similarity: 1 if same game pattern, 0 otherwise
        cat1 = get_game_pattern(x1, score_diff_1)
        cat2 = get_game_pattern(x2, score_diff_2)
        
        # Games are similar only if they follow the same pattern:
        # close_game, wire_to_wire, momentum_shift, or blowout
        label = 1 if cat1 == cat2 else 0
        
        return x1, x2, torch.FloatTensor([label]).squeeze()
    
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=64):
        super(SiameseNetwork, self).__init__()
        # Balanced LSTM-based encoder for sequential data
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Moderate complexity feature extraction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward_one(self, x):
        """
        x: [batch, seq, input_dim]
        """
        # Single LSTM processes the sequence
        lstm_out, (hidden, _) = self.lstm(x)  # [batch, seq, hidden_dim]
        
        # Use the last hidden state as the sequence representation
        sequence_repr = hidden[-1]  # [batch, hidden_dim]
        
        # Pass through the head network
        x = self.head(sequence_repr)  # [batch, output_dim]
        
        # L2 normalization for stable training
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
        x = x / norm
        return x
        
    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)

def train_siamese_network(model, train_loader, epochs, optimizer, criterion, device, val_loader = None):
    # This function is now deprecated - training is handled in SiameseClassifier.fit()
    pass

def evaluate_siamese_network(model, test_loader, criterion, device):
    # This function is now deprecated - evaluation is handled in SiameseClassifier._evaluate()
    pass

class SiameseClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None):
        """
        Initializes the SiameseClassifier.
        """
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
    
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
        train_dataset = NFLDataset(X, y, max_pairs_per_sample=15)  # More pairs for better generalization

        print("Data loaded!")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_X is not None and val_y is not None:
            val_dataset = NFLDataset(val_X, val_y, max_pairs_per_sample=10)  # More validation pairs for better evaluation
            # Use the same batch size for validation to avoid batch norm issues
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training with scheduler and early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 5  # Aggressive early stopping to prevent overfitting
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                x1, x2, y = batch
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                
                # Check for NaN in inputs
                if torch.isnan(x1).any() or torch.isnan(x2).any():
                    continue
                    
                output1, output2 = self.model(x1, x2)
                
                # Check for NaN in outputs
                if torch.isnan(output1).any() or torch.isnan(output2).any():
                    continue
                    
                self.optimizer.zero_grad()
                loss = self.criterion(output1, output2, y)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                # Count the number of correct predictions based on distance threshold
                with torch.no_grad():
                    distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)
                    # Use half the contrastive loss margin as threshold (more principled)
                    threshold = self.criterion.margin / 2.0  # 1.0 if margin=2.0
                    predictions = (distances < threshold).float()  # 1 if similar, 0 if dissimilar
                    train_correct += (predictions == y.reshape(-1)).float().sum().item()
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
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)  # Restore best model
                            print(f"Restored model from best epoch with val_loss: {best_val_loss:.6f}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}")
                # Save model state even without validation for consistency
                if best_model_state is None:
                    best_model_state = self.model.state_dict().copy()
    
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
                    
                output1, output2 = self.model(x1, x2)
                
                # Skip batch if outputs contain NaN
                if torch.isnan(output1).any() or torch.isnan(output2).any():
                    continue
                    
                loss = self.criterion(output1, output2, y)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    continue
                    
                total_loss += loss.item()
                
                # Count the number of correct predictions based on distance threshold
                distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)
                # Use half the contrastive loss margin as threshold (more principled)
                threshold = self.criterion.margin / 2.0  # 1.0 if margin=2.0
                predictions = (distances < threshold).float()  # 1 if similar, 0 if dissimilar
                correct += (predictions == y).float().sum().item()
                total_samples += len(predictions)
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def predict(self, x1, x2):
        """
        Predicts the similarity between two inputs based on embedding distance.
        Returns distance (lower = more similar).
        """
        self.model.eval()
        with torch.no_grad():
            output1, output2 = self.model(x1, x2)
            distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            return distance

