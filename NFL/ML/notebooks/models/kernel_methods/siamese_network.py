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
    def __init__(self, data_x, data_y, max_pairs_per_sample=25, device='cpu', add_noise=False, noise_std=0.01):
        # Store device for tensor creation
        self.device = device
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.LongTensor(data_y)
        self.pairs = []
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Score difference-based categorization
        def get_game_pattern(final_score_diff):
            # Categorize games by score difference magnitude
            abs_diff = abs(final_score_diff)
            if abs_diff <= 3:
                return "very_close"  # Field goal or less
            elif abs_diff <= 7:
                return "close"  # One touchdown
            elif abs_diff <= 14:
                return "moderate"  # Two touchdowns
            elif abs_diff <= 21:
                return "large"  # Three touchdowns
            else:
                return "blowout"  # More than three touchdowns
        
        # Group samples by game pattern categories
        category_groups = {}
        for i, y in enumerate(self.data_y):
            cat = get_game_pattern(y.item())
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(i)
        
        # Create pairs based on categorical similarity
        random.seed(42)  # For reproducibility
        
        for i in range(len(self.data_x)):
            current_cat = get_game_pattern(self.data_y[i].item())
            pairs_created = 0
            
            # Create positive pairs (same category)
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
            
            # Create negative pairs (different categories)
            negative_candidates = []
            for cat, indices in category_groups.items():
                if cat != current_cat:
                    negative_candidates.extend([j for j in indices if j != i])
            
            # Sample negative pairs - ensure balance
            num_negative = min(max_pairs_per_sample - pairs_created, len(negative_candidates))
            if negative_candidates:
                negative_samples = random.sample(negative_candidates, num_negative)
                for j in negative_samples:
                    self.pairs.append((i, j))
        
        # Shuffle pairs for better training
        random.shuffle(self.pairs)
        print(f"Created {len(self.pairs)} pairs based on categorical similarity")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        x1 = self.data_x[i]
        x2 = self.data_x[j]
        y1 = self.data_y[i]
        y2 = self.data_y[j]
        
        # Add noise for data augmentation if enabled
        if self.add_noise:
            noise1 = torch.randn_like(x1) * self.noise_std
            noise2 = torch.randn_like(x2) * self.noise_std
            x1 = x1 + noise1
            x2 = x2 + noise2
        
        score_diff_1 = y1.item()
        score_diff_2 = y2.item()
        
        def get_game_pattern(final_score_diff):
            # Categorize games by score difference magnitude
            abs_diff = abs(final_score_diff)
            if abs_diff <= 3:
                return "very_close"  # Field goal or less
            elif abs_diff <= 7:
                return "close"  # One touchdown
            elif abs_diff <= 14:
                return "moderate"  # Two touchdowns
            elif abs_diff <= 21:
                return "large"  # Three touchdowns
            else:
                return "blowout"  # More than three touchdowns
        
        # Calculate similarity based on categorical match
        cat1 = get_game_pattern(score_diff_1)
        cat2 = get_game_pattern(score_diff_2)
        
        # Binary similarity: 1 if same category, 0 if different
        label = 1 if cat1 == cat2 else 0
        return x1.to(self.device), x2.to(self.device), torch.FloatTensor([label]).to(self.device)
    
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, head_output_dim=8, dropout_rate=0.3):
        super(SiameseNetwork, self).__init__()
        # Store constructor parameters for saving/loading
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_output_dim = head_output_dim
        self.dropout_rate = dropout_rate
        
        # Simpler, shallower architecture with stronger regularization
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, head_output_dim)
        )
        
        # Initialize weights for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_one(self, x):
        x = self.head(x)
        # L2 normalize the output for better similarity computation
        x = F.normalize(x, p=2, dim=1)
        return x
        
    def forward(self, x1, x2):
        # Get normalized feature representations
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(x1, x2)
        
        # Convert cosine similarity from [-1, 1] to [0, 1] range
        similarity = (cosine_sim + 1) / 2
        
        # Clamp with small epsilon to preserve gradients
        eps = 1e-7
        similarity = torch.clamp(similarity, eps, 1.0 - eps)
        
        # Reshape to match expected output format [batch_size, 1]
        similarity = similarity.unsqueeze(1)
        
        return similarity

class SiameseClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None):
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        # Track training metrics
        self.final_train_loss = None
        self.final_train_accuracy = None
        self.final_val_loss = None
        self.final_val_accuracy = None
        self.best_epoch = None
        self.epochs_trained = None
    
    def fit(self, X, y, val_X=None, val_y=None, batch_size=64):  # Reduced batch size
        # Reduced max_pairs_per_sample to prevent overfitting
        train_dataset = NFLDataset(X, y, max_pairs_per_sample=30, device=self.device, 
                                 add_noise=True, noise_std=0.005)  # Added noise augmentation

        print("Data loaded!")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_X is not None and val_y is not None:
            val_dataset = NFLDataset(val_X, val_y, max_pairs_per_sample=25, device=self.device, 
                                   add_noise=False)  # No noise for validation
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # More aggressive early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 5  # Reduced patience from 10 to 5
        best_epoch = 0
        min_improvement = 1e-4  # Minimum improvement threshold
        
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
                
                # More aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
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
                
                # Learning rate scheduling based on validation loss
                if self.scheduler:
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                
                # More stringent early stopping
                if val_loss < best_val_loss - min_improvement:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
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
                            self.model.load_state_dict(best_model_state)
                            print(f"Restored model from best epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}")
                if best_model_state is None:
                    best_model_state = self.model.state_dict().copy()
                self.final_train_loss = avg_train_loss
                self.final_train_accuracy = train_accuracy
                best_epoch = epoch + 1
        
        # Store final training info
        self.best_epoch = best_epoch
        self.epochs_trained = epoch + 1
    
    def embed(self, data: torch.Tensor):
        """
        Embed a single data point.
        
        Args:
            data: torch.Tensor of shape (n_samples, input_dim)
            
        Returns:
            torch.Tensor of shape (n_samples, embedding_dim)
        """
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model.forward_one(data)


    def embed_data(self, data):
        """
        Embed data from dictionary format.
        
        Args:
            data: List of dictionaries with format {"row": np.array, "label": float}
            
        Returns:
            tuple: (embeddings, labels) where embeddings is a list of torch.Tensors of shape (n_samples, embedding_dim)
                   and labels is a list of original labels
        """
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for item in data:
                # Extract the row data and label from dictionary
                row_data = item["rows"]
                label = item["label"]
                
                # Convert numpy array to tensor and move to device
                if isinstance(row_data, np.ndarray):
                    x = torch.FloatTensor(row_data).to(self.device)
                else:
                    x = torch.FloatTensor(row_data).to(self.device)
                
                # Handle single sample (add batch dimension if needed)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                # Get embedding
                embedding = self.model.forward_one(x)
                embedding = embedding.squeeze(0)
                embeddings.append(embedding.cpu())  # Move back to CPU to save GPU memory
                labels.append(label)
            embeddings = torch.stack(embeddings)
        return embeddings, labels
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
        self.model.eval()
        with torch.no_grad():
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            output = self.model(x1, x2)
            return output
    
    def save_model(self, filepath_prefix, timesteps_range):
        val_acc = self.final_val_accuracy
        val_loss = self.final_val_loss
        
        filename = f"{filepath_prefix}_{timesteps_range[0]}-{timesteps_range[1]}_ep{self.best_epoch}"
        filename += f"_valAcc{val_acc:.3f}_valLoss{val_loss:.3f}.pth"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestep': timesteps_range[0],
            'val_loss': self.final_val_loss,
            'val_accuracy': self.final_val_accuracy,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'head_output_dim': self.model.head_output_dim,
                'dropout_rate': self.model.dropout_rate
            }
        }
        
        torch.save(checkpoint, filename)
        print(f"Model saved: {filename}")
        return filename
    
    @classmethod
    def load_model(cls, filepath, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate the model architecture
        model_config = checkpoint['model_config']
        siamese_network = SiameseNetwork(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            head_output_dim=model_config['head_output_dim'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # Load model state
        siamese_network.load_state_dict(checkpoint['model_state_dict'])
        siamese_network.to(device)
        
        # Create classifier instance
        criterion = nn.BCELoss() 
        optimizer = torch.optim.AdamW(siamese_network.parameters(), lr=0.001)
        
        classifier = cls(siamese_network, 1, optimizer, criterion, device)
        
        # Restore training metrics
        classifier.final_val_loss = checkpoint.get('val_loss')
        classifier.final_val_accuracy = checkpoint.get('val_accuracy')
        classifier.best_epoch = checkpoint.get('timestep')
        
        print(f"Model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}")
        
        return classifier

