import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class NFLSequenceDataset(Dataset):
    def __init__(self, data_x, data_y, sequence_length=None, max_pairs_per_sample=25, device='cpu'):
        """
        Dataset for LSTM Siamese Network
        
        Args:
            data_x: Input sequences of shape (n_samples, n_features) or (n_samples, seq_length, n_features)
            data_y: Target labels
            sequence_length: If data_x is 2D, reshape to (n_samples, sequence_length, n_features)
            max_pairs_per_sample: Maximum pairs per sample for contrastive learning
            device: Device to store tensors
        """
        self.device = device
        
        # Handle input reshaping for sequences
        if len(data_x.shape) == 2 and sequence_length is not None:
            # Reshape from (n_samples, n_features) to (n_samples, sequence_length, n_features_per_step)
            n_samples, total_features = data_x.shape
            features_per_step = total_features // sequence_length
            if total_features % sequence_length != 0:
                raise ValueError(f"Total features {total_features} not divisible by sequence_length {sequence_length}")
            data_x = data_x.reshape(n_samples, sequence_length, features_per_step)
        
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.LongTensor(data_y)
        self.pairs = []
        # Pre-compute similarities
        print("Computing similarity matrix LSTM pair construction...")
        n_samples = len(self.data_x)
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sim = self.calculate_similarity(self.data_y[i].item(), self.data_y[j].item())
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Create pairs with improved strategy
        random.seed(42)
        similarity_threshold = 0.6
        
        for i in range(len(self.data_x)):
            pairs_created = 0
            
            # Get similarity scores
            similarities = similarity_matrix[i, :]
            
            # Positive pairs: high similarity
            positive_candidates = [j for j in range(n_samples) 
                                 if j != i and similarities[j] >= similarity_threshold]
            
            if positive_candidates:
                positive_candidates.sort(key=lambda j: similarities[j], reverse=True)
                num_positive = min(max_pairs_per_sample // 2, len(positive_candidates))
                top_k = min(3, len(positive_candidates))
                selected = positive_candidates[:top_k]
                if len(positive_candidates) > top_k:
                    remaining = positive_candidates[top_k:]
                    selected.extend(random.sample(remaining, min(num_positive - top_k, len(remaining))))
                for j in selected:
                    self.pairs.append((i, j))
                    pairs_created += 1
            
            # Negative pairs: hard negative mining
            negative_candidates = [j for j in range(n_samples) 
                                 if j != i and similarities[j] < similarity_threshold]
            
            if negative_candidates:
                negative_similarities = [(j, similarities[j]) for j in negative_candidates]
                negative_similarities.sort(key=lambda x: x[1])
                
                # Hard negatives: similarity 0.2 to 0.5
                hard_negatives = [j for j, sim in negative_similarities if 0.2 <= sim < 0.5]
                easy_negatives = [j for j, sim in negative_similarities if sim < 0.2]
                
                num_negative = min(max_pairs_per_sample - pairs_created, len(negative_candidates))
                if hard_negatives:
                    num_hard = min(num_negative * 2 // 3, len(hard_negatives))
                    selected_neg = random.sample(hard_negatives, num_hard)
                    if len(easy_negatives) > 0 and num_negative > num_hard:
                        selected_neg.extend(random.sample(easy_negatives, 
                                                          min(num_negative - num_hard, len(easy_negatives))))
                else:
                    selected_neg = random.sample(negative_candidates, 
                                                 min(num_negative, len(negative_candidates)))
                
                for j in selected_neg:
                    self.pairs.append((i, j))
        
        # Shuffle pairs for better training
        random.shuffle(self.pairs)
        print(f"Created {len(self.pairs)} pairs")
        print(f"Sequence shape: {self.data_x.shape}")

    def calculate_similarity(self, score_diff_1, score_diff_2):
        """
        Prioritizes win/loss alignment; smooths by margin similarity.
        """
        import math
        same_outcome = (score_diff_1 >= 0) == (score_diff_2 >= 0)
        abs_diff_1, abs_diff_2 = abs(score_diff_1), abs(score_diff_2)

        # Strong base similarity if both wins or both losses
        base = 0.8 if same_outcome else 0.0

        # Margin similarity within same outcome
        margin_similarity = math.exp(-abs(abs_diff_1 - abs_diff_2) / 7.0)

        # Combine with weighting
        similarity = base + 0.15 * margin_similarity
        return min(1.0, similarity)
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        x1 = self.data_x[i]
        x2 = self.data_x[j]
        y1 = self.data_y[i]
        y2 = self.data_y[j]
        
        
        score_diff_1 = y1.item()
        score_diff_2 = y2.item()
        
        similarity = self.calculate_similarity(score_diff_1, score_diff_2)
        label = 1.0 if similarity >= 0.6 else 0.0
        return x1.to(self.device), x2.to(self.device), torch.FloatTensor([label]).to(self.device)


class SiameseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers=2, head_output_dim=8, dropout_rate=0.3, bidirectional=True):
        """
        LSTM-based Siamese Network
        
        Args:
            input_dim: Number of features per timestep
            hidden_dim: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            head_output_dim: Output dimension of the embedding
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        # Store constructor parameters for saving/loading
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.head_output_dim = head_output_dim
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Projection head after LSTM
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.BatchNorm1d(lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_dim // 2, head_output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_one(self, x):
        """
        Forward pass for one input sequence
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Normalized embedding tensor of shape (batch_size, head_output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n.shape = (num_layers * D, batch, hidden_dim) where D is number of directions
        if self.bidirectional:
            h_fwd = h_n[-2, :, :]  # (batch, hidden_dim)
            h_bwd = h_n[-1, :, :]  # (batch, hidden_dim)
            final_hidden = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, 2*hidden_dim)
        else:
            final_hidden = h_n[-1, :, :]  # (batch, hidden_dim)

        # Alternative: pooling
        # final_hidden = lstm_out.mean(dim=1)  # (batch, hidden_dim * num_directions)
        embedding = self.head(final_hidden)
        
        # L2 normalize the output for better similarity computation
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
        
    def forward(self, x1, x2):
        """
        Forward pass for Siamese network
        
        Args:
            x1, x2: Input sequences of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Similarity scores of shape (batch_size, 1)
        """
        # Get normalized feature representations
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(emb1, emb2)
        
        # Convert cosine similarity from [-1, 1] to [0, 1] range
        similarity = (cosine_sim + 1) / 2
        
        # Clamp with small epsilon to preserve gradients
        eps = 1e-7
        similarity = torch.clamp(similarity, eps, 1.0 - eps)
        
        # Reshape to match expected output format [batch_size, 1]
        similarity = similarity.unsqueeze(1)
        
        return similarity


class SiameseLSTMClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None, sequence_length=None):
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        self.sequence_length = sequence_length
        
        # Track training metrics
        self.final_train_loss = None
        self.final_train_accuracy = None
        self.final_val_loss = None
        self.final_val_accuracy = None
        self.best_epoch = None
        self.epochs_trained = None
    
    def fit(self, X, y, val_X=None, val_y=None, batch_size=32):
        """
        Fit the LSTM Siamese model
        
        Args:
            X: Training sequences
            y: Training labels
            val_X: Validation sequences
            val_y: Validation labels
            batch_size: Batch size for training
        """
        # Create dataset with sequence handling
        train_dataset = NFLSequenceDataset(
            X, y, 
            sequence_length=self.sequence_length,
            max_pairs_per_sample=25, 
            device=self.device, 
        )

        print("LSTM data loaded!")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_X is not None and val_y is not None:
            val_dataset = NFLSequenceDataset(
                val_X, val_y, 
                sequence_length=self.sequence_length,
                max_pairs_per_sample=10, 
                device=self.device, 
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Early stopping parameters
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 7  # Slightly more patience for LSTM
        best_epoch = 0
        min_improvement = 1e-4
        
        print(f"Starting LSTM training on device: {self.device}")
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
                    print(f"LSTM Input tensors device: x1={x1.device}, x2={x2.device}, y={y.device}")
                    print(f"LSTM Input shapes: x1={x1.shape}, x2={x2.shape}")
                
                # Check for NaN in inputs
                if torch.isnan(x1).any() or torch.isnan(x2).any():
                    print("NaN in LSTM inputs")
                    continue
                    
                output = self.model(x1, x2)
                
                # Check for NaN in outputs
                if torch.isnan(output).any():
                    print("NaN in LSTM outputs")
                    continue
                    
                self.optimizer.zero_grad()
                loss = self.criterion(output, y)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print("NaN in LSTM loss")
                    continue
                    
                loss.backward()
                
                # Gradient clipping for LSTM stability
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
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                
                # Early stopping
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
                            print(f"Restored LSTM model from best epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
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
        Embed a single data point or batch
        
        Args:
            data: torch.Tensor of shape (n_samples, seq_length, input_dim) or (seq_length, input_dim)
            
        Returns:
            torch.Tensor of shape (n_samples, embedding_dim) or (embedding_dim,)
        """
        if data.dim() == 2:
            data = data.unsqueeze(0)  # Add batch dimension
            
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.forward_one(data)
            if embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)  # Remove batch dimension if single sample
            return embedding

    def embed_data(self, data):
        """
        Embed data from dictionary format
        
        Args:
            data: List of dictionaries with format {"row": np.array, "label": float}
            
        Returns:
            tuple: (embeddings, labels)
        """
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for item in data:
                # Extract the row data and label from dictionary
                row_data = item["rows"]
                label = item["label"]
                
                # Convert numpy array to tensor and handle reshaping for sequences
                if isinstance(row_data, np.ndarray):
                    x = torch.FloatTensor(row_data)
                else:
                    x = torch.FloatTensor(row_data)
                
                # Reshape for sequence if needed
                if x.dim() == 1 and self.sequence_length is not None:
                    features_per_step = x.shape[0] // self.sequence_length
                    x = x.reshape(self.sequence_length, features_per_step)
                
                # Add batch dimension
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                x = x.to(self.device)
                
                # Get embedding
                embedding = self.model.forward_one(x)
                embedding = embedding.squeeze(0)
                embeddings.append(embedding.cpu())
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
                    print("NaN in LSTM Evaluation")
                    continue
                    
                loss = self.criterion(output, y)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    print("NaN Loss in LSTM Evaluation")
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
        
        filename = f"{filepath_prefix}_LSTM_{timesteps_range[0]}-{timesteps_range[1]}"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestep': timesteps_range[0],
            'val_loss': self.final_val_loss,
            'val_accuracy': self.final_val_accuracy,
            'sequence_length': self.sequence_length,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'lstm_layers': self.model.lstm_layers,
                'head_output_dim': self.model.head_output_dim,
                'dropout_rate': self.model.dropout_rate,
                'bidirectional': self.model.bidirectional
            }
        }
        
        torch.save(checkpoint, filename)
        print(f"LSTM Model saved: {filename}")
        return filename
    
    @classmethod
    def load_model(cls, filepath, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate the model architecture
        model_config = checkpoint['model_config']
        siamese_lstm = SiameseLSTM(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            lstm_layers=model_config['lstm_layers'],
            head_output_dim=model_config['head_output_dim'],
            dropout_rate=model_config['dropout_rate'],
            bidirectional=model_config['bidirectional']
        )
        
        # Load model state
        siamese_lstm.load_state_dict(checkpoint['model_state_dict'])
        siamese_lstm.to(device)
        
        # Create classifier instance
        criterion = nn.BCELoss() 
        optimizer = torch.optim.AdamW(siamese_lstm.parameters(), lr=0.001)
        
        classifier = cls(
            siamese_lstm, 1, optimizer, criterion, device, 
            sequence_length=checkpoint.get('sequence_length')
        )
        
        # Restore training metrics
        classifier.final_val_loss = checkpoint.get('val_loss')
        classifier.final_val_accuracy = checkpoint.get('val_accuracy')
        classifier.best_epoch = checkpoint.get('timestep')
        
        print(f"LSTM Model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}")
        
        return classifier 