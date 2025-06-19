import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 1 if similar, 0 if dissimilar
        # Add small epsilon to prevent sqrt of zero/negative numbers
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


class NFLDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.LongTensor(data_y)
        self.pairs = []
        # Create pairs of indices instead of actual data
        for i in range(len(self.data_x)):
            for j in range(i+1, min(i+100, len(self.data_x))):
                    self.pairs.append((i, j))
        print(len(self.pairs))
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
        
        # Method 1: Categorical approach considering both magnitude and sign
        def get_score_category(score_diff):
            if score_diff <= -14:
                return 4  # Big loss (lost by more than 2 TDs)
            elif score_diff <= -7:
                return 3  # Moderate loss (lost by 1-2 TDs)
            elif score_diff <= -3:
                return 2  # Close loss (lost by 4-7 points)
            elif score_diff < 0:
                return 1  # Very close loss (lost by 1-3 points)
            elif score_diff == 0:
                return 0  # Tie game
            elif score_diff <= 3:
                return -1  # Very close win (won by 1-3 points)
            elif score_diff <= 7:
                return -2  # Close win (won by 4-7 points)
            elif score_diff <= 14:
                return -3  # Moderate win (won by 1-2 TDs)
            else:
                return -4  # Big win (won by more than 2 TDs)
        

        def score_similarity(score_diff_1, score_diff_2):
            cat1 = get_score_category(score_diff_1)
            cat2 = get_score_category(score_diff_2)
            # Same winner if both categories have same sign or both are zero (tie)
            same_winner = (cat1 == 0 and cat2 == 0) or (cat1 * cat2 > 0)
            max_dist = 8  # Max possible distance in your categories
            dist = abs(cat1 - cat2)
            
            # Ensure similarity is never < 0, and respects same_winner constraint
            if same_winner:
                # Map distance to (0.5, 1.0] - closer distance = higher similarity
                # When dist=0, sim=1.0. When dist=max_dist, sim > 0.5
                sim = 0.5 + 0.5 * (max_dist - dist + 1) / (max_dist + 1)
            else:
                # Map distance to [0, 0.5) - closer distance = higher similarity
                # When dist=0, sim < 0.5. When dist=max_dist, sim=0
                sim = 0.5 * (max_dist - dist) / (max_dist + 1)
            
            return sim
        
        # Label = 1 if both games are in the same category, 0 otherwise
        label = score_similarity(score_diff_1, score_diff_2)
        
        # Alternative Method 2: Threshold-based with sign consideration
        # Uncomment the lines below to use this method instead:
        # threshold = 7  # Games are similar if score differences are within 7 points AND same sign
        # same_sign = (score_diff_1 >= 0) == (score_diff_2 >= 0)  # Both positive or both negative
        # within_threshold = abs(score_diff_1 - score_diff_2) <= threshold
        # label = 1 if same_sign and within_threshold else 0
        
        # Alternative Method 3: Separate win/loss similarity
        # Uncomment the lines below to use this method instead:
        # if (score_diff_1 >= 0) != (score_diff_2 >= 0):  # Different signs (one win, one loss)
        #     label = 0
        # else:  # Same sign (both wins or both losses)
        #     # Now check magnitude similarity
        #     threshold = 10
        #     label = 1 if abs(abs(score_diff_1) - abs(score_diff_2)) <= threshold else 0
        
        return x1, x2, torch.FloatTensor([label]).squeeze()
    


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=64):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 256)
        self.fc4 = nn.Linear(256, hidden_dim)
        # Use track_running_stats=False to make batch norm more stable
        self.bn1 = nn.BatchNorm1d(hidden_dim, track_running_stats=False, eps=1e-5, momentum=0.1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
                
        # Initialize weights properly
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        nn.init.zeros_(self.fc5.bias)

    def forward_one(self, x):
        """Forward pass for one input"""
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # Only apply batch norm if batch size > 1
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc5(x)
        # Add L2 normalization to embeddings for stability with epsilon
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
        x = x / norm
        return x
        
    def forward(self, x1, x2):
        """Forward pass for both inputs, returns embeddings"""
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2



def train_siamese_network(model, train_loader, epochs, optimizer, criterion, device, val_loader = None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        total_count = 0
        for batch in train_loader:
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            # Check for NaN in inputs
            if torch.isnan(x1).any() or torch.isnan(x2).any():
                print("Warning: NaN detected in inputs, skipping batch")
                continue
                
            output1, output2 = model(x1, x2)
            
            # Check for NaN in outputs
            if torch.isnan(output1).any() or torch.isnan(output2).any():
                print("Warning: NaN detected in outputs, skipping batch")
                continue
                
            optimizer.zero_grad()
            loss = criterion(output1, output2, y)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print("Warning: NaN detected in loss, skipping batch")
                continue
                
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Count the number of correct predictions based on distance threshold
            with torch.no_grad():
                distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)
                predictions = (distances < 0.5).float()  # Threshold for similarity
                count += (predictions == y.reshape(-1)).float().sum().item()
                total_count += len(predictions)
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {count / total_count if total_count != 0 else 'nan'}")
        if val_loader is not None:
            val_loss, val_accuracy = evaluate_siamese_network(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

def evaluate_siamese_network(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            # Skip batch if inputs contain NaN
            if torch.isnan(x1).any() or torch.isnan(x2).any():
                continue
                
            output1, output2 = model(x1, x2)
            
            # Skip batch if outputs contain NaN
            if torch.isnan(output1).any() or torch.isnan(output2).any():
                continue
                
            loss = criterion(output1, output2, y)
            
            # Skip batch if loss is NaN
            if torch.isnan(loss):
                continue
                
            total_loss += loss.item()
            
            # Count the number of correct predictions based on distance threshold
            distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)
            predictions = (distances < 0.5).float()  # Threshold for similarity
            count += (predictions == y).float().sum().item()
            total_samples += len(predictions)
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    accuracy = count / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

class SiameseClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device):
        """
        Initializes the SiameseClassifier.
        """
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def fit(self, X, y, val_X = None, val_y = None, batch_size = 128, score_difference_index = 1):
        """
        Fits the SiameseClassifier to the data.
        Args:
            X: np.array of shape (n_samples, n_features)
            y: np.array of shape (n_samples,)
            val_X: np.array of shape (n_val_samples, n_features)
            val_y: np.array of shape (n_val_samples,)
            score_difference_index: int, index of the score difference feature in X
        """
        train_dataset = NFLDataset(X, y)

        print("Data loaded!")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_X is not None and val_y is not None:
            val_dataset = NFLDataset(val_X, val_y)
            # Use the same batch size for validation to avoid batch norm issues
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        train_siamese_network(self.model, train_loader, self.epochs, self.optimizer, self.criterion, self.device, val_loader)

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

