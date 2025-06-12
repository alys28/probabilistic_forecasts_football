import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 1 if similar, 0 if dissimilar
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


class NFLDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.pairs = []
        for i in range(len(self.data_x)):
            for j in range(len(self.data_x)):
                if i != j:
                    self.pairs.append((self.data_x[i], self.data_x[j], self.data_y[i], self.data_y[j]))
        self.pairs = torch.tensor(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns a pair of data points and a label. The label is 1 if the two data points have the same game outcome, 0 otherwise.
        Assumes that each data point has the label in the first column.
        """
        pair = self.pairs[idx]
        return pair[0], pair[1], 1 if pair[2] == pair[3] else 0
    


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=64):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward_one(self, x):
        """Forward pass for one input"""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        for batch in train_loader:
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            output1, output2 = model(x1, x2)
            loss = criterion(output1, output2, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Count the number of correct predictions based on distance threshold
            with torch.no_grad():
                distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
                predictions = (distances < 0.5).float()  # Threshold for similarity
                count += (predictions == y).float().sum()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {count / len(train_loader)}")
        if val_loader is not None:
            val_loss, val_accuracy = evaluate_siamese_network(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

def evaluate_siamese_network(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            output1, output2 = model(x1, x2)
            loss = criterion(output1, output2, y)
            total_loss += loss.item()
            # Count the number of correct predictions based on distance threshold
            distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            predictions = (distances < 0.5).float()  # Threshold for similarity
            count += (predictions == y).float().sum()
    return total_loss / len(test_loader), count / len(test_loader)

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
    
    def fit(self, X, y, val_X = None, val_y = None, batch_size = 128):
        """
        Fits the SiameseClassifier to the data.
        Args:
            X: np.array of shape (n_samples, n_features)
            y: np.array of shape (n_samples,)
            val_X: np.array of shape (n_val_samples, n_features)
            val_y: np.array of shape (n_val_samples,)
        """
        train_dataset = NFLDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_X is not None and val_y is not None:
            val_dataset = NFLDataset(val_X, val_y)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
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

