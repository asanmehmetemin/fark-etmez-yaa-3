import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLPTrainer:
    def __init__(self, model, learning_rate = 0.001, patience=5, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.patience = patience
        self.best_model_weights = None

    def train_and_validate(self, dataset, batch_size=64, max_epochs = 50, logger = None):
        x_train, y_train = dataset["X_train"].float(), dataset["Y_train"].long()
        x_val, y_val = dataset["X_val"].float(), dataset["Y_val"].long()
        class_weights = dataset["class_weights"].to(self.device)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr= self.learning_rate)

        best_loss = float('inf')
        patience_ctr = 0

        for epoch in range(max_epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs,batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_x, val_y = x_val.to(self.device), y_val.to(self.device)
                val_outputs = self.model(val_x)
                validation_loss = criterion(val_outputs, val_y).item()

            if validation_loss < best_loss:
                best_loss = validation_loss
                patience_ctr = 0
                self.best_model_weights = self.model.state_dict().copy()
            else:
                patience_ctr += 1

            if patience_ctr >= self.patience:
                stop_epoch = epoch + 1
                if logger:
                    logger.log(f"Early Stopping is applied when Epoch {stop_epoch}/{max_epochs}.")
                break                    

        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        return self.model, best_loss        