from copy import deepcopy
import torch
from sklearn.metrics import accuracy_score

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None
        self.restore_best_weights = restore_best_weights


    def __call__(self, val_loss, model=None):

        if val_loss < self.best_loss - self.min_delta:

            if self.restore_best_weights and model is not None:
                self.best_weights = deepcopy(model.state_dict())

            self.best_loss = val_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                
        return self.early_stop

    @staticmethod
    def compute_val_metrics(model, criterion, X_val, y_val):

        val_loss = 0.0
        val_accuracy = 0    

        with torch.no_grad():
            outputs = model(X_val)
            print("outputs :", outputs)
            loss = criterion(outputs, y_val)

            val_loss = loss.item() / len(y_val)
            # val_accuracy = accuracy_score(y_true=y_val.numpy(), y_pred=outputs.argmax(dim=1).numpy())
            val_accuracy = (outputs.argmax(dim=1) == y_val).sum().item() / len(y_val)

        return val_loss, val_accuracy
