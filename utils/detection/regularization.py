import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path="checkpoint.pt", verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0 
        self.best_score = None
        self.early_stop = False
        self.min_val_loss = torch.tensor(float('inf'))
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta :
            self.counter += 1
            if self.verbose:
                print("Early Stopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print("Validation loss decrese from {.4f} to {.4f}".format(self.min_val_loss, val_loss))
        self.min_val_loss = val_loss
        torch.save(model.state_dict(), self.path)