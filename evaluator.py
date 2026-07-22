import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, confusion_matrix

class Evaluator:
    @staticmethod
    def evaluate(model, x_eval, y_eval, device = None):
        device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        x_tensor = x_eval.float().to(device)
        with torch.no_grad():
            log = model(x_tensor)
            pred = torch.argmax(log, dim = 1).cpu().numpy()

        y_true = None
        if isinstance(y_eval, torch.Tensor):
            y_true = y_eval.numpy()
        else: 
            y_true = y_eval
        accuracy = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, average='macro')
        mse = mean_squared_error(y_true, pred)
        raw_cm = confusion_matrix(y_true, pred)

        num_classes = raw_cm.shape[0]
        if num_classes == 2:
            tn, fp, fn, tp = raw_cm.ravel()
            cm_string = (
                f"                   Negative    Positive\n"
                f"        Negative:    {tn:<7} (TN)  {fp:<7} (FP)\n"
                f"        Positive:    {fn:<7} (FN)  {tp:<7} (TP)"
            )
        elif num_classes == 3:
            cm_string = (
                f"                   Negative    Positive    Neutral\n"
                f"        Negative:    {raw_cm[0][0]:<10} {raw_cm[0][1]:<10} {raw_cm[0][2]:<10}\n"
                f"        Positive:    {raw_cm[1][0]:<10} {raw_cm[1][1]:<10} {raw_cm[1][2]:<10}\n"
                f"        Neutral:       {raw_cm[2][0]:<10} {raw_cm[2][1]:<10} {raw_cm[2][2]:<10}"
            )
        else:
            cm_string = np.array2string(raw_cm)

        return accuracy, f1, mse, cm_string    
