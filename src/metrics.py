from sklearn.metrics import roc_auc_score

def roc_auc(y, y_pred_probs):
    y_np = y.cpu().numpy()
    y_pred_np = y_pred_probs.cpu().detach().numpy()

    return roc_auc_score(y_np, y_pred_np)
