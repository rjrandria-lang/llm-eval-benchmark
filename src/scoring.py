def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy score.
    """
    return sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)) / len(y_true)


def precision_score(y_true, y_pred):
    """
    Compute the precision score.
    """
    tp = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    fp = sum((y_t == 0 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Compute the recall score.
    """
    tp = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    fn = sum((y_t == 1 and y_p == 0) for y_t, y_p in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    Compute the F1 score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0