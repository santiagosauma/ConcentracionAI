from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, brier_score_loss, precision_recall_curve, auc

def compute_metrics(y_true, y_pred, y_proba=None):
	"""Calcula métricas principales de clasificación."""
	metrics = {
		'accuracy': accuracy_score(y_true, y_pred),
		'precision': precision_score(y_true, y_pred),
		'recall': recall_score(y_true, y_pred),
		'f1': f1_score(y_true, y_pred),
		'roc_auc': roc_auc_score(y_true, y_proba) if y_proba is not None else None,
		'mcc': matthews_corrcoef(y_true, y_pred),
		'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
		'brier_score': brier_score_loss(y_true, y_proba) if y_proba is not None else None
	}
	if y_proba is not None:
		precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
		metrics['pr_auc'] = auc(recall_curve, precision_curve)
	return metrics
