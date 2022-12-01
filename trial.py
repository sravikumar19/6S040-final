import ls

# Load the Tox21 dataset.
data = ls.datasets.Tox21()

# # Learning to split the Tox21 dataset.
# # Here we use a simple mlp as our model backbone and use roc_auc as the evaluation metric.
train_data, test_data = ls.learning_to_split(
    data, model={'name': 'mlp'}, metric='roc_auc')
