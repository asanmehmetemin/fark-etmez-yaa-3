import numpy as np
import torch

from dataLoader import DataLoader
from featureExtractor import FeatureExtractor

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def prepare_dataset(loader, max_features = 5000):
    text, label = loader()
    text_train, text_val, text_test, label_train, label_val, label_test = DataLoader.split_data(text, label)

    feature_extractor = FeatureExtractor(max_features=max_features)
    train_vectors = feature_extractor.fit_transform_train(text_train)
    val_vectors = feature_extractor.transform_unseen(text_val)
    test_vectors = feature_extractor.transform_unseen(text_test)

    unique_class, class_counts = np.unique(label_train, return_counts=True)

    num_of_class = len(unique_class)
    class_weight_val = np.zeros(num_of_class, dtype=np.float32)
    total_samples = class_counts.sum()

    for class_label, class_count in zip(unique_class, class_counts):
        class_weight_val[int(class_label)] = total_samples / (num_of_class * class_count)
    
    class_weights = torch.tensor(class_weight_val)

    dataset = {
        "X_train": torch.tensor(train_vectors, dtype=torch.float32),
        "X_val": torch.tensor(val_vectors, dtype=torch.float32),
        "X_test": torch.tensor(test_vectors, dtype=torch.float32),
        "Y_train": torch.tensor(label_train, dtype=torch.long),
        "Y_val": torch.tensor(label_val, dtype=torch.long),
        "Y_test": torch.tensor(label_test, dtype=torch.long),
        "class_weights": class_weights,
        "n_classes": num_of_class,
        "n_features": train_vectors.shape[1]
    }

    return dataset