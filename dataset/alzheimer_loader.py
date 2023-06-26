from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from dataset.alzheimer_dataset import AlzheimerDataset
from sklearn.model_selection import train_test_split
import numpy as np

class AlzheimerDataLoader:
    def __init__(self, config):
        self.config = config

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(self.config.mean,), std=(self.config.std,))
        ])

        dataset = AlzheimerDataset(config, transforms=transform)

        train_idx, validation_idx = train_test_split(
            np.arange(len(dataset)),
            test_size=config.validation_split,
            random_state=config.random_seed,
            stratify=dataset.targets
        )

        train_dataset = Subset(dataset, train_idx)

        validation_dataset = Subset(dataset, validation_idx)

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 

        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )