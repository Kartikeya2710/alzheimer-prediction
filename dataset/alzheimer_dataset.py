import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class AlzheimerDataset(Dataset):
	def __init__(self, config, transforms=None):
		self.config = config
		self.transforms = transforms

		self.root = config.root_dir
		self.imageset = pd.read_csv(config.annotations_dir, header=None)
		self.transforms = transforms
		# We need these targets for stratified splitting
		self.targets = list(self.imageset.iloc[:, 1])
		self.targets = list(map(config.label_to_idx.get, self.targets))

	def __getitem__(self, idx):
		img_name = self.imageset.iloc[idx, 0]
		label = self.imageset.iloc[idx, 1]
		target = float(self.config.label_to_idx[label])
		img_path = os.path.join(self.root, label, img_name)

		img = Image.open(img_path)

		if self.transforms is not None:
			img = self.transforms(img)

		return (img, target)

	def __len__(self):
		return len(self.imageset)