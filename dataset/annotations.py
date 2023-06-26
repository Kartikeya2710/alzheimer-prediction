import os
from csv import writer
import yaml

def create_annotations(config):
	with open(config.annotations_dir, 'w', newline='') as f:
		writer_obj = writer(f)

		for category in os.listdir(config.root_dir):
			category_path = os.path.join(config.root_dir, category)
			for image_id in os.listdir(category_path):
				row = [image_id, category]
				writer_obj.writerow(row)

		f.close()