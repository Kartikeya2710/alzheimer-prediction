import torch
from dataset.alzheimer_loader import AlzheimerDataLoader

def get_mean_and_std(data_loader: AlzheimerDataLoader):
    """
    Get the mean and std for each channel in the training set of the dataloader
    :param data_loader: Custom data loader object
    :return: mean and standard deviation
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in data_loader.train_loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) ** 0.5

    return mean, std