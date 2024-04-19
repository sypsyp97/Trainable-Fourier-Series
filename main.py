import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gee_loss import gee_loss
from model import ParReconstruction2D_Eff


class PairedHDF5Dataset(Dataset):
    def __init__(self, data_dir, groundtruth_dir):
        super(PairedHDF5Dataset, self).__init__()

        # Get a list of paths
        self.data_paths = sorted(
            [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.hdf5')])
        self.groundtruth_paths = sorted(
            [os.path.join(groundtruth_dir, file) for file in os.listdir(groundtruth_dir) if file.endswith('.hdf5')])

        assert len(self.data_paths) == len(self.groundtruth_paths), "Mismatch between data and groundtruth files"

        # Calculate the cumulative sizes
        self.cumulative_sizes = []
        cum_size = 0
        for path in self.data_paths:
            with h5py.File(path, 'r') as file:
                cum_size += len(file['data'])
                self.cumulative_sizes.append(cum_size)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        # Find the correct file and local index
        file_idx = next(i for i, cum_size in enumerate(self.cumulative_sizes) if cum_size > index)
        if file_idx == 0:
            local_index = index
        else:
            local_index = index - self.cumulative_sizes[file_idx - 1]

        with h5py.File(self.data_paths[file_idx], 'r') as data_file, h5py.File(self.groundtruth_paths[file_idx],
                                                                               'r') as gt_file:
            data = torch.tensor(np.expand_dims(data_file['data'][local_index], axis=0).squeeze())
            groundtruth = torch.tensor(np.expand_dims(gt_file['data'][local_index], axis=0).squeeze())
        return data, groundtruth


def train_model(model, data_loader, optimizer, num_epochs, val_loader=None, save_path="best_model_effft2.pth"):
    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, groundtruth) in enumerate(data_loader):
            data, groundtruth = data.cuda(), groundtruth.cuda()
            optimizer.zero_grad()
            outputs, fp = model(data)
            loss = gee_loss(outputs.unsqueeze(0), groundtruth.unsqueeze(0))
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(data_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for data, groundtruth in val_loader:
                    data, groundtruth = data.cuda(), groundtruth.cuda()
                    outputs, _ = model(data)
                    loss = gee_loss(outputs.unsqueeze(0), groundtruth.unsqueeze(0))
                    val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

            val_loss_history.append(avg_val_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Finished Training")
    return train_loss_history, val_loss_history


if __name__ == "__main__":
    # Usage:
    train_dataset = PairedHDF5Dataset(
        'D:\\datasets\LoDoPaB-CT\observation_train',
        'D:\\datasets\LoDoPaB-CT\ground_truth_train'
    )

    validation_dataset = PairedHDF5Dataset(
        'D:\\datasets\LoDoPaB-CT\observation_validation',
        'D:\\datasets\LoDoPaB-CT\ground_truth_validation'
    )
    test_dataset = PairedHDF5Dataset(
        'D:\\datasets\LoDoPaB-CT\observation_test',
        'D:\\datasets\LoDoPaB-CT\ground_truth_test'
    )

    generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(validation_dataset, batch_size=1)
    # Volume parameters:
    volume_size = 362  # size of the volume/image
    volume_shape = [volume_size, volume_size]  # shape of the volume as [height, width]
    volume_spacing = [1, 1]  # spacing between pixels in the volume

    # Detector parameters:
    detector_shape = [513]  # shape of the detector
    detector_spacing = [1]  # spacing between detector pixels

    # Trajectory parameters:
    number_of_projections = 1000  # number of projections in the sinogram
    angular_range = -np.pi  # angular range of the trajectory (half-circle in this case)

    # Create an instance of the Geometry class and initialize it with the above parameters
    geometry = Geometry()
    geometry.init_from_parameters(volume_shape=volume_shape, volume_spacing=volume_spacing,
                                  detector_shape=detector_shape, detector_spacing=detector_spacing,
                                  number_of_projections=number_of_projections, angular_range=angular_range,
                                  trajectory=circular_trajectory_2d)

    model = ParReconstruction2D_Eff(geometry).cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, steps_per_epoch=len(data_loader), epochs=25)
    num_epochs = 25

    loss_history, val_loss_history = train_model(model, data_loader, optimizer, num_epochs, val_loader)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()
