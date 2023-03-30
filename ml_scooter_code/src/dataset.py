import ast

import torch
import pandas as pd


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, grid_size, subset, train_percent):
        # Define the member variables
        self.grid_size = grid_size

        # Load and prepare the csv dataset
        self.dataset = self._load_csv(dataset_path)

        # Get the train-test split
        self._split_dataset(subset, train_percent)

    def __len__(self):
        '''
        Return the length of the dataset
        '''
        return len(self.dataset)

    def _load_csv(self, dataset_path):
        '''
        Load and prepare the csv dataset
        '''
        # Read the csv dataset
        df = pd.read_csv(dataset_path)

        # Loop through every group and store the indices
        # and values of the sparse tensor
        dataset = []
        for _, group in df.groupby('time'):
            dataset.append({
                'indices': group['grid_index'].apply(ast.literal_eval).tolist(),
                'values': (group['destination_count'] - group['origin_count']).tolist()
            })

        return dataset

    def _split_dataset(self, subset, train_percent):
        '''
        Perform dataset split accordingly if its train or test set
        '''
        train_num = int(len(self.dataset) * train_percent)
        if subset == 'train':
            self.dataset = self.dataset[: train_num]
        elif subset == 'test':
            self.dataset = self.dataset[train_num: ]

    def __getitem__(self, i):
        '''
        Create a dense tensor from the indices and values of the sparse tensor
        '''
        return torch.sparse_coo_tensor(
            list(zip(*self.dataset[i]['indices']))[::-1],
            self.dataset[i]['values'],
            self.grid_size,
            dtype=torch.float32
        ).to_dense()[None]


class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, grid_size, seq_len, subset, train_percent):
        # Define the member variables
        self.seq_len = seq_len
        self.grid_size = grid_size

        # Load and prepare the csv dataset
        self.dataset = self._load_csv(dataset_path)

        # Construct sequences
        self._construct_sequences()

        # Get the train-test split
        self._split_dataset(subset, train_percent)

    def __len__(self):
        '''
        Return the length of the dataset
        '''
        return len(self.dataset)

    def _load_csv(self, dataset_path):
        '''
        Load and prepare the csv dataset
        '''
        # Read the csv dataset
        df = pd.read_csv(dataset_path)

        # Loop through every group and store the indices
        # and values of the sparse tensor
        dataset = []
        for _, group in df.groupby('time'):
            dataset.append({
                'indices': group['grid_index'].apply(ast.literal_eval).tolist(),
                'values': (group['destination_count'] - group['origin_count']).tolist()
            })

        return dataset

    def _split_dataset(self, subset, train_percent):
        '''
        Perform dataset split accordingly if its train or test set
        '''
        train_num = int(len(self.dataset) * train_percent)
        if subset == 'train':
            self.dataset = self.dataset[: train_num]
        elif subset == 'test':
            self.dataset = self.dataset[train_num: ]

    def _construct_sequences(self):
        '''
        Construct the sequences of demand matrices for recurrent prediction
        '''
        self.dataset = [
            self.dataset[i: i + self.seq_len + 1]
            for i in range(len(self.dataset) - self.seq_len)
        ]

    def __getitem__(self, i):
        '''
        Create a sequence of dense tensors from the indices and values of sparse tensors
        '''
        # Entire sequence except last item as the input
        input = torch.stack([
            torch.sparse_coo_tensor(
                list(zip(*self.dataset[i][j]['indices']))[::-1],
                self.dataset[i][j]['values'],
                self.grid_size,
                dtype=torch.float32
            ).to_dense()[None]
            for j in range(self.seq_len - 1)
        ])

        # Last item of the sequence must be predicted by the LSTM
        output = torch.sparse_coo_tensor(
            list(zip(*self.dataset[i][-1]['indices']))[::-1],
            self.dataset[i][-1]['values'],
            self.grid_size,
            dtype=torch.float32
        ).to_dense()[None]

        return {'input': input, 'output': output}