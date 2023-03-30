import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        # Define the convolutional encoder 
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=1, padding='same'),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 4, 3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(4, 4, 3, stride=1, padding='same'),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(grid_size[0] * grid_size[1], 512)
        )

        # Define the deconvolutional Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, grid_size[0] * grid_size[1]),
            torch.nn.Unflatten(1, (4, grid_size[0] // 2, grid_size[1] // 2)),
            torch.nn.ConvTranspose2d(4, 16, 3, stride=2, padding=1, output_padding=1),            
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1),            
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        '''
        Perform forward pass of the encoder and decoder
        '''
        return self.decoder(self.encoder(x))


class LSTM(torch.nn.Module):
    def __init__(self, checkpoint_path, grid_size):
        super().__init__()
        # Define the autoencoder for encoding and decoding
        self.autoencoder = AutoEncoder(grid_size)

        # Define the LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=5,
            batch_first=True
        )

        # Load and freeze the weights of the autoencoder
        self.autoencoder.load_state_dict(torch.load(checkpoint_path))
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Get the sequence length
        seq_length = x.shape[1]

        # Deconstruct the batch and sequence length for encoder forward pass
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        # Perform encoder forward pass
        x = self.autoencoder.encoder(x)

        # Convert to batch, seq_length, feature_size
        x = x.reshape(x.shape[0] // seq_length, seq_length, x.shape[1])

        # Perform LSTM forward pass (access the last hidden state)
        x = self.lstm(x)[1][0][-1]

        # Perform decoder foward pass
        return self.autoencoder.decoder(x)