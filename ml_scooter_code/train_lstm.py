import os
import argparse
import datetime

import pkbar
import torch
from torch.utils.tensorboard import SummaryWriter

from src import model
from src import dataset


class Trainer:
    def __init__(self, device):
        # Define member variables
        self.device = device

        # Define the train dataloader
        self.train_loader = torch.utils.data.DataLoader(
            dataset.LSTMDataset(
                FLAGS.dataset_path,
                (FLAGS.grid_height, FLAGS.grid_width),
                FLAGS.seq_len,
                'train',
                FLAGS.train_percent),
            batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

        # Define the test dataloader
        self.test_loader = torch.utils.data.DataLoader(
            dataset.LSTMDataset(
                FLAGS.dataset_path,
                (FLAGS.grid_height, FLAGS.grid_width),
                FLAGS.seq_len,
                'test',
                FLAGS.train_percent),
            batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)

        # Define the autoencoder model
        self.model = model.LSTM(
            FLAGS.checkpoint_path,
            (FLAGS.grid_height, FLAGS.grid_width)
        ).to(device)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.lstm.parameters(),
            FLAGS.learning_rate,
            weight_decay=1e-5)

        # Define the loss function
        self.loss_fn = torch.nn.MSELoss()

        # Create a progress bar object
        self.progress_bar = None

    def _learning_step(self, data, subset, step=None):
        '''
        Implements the train or test cycle for a batch
        '''
        # Load the demand matrix input
        input = data['input'].to(self.device)
        output = data['output'].to(self.device)

        # Perform forward pass to get the reconstructed demand matrix
        pred = self.model(input)

        # Compute the loss
        loss = self.loss_fn(output, pred)

        # Extra steps for the train cycle
        if subset == 'train':
            # Perform backpropagation
            loss.backward()

            # Update the progress bar
            self.progress_bar.update(step, values=[('train_mse_loss', loss.item())])

        return loss.item()

    def train(self):
        '''
        Implements the training script
        '''
        # Create the tensorboard summary writer
        run_folder = f'runs/lstm/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(run_folder)

        # Execute the training loop
        best_test_loss = float('inf')
        for epoch in range(FLAGS.epochs):
            # Create a progress bar object
            self.progress_bar = pkbar.Kbar(
                target=len(self.train_loader),
                epoch=epoch,
                num_epochs=FLAGS.epochs,
                width=8,
                always_stateful=False
            )

            # Execute the train cycle
            train_loss = 0
            self.model.train()
            for step, data in enumerate(self.train_loader):
                # Zero out the gradients before forward pass
                self.optimizer.zero_grad()

                # Perform forward pass and backpropagation
                train_loss += self._learning_step(data, 'train', step)

                # Update the weights
                self.optimizer.step()

            # Aggregate the train loss
            train_loss /= (step + 1)

            # Execute the test cycle if required
            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for step, data in enumerate(self.test_loader):
                    test_loss += self._learning_step(data, 'test')

            # Aggregate the test loss
            test_loss /= (step + 1)

            # If the test loss reduces, update the best test loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss

                # Store the checkpoint
                torch.save(self.model.state_dict(), os.path.join(
                    run_folder, f'model_{epoch}_{best_test_loss:.4f}.pth'))

            # Write the tensorboard scalars
            writer.add_scalar(f'mse_reconstruction_loss/train', train_loss, epoch + 1)
            writer.add_scalar(f'mse_reconstruction_loss/test', test_loss, epoch + 1)

            # Update the progress bar
            self.progress_bar.add(1, values=[('test_mse_loss', test_loss)])

        # Print the final test loss
        print(f'\nTraining completed\nBest test loss: {best_test_loss}')


def main():
    # Define the trainer object
    trainer = Trainer(torch.device('cuda'))

    # Train the convolutional autoencoder model
    trainer.train()


def parse_arguments():
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser(description='Train the convolutional autoencoder model to convert sparse to dense features')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='runs/autoencoder/20221208_014803/model_96_0.0078.pth',
        help='''Path to the pretrained autoencoder'''
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/preprocessed_scooter_data.csv',
        help='''Path to the preprocessed CSV dataset'''
    )

    parser.add_argument(
        '--train_percent',
        type=float,
        default=0.8,
        help='''Percentage of the dataset to use for training'''
    )

    parser.add_argument(
        '--grid_height',
        type=int,
        default=126,
        help='''Height of the demand matrix'''
    )

    parser.add_argument(
        '--grid_width',
        type=int,
        default=130,
        help='''Width of the demand matrix'''
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        default=5,
        help='''Length of the input sequence'''
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='''Batch size used in training'''
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
        help='''Number of workers for the dataloader'''
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='''Number of training iterations'''
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='''Learning rate of the optimizer'''
    )

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    FLAGS = parse_arguments()

    # Enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Train the convolutional autoencoder model
    main()