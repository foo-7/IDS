import torch
import torch.nn as nn

class CNN_Model(nn.Module):

    def __init__(self, input_length):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        )

        """
        We want to know how big the output is after all convolution and max pool layers
        so that we can:
            - Flatten output to feed it into the first parameter in the first dense layer
            - Input size to Linear = channels * new_length after convolution layers
        """
        # Not to track gradients during forward pass.
        with torch.no_grad():
            input = torch.zeros(1, 1, input_length)
            conv_out = self.cnn_layers(input)
            conv_out_size = conv_out.view(1, -1).size(1)

        self.dense_layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(conv_out_size, 128),      # First hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 128),                # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1)                   # Output layer
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_function = nn.BCELoss()
        self.device_location = ("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.dense_layers(x)
        return torch.sigmoid(x)
    
    def train_model(self, *, train_loader, validation_loader, epochs=100):
        device = next(self.parameters()).device
        self.to(self.device_location)

        if train_loader:
            for epoch in range(1, epochs+1):
                self.train()
                train_loss = 0.
                correct = total = best_accuracy = 0
                best_model_path = 'IDS_CNN_BEST.pth'

                for inputs, targets in train_loader:
                    # To GPU
                    inputs = inputs.to(self.device_location)
                    targets = targets.to(self.device_location)

                    # Ensure targets' shape matches output shape: [batch_size, 1]
                    targets = targets.view(-1,1)

                    self.optimizer.zero_grad()  # Clear old gradients from previous backpropagation
                    output = self(inputs)
                    loss = self.loss_function(output, targets)
                    loss.backward()
                    self.optimizer.step()       # Update parameters
                    train_loss += loss.item() * inputs.size(0)

                    predicted = (output >= 0.5).float()
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

                train_loss /= len(train_loader.dataset)
                accuracy = correct / total

                if validation_loader:
                    self.eval()
                    val_loss = 0.
                    val_correct = val_total = 0
                    
                    # Context manager that tells the framework not to compute or store gradients during the
                    # operations inside its block
                    with torch.no_grad():
                        for inputs, targets in validation_loader:
                            inputs = inputs.to(self.device_location)
                            targets = targets.to(self.device_location)
                            targets = targets.view(-1, 1)

                            output = self(inputs)
                            loss = self.loss_function(output, targets)
                            val_loss += loss.item() * inputs.size(0)

                            predicted = (output >= 0.5).float()
                            val_correct += (predicted == targets).sum().item()
                            val_total += targets.size(0)

                    val_loss /= len(validation_loader.dataset)
                    val_accuracy = val_correct / val_total

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        self.save_model()
                    else:
                        print('[WARNING] Please provide validation dataset in order to save the best parameters for CNN model.')


                if epoch % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    output_string = \
                        f'[TRAIN INFO] Current epoch: {epoch} | Train accuracy: {accuracy:.3f} | ' + \
                        f'Train loss: {train_loss:.10f} | LR: {current_lr}'
                    if validation_loader:
                        output_string += \
                            f' | Validation accuracy: {val_accuracy:.3f} | ' + \
                            f'Validation loss: {val_loss:.10f} | '
                    print(output_string)
            
        else:
            raise ValueError(f'[ERROR] Expected train dataset to be passed on train_model function call, try again')

    def test_model(self, test_loader):
        if test_loader:
            try:
                self.load_model()
            except FileNotFoundError:
                raise FileNotFoundError('[ERROR] File not found on test_model function call. Please check.')
            
            self.eval()
            test_loss = 0.
            correct = total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device_location), targets.to(self.device_location)
                    targets = targets.view(-1, 1)

                    output = self(inputs)
                    loss = self.loss_function(output, targets)
                    test_loss += loss.item() * inputs.size(0)

                    predicted = (output >= 0.5).float()
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            test_loss /= len(test_loader.dataset)
            accuracy = correct / total
            print(f'[TEST INFO] Accuracy: {accuracy:.3f} | Loss: {test_loss:.10f}')

        else:
            raise ValueError(f'[ERROR] Expected test dataset to be passed on test_model function call, try again')
        
    def load_model(self):
        path = 'IDS_CNN_BEST.pth'
        self.load_state_dict(torch.load('IDS_CNN_BEST.pth', map_location=self.device_location))
        self.eval()
        print(f'[LOAD INFO]: Model loaded from {path}')

    def save_model(self):
        path = 'IDS_CNN_BEST.pth'
        torch.save(self.state_dict(), path)
        #print(f'[SAVE INFO]: Model saved to {path}')