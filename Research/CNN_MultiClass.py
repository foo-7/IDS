import torch
import torch.nn as nn
from torchmetrics.classification import Precision, Accuracy, Recall, F1Score

class CNN_MultiClass(nn.Module):

    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        with torch.no_grad():
            input = torch.zeros(1, 1, input_length)
            conv_out = self.cnn_layers(input)
            flattened_size = conv_out.view(1, -1).shape[1]

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.Dropout(0.5),

            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

        self.__loss_function = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        self.__device_location = ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.__device_location)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = self.dense_layers(x)
        return x

    def train_model(self, *,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader | None = None,
        epochs: int | None = 10
    ) -> None:
        self.to(self.__device_location)
        best_accuracy = 0
        smallEpoch = True if epochs <= 100 else False

        accuracy_metric = Accuracy(
            task='multiclass',
            num_classes=self.dense_layers[-1].out_features,
        ).to(self.__device_location)

        if train_loader:
            for epoch in range(1, epochs+1):
                self.train()
                train_loss = 0.
                accuracy_metric.reset()

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.__device_location), targets.to(self.__device_location)

                    if targets.ndim > 1:
                        targets = torch.argmax(targets, dim=1)
                    targets = targets.view(-1).long()

                    self.__optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = self.__loss_function(outputs, targets)
                    loss.backward()
                    self.__optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    accuracy_metric.update(predicted, targets)

                train_loss /= len(train_loader.dataset)
                accuracy = accuracy_metric.compute().item()

                if validation_loader:
                    self.eval()
                    val_loss = 0.
                    val_accuracy_metric = Accuracy(
                        task='multiclass',
                        num_classes=self.dense_layers[-1].out_features,
                    ).to(self.__device_location)
                    val_accuracy_metric.reset()

                    with torch.no_grad():
                        for inputs, targets in validation_loader:
                            inputs, targets = inputs.to(self.__device_location), targets.to(self.__device_location)

                            if targets.ndim > 1:
                                targets = torch.argmax(targets, dim=1)
                            targets = targets.view(-1).long()
                            
                            outputs = self(inputs)
                            loss = self.__loss_function(outputs, targets)

                            val_loss += loss.item() * inputs.size(0)
                            _, predicted = torch.max(outputs, 1)
                            val_accuracy_metric.update(predicted, targets)

                    val_loss /= len(validation_loader.dataset)
                    val_accuracy = val_accuracy_metric.compute().item()

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        self.__save_model()
                        print(f'[BEST MODEL SAVED] Validation Accuracy: {val_accuracy:.4f} at Epoch {epoch}')

                if epoch % 10 == 0 or smallEpoch:
                    current_lr = self.__optimizer.param_groups[0]['lr']
                    output_string = \
                        f'[TRAIN INFO] Current epoch: {epoch} | Train accuracy: {accuracy:.5f} | ' + \
                        f'Train loss: {train_loss:.10f} | LR: {current_lr}'
                    if validation_loader:
                        output_string += \
                            f' | Validation accuracy: {val_accuracy:.5f} | ' + \
                            f'Validation loss: {val_loss:.10f}'
                    print(output_string)

                if not validation_loader:
                    print('[WARNING] Please provide validation dataset in order to save the best parameters.')

        else:
            raise ValueError('[ERROR] Expected training data, but no training data provided. Try Again.')
        
    def test_model(self,
        test_loader: torch.utils.data.DataLoader
    ) -> None:
        if test_loader:
            try:
                self.__load_model()
            except FileNotFoundError:
                raise FileNotFoundError('[ERROR] No saved model found. Please train the model first.')
            
            self.eval()
            test_loss = 0.

            accuracy = Accuracy(task='multiclass', num_classes=self.dense_layers[-1].out_features).to(self.__device_location)
            precision = Precision(task='multiclass', num_classes=self.dense_layers[-1].out_features, average='macro').to(self.__device_location)
            recall = Recall(task='multiclass', num_classes=self.dense_layers[-1].out_features, average='macro').to(self.__device_location)
            f1_score = F1Score(task='multiclass', num_classes=self.dense_layers[-1].out_features, average='macro').to(self.__device_location)

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.__device_location), targets.to(self.__device_location)
                    
                    if targets.ndim > 1:
                        targets = torch.argmax(targets, dim=1)
                    targets = targets.view(-1).long()

                    outputs = self(inputs)
                    loss = self.__loss_function(outputs, targets)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)

                    accuracy.update(predicted, targets)
                    precision.update(predicted, targets)
                    recall.update(predicted, targets)
                    f1_score.update(predicted, targets)

            test_loss /= len(test_loader.dataset)
            acc = accuracy.compute().item()
            prec = precision.compute().item()
            rec = recall.compute().item()
            f1 = f1_score.compute().item()

            print(f'[TEST INFO] Test Loss: {test_loss:.10f} | Test Accuracy: {acc:.5f} | ' +
                  f'Test Precision: {prec:.5f} | Test Recall: {rec:.5f} | Test F1 Score: {f1:.5f}')
            
        else:
            raise ValueError('[ERROR] Expected test data, but no test data provided. Try Again.')
        
    def __load_model(self) -> None:
        path = 'IDS_CNN_BEST.pth'
        self.load_state_dict(torch.load(path, map_location=self.__device_location))
        self.eval()
        print(f'[MODEL LOADED] Model loaded from {path}')

    def __save_model(self) -> None:
        path = 'IDS_CNN_BEST.pth'
        torch.save(self.state_dict(), path)
        print(f'[MODEL SAVED] Model saved to {path}')