import torch
import torch.nn as nn
from torchmetrics.classification import Precision, MulticlassAccuracy, Recall, F1Score

class MLP_MultiClass(nn.Module):
    """
    Multi-Layer Perceptron using the same sequence as the CNN
    """

    def __init__(self, num_features: int, window_size: int, num_classes: int):
        super().__init__()
        flattened_input_size = num_features * window_size

        self.layers = nn.Sequential(
            nn.Flatten(), # This allows us to turn the input [Batch, Window, Features] into [Batch, Flattened]
            nn.Linear(flattened_input_size, 512),
            nn.SiLU(), # Change to ReLU if CNN changes to SiLU
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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
        return self.layers(x)
    
    def train_model(self, *,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader | None = None,
        epochs: int | None = 10,
        path: str | None = None
    ) -> None:
        """
        To train the model using the provided datasets.

        Args:
            train_loader (torch.utils.data.DataLoader): The training dataset
            validation_loader (torch.utils.data.DataLoader): The validation datset. Defaults to none.
            epochs (int): The amount of complete pass through the training dataset to train the model.
            path (str): The path to save the best weights
        """        
        self.to(self.__device_location)
        best_accuracy = 0
        smallEpoch = True if epochs <= 100 else False
        currentPath = path if path else 'IDS_DEFAULT_BEST.pth'

        accuracy_metric = MulticlassAccuracy(
            num_classes=self.layers[-1].out_features,
            average='macro'
        ).to(self.__device_location)

        if train_loader:
            for epoch in range(1, epochs+1):
                self.train()
                train_loss = 0.
                accuracy_metric.reset()

                for inputs, targets in train_loader:
                    inputs = inputs.to(self.__device_location)
                    targets = targets.to(self.__device_location)

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
                    val_accuracy_metric = MulticlassAccuracy(
                        num_classes=self.layers[-1].out_features,
                        average='macro'
                    ).to(self.__device_location)
                    val_accuracy_metric.reset()

                    with torch.no_grad():
                        for inputs, targets in validation_loader:
                            inputs = inputs.to(self.__device_location)
                            targets = targets.to(self.__device_location)
                            
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
                        self.__save_model(path=currentPath)
                        print(f'[BEST MODEL SAVED]: Validation Accuracy: {val_accuracy:.4f} at Epoch {epoch}')

                if epoch % 10 == 0 or smallEpoch:
                    current_lr = self.__optimizer.param_groups[0]['lr']
                    output_string = \
                        f'[TRAIN INFO]: Current epoch: {epoch} | Train accuracy: {accuracy:.5f} | ' + \
                        f'Train loss: {train_loss:.10f} | LR: {current_lr}'
                    if validation_loader:
                        output_string += \
                            f' | Validation accuracy: {val_accuracy:.5f} | ' + \
                            f'Validation loss: {val_loss:.10f}'
                    print(output_string)

                if not validation_loader:
                    print('[WARNING]: Please provide validation dataset in order to save the best parameters.')

        else:
            raise ValueError('[ERROR]: Expected training data, but no training data provided. Try Again.')
        
           
    def test_model(self,
        test_loader: torch.utils.data.DataLoader,
        path: str | None = None
    ) -> dict:
        """
        Evaluates the model on the provided test dataset.

        This method iterates over the test_loader, performs forward passes 
        through the model, and can be extended to compute performance metrics 
        such as accuracy, precision, recall, or F1-score.

        Args:
            test_loader (torch.utils.data.DataLoader): The test dataset
            path (str): The path to get the best weights

        Return:
            A dictionary filled with the model's metrics
        """
        if test_loader:
            try:
                currentPath = path if path else 'IDS_DEFAULT_BEST.pth'
                self.__load_model(path=currentPath)
            except FileNotFoundError:
                raise FileNotFoundError('[ERROR] No saved model found. Please train the model first.')
            
            self.eval()
            test_loss = 0.
            num_classes = self.layers[-1].out_features

            accuracy = MulticlassAccuracy(num_classes=self.layers[-1].out_features, average='macro').to(self.__device_location)
            precision = Precision(task='multiclass', num_classes=num_classes, average='macro').to(self.__device_location)
            recall = Recall(task='multiclass', num_classes=num_classes, average='macro').to(self.__device_location)
            f1_score = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(self.__device_location)

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.__device_location)
                    targets = targets.to(self.__device_location)
                    
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

            print(f'[TEST INFO]: Test Loss: {test_loss:.10f} | Test Accuracy: {acc:.5f} | ' +
                  f'Test Precision: {prec:.5f} | Test Recall: {rec:.5f} | Test F1 Score: {f1:.5f}')
            
            return {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
        else:
            raise ValueError('[ERROR] Expected test data, but no test data provided. Try Again.')
        
    def __load_model(self, path: str | None = 'IDS_MLP_BEST.pth') -> None:
        """
        Loads the pre-trained model weights from a file.

        The model state dictionary is loaded from 'IDS_CNN_BEST.pth' 
        and mapped to the device specified in `self.__device_location`.
        The model is set to evaluation mode after loading.
        """
        self.load_state_dict(torch.load(path, map_location=self.__device_location))
        self.eval()
        print(f'[MODEL LOADED]: Model loaded from {path}')

    def __save_model(self, path: str | None = 'IDS_MLP_BEST.pth') -> None:
        """
        Saves the current model weights to a file.

        The model's state dictionary is saved to 'IDS_CNN_BEST.pth'.
        This allows reloading the model later without retraining.
        """
        torch.save(self.state_dict(), path)
        print(f'[MODEL SAVED]: Model saved to {path}')