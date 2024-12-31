import os
import numpy
import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from src.cats_dataset import CatsDataset
from src.models.cat_evaluation_result import CatEvaluationResult
from src.models.cats_evaluation_report import CatsEvaluationReport

class CatDiscriminatorNeuralNet(nn.Module):
    def __init__(self, learning_rate=0.1, saved_model_path=None) -> None:
        super().__init__()
        
        self.initialize_layers()

        if saved_model_path is not None:
            self.try_load_model(saved_model_path)
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def initialize_layers(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=10) # 10x10 kernels, moving across the 3 input channels (rgb), outputting 12 channels. The new (12) channels will be 504x504
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 12 feature maps, 252x252
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5) # 12 input feature maps, 24 output feature maps. The new (24) channels will be 248x248
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # this declaration is unnecessary, since we could reuse the first pool. The output will be 24 feature maps 124x124
        
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5) # 24 input feature maps, 48 output feature maps. The new (48) channels will be 120x120
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # The output will be 48 feature maps 70x70
        
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3) # 48 input feature maps, 96 output feature maps. The new (96) channels will be 68x68
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # The output will be 96 feature maps 34x34

        # These are the fully connected layers
        self.fc1 = nn.Linear(96 * 29 * 29, 1624)
        self.fc2 = nn.Linear(1624, 864)
        self.fc3 = nn.Linear(864, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 3)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

    def try_load_model(self, saved_model_path):
        if os.path.isfile(saved_model_path):
            self.load_state_dict(torch.load(saved_model_path, weights_only=False))
        else:
            print(f"WARNING: Model file not found '{saved_model_path}'. Generating a new model!")


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x) 
        x = F.relu(x) 
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flattening from feature maps to fully connected layers

        x = torch.flatten(x, 1) 
        
        # Fully connected layers

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x) 
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)

        return x
    
    def run_training_batch(self, learning_rate, dataloader):
        self.train(True)

        running_loss = 0.0
        for i, data in enumerate(dataloader):
            input_set, label_set = data
            input_set.to('cuda')
            label_set.to('cuda')
            
            self.optimizer.zero_grad()

            output_set = self(input_set)
        
            loss = self.loss_function(output_set, label_set)
            loss.backward()
        
            self.optimizer.step()
        
            running_loss += loss.item()
        
        print (f'Loss: {running_loss / len(dataloader):.4f}')
        epoch += 1


    def evaluate(self, data_loader: DataLoader) -> CatsEvaluationReport:
        if data_loader.batch_size != 1:
            raise ValueError("Expected DataLoader to have a batch size of 1 in evaluation mode!")
        
        self.eval()
        evaluation_report = CatsEvaluationReport()

        for inputs_set, labels_set in data_loader:
            evaluation_result = self.evaluate_single_image(inputs_set, labels_set)
            evaluation_report.add_result(evaluation_result)

        return evaluation_report.finalize()
    
    def evaluate_single_image(self, inputs_set, labels_set) -> CatEvaluationResult:
        self.eval()
        with torch.no_grad():
            inputs_set = inputs_set.to('cuda')
            if labels_set is not None:
                labels_set = labels_set.to('cuda')

            outputs_set = self(inputs_set) 

            evaluation_result = self.build_evaluation_result(labels_set, outputs_set)
            return evaluation_result

    def build_evaluation_result(self, labels_set, outputs_set):
        r = CatEvaluationResult()

        if labels_set is not None:
            numpy_labels = labels_set.flatten().cpu().numpy()
            max_label_index = numpy.argmax(numpy_labels)
            r.actual_label = CatsDataset.index_to_class_name(max_label_index)

        numpy_outputs = outputs_set.flatten().cpu().numpy()
        max_output_index = numpy.argmax(numpy_outputs)

        r.predicted_label = CatsDataset.index_to_class_name(max_output_index)

        r.bathrooom_cat_score = numpy_outputs[CatsDataset.class_name_to_index('bathroom-cat')]
        r.captain_score = numpy_outputs[CatsDataset.class_name_to_index('captain')]
        r.control_score = numpy_outputs[CatsDataset.class_name_to_index('control')]

        softmax_outputs = F.softmax(outputs_set, dim=1)
        softmax_numpy_outputs = softmax_outputs.flatten().cpu().numpy()

        r.bathrooom_cat_percent = softmax_numpy_outputs[CatsDataset.class_name_to_index('bathroom-cat')] * 100
        r.captain_percent = softmax_numpy_outputs[CatsDataset.class_name_to_index('captain')] * 100
        r.control_percent = softmax_numpy_outputs[CatsDataset.class_name_to_index('control')] * 100

        return r