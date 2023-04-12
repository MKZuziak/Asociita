# Libraries imports
import sys, warnings, torch, gc, logging
import numpy as np
from datasets import arrow_dataset
# Modules imports
from collections import Counter
from typing import Any, Generic, Mapping, TypeVar, Union
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn, optim


class FederatedModel:
    """This class is used to encapsulate the (PyTorch) federated model that
    we will train. It accepts only the PyTorch models and 
    provides a utility functions to initialize the model, 
    retrieve the weights or perform an indicated number of traning
    epochs.
    """
    def __init__(
        self,
        settings: dict,
        net: nn.Module,
        local_dataset: list[arrow_dataset.Dataset, arrow_dataset.Dataset] |
                                list[arrow_dataset.Dataset],
        node_name: int
    ) -> None:
        """Initialize the Federated Model. This model will be attached to a 
        specific client and will wait for further instructions.
        -------------
        Args:
            settings (dict): Settings for this run.
            net (nn.Module): Neural Network architecture that we want to use.
            local_dataset (list[...]): local dataset that will be used with this set.
            node_name (int): identifier for the node that uses this container.
        -------------
        Returns:
            None
        """
        self.device = None
        self.optimizer: optim.Optimizer = None
        
        #TODO: Add support for training on different GPUs.
        #gpus = preferences.gpu_config
        #expected_len = 11
        #if len(node_name) == expected_len:
            #if gpus:
                #gpu_name = gpus[int(node_name[10]) % len(gpus)]
            #self.device = torch.device(
                #gpu_name if torch.cuda.is_available() and gpus else "cpu",
            #)
            #logger.debug(f"Running on {self.device}")
        
        # Checks for all the necessary elements:
        assert settings, "Could not find settings, please ensure that a valid dictionary containing settings was passed in a function call."
        assert net, "Could not find net object, please ensure that a valid nn.Module was passed in a function call."
        assert local_dataset, "Could not find local dataset that should be used with that model. Pleasure ensure that local dataset was passed in a function call."
        
        self.net = net
        self.settings = settings
        self.node_name = node_name
        # If both, train and test data were provided
        if len(local_dataset) == 2:
            self.trainloader, self.testloader = self.prepare_data(local_dataset)
        # If only a test dataset was provided.
        elif len(local_dataset) == 1:
            self.testloader = self.prepare_data(local_dataset, only_test=True)
        else:
            raise "The provided dataset object seem to be wrong. Please provide list[train_set, test_set] or list[test_set]"


        # List containing all the parameters to update
        params_to_update = []
        for _, param in self.net.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)

        # Choosing an optimizer based on settings
        if self.settings['optimizer'] == "Adam":
            raise "Using Adam Optimizer has not been implemented yet."
            # TODO #self.optimizer = torch.optim.Adam(...
        elif self.settings['optimizer'] == "SGD":
            raise "Using SGD Optimizer has not been implemented yet."
            #TODO # self.optimizer = torch.optim.SGD(...
        elif self.settings['optimizer'] == "RMS":
            self.optimizer = optim.RMSprop(
                params_to_update,
                lr=self.settings["learning_rate"],)
        else:
            raise "The provided optimizer name may be incorrect or not implemeneted.\
            Please provide list[train_set, test_set] or list[test_set]"


    def prepare_data(
        self,
        local_dataset: list[arrow_dataset.Dataset, arrow_dataset.Dataset] |
                                list[arrow_dataset.Dataset],
        only_test: bool = False
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Convert training and test data stored on the local client into
        torch.utils.data.DataLoader.
        Args:
        -------------
        local_dataset (list[...]: local dataset that should be loaded into DataLoader)
        only_test (bool, default to False): If true, only a test set will be returned
        Returns
        -------------
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and test set
            or
            Tuple[torch.utils.data.DataLoader]: test set, if only_test == True.
        """
        if only_test == False:
            batch_size = self.settings["batch_size"]
            trainloader = torch.utils.data.DataLoader(
                local_dataset[0],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

            testloader = torch.utils.data.DataLoader(
                local_dataset[1],
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )
            #self.print_data_stats(trainloader) #TODO
            return trainloader, testloader
        else:
            testloader = torch.utils.data.DataLoader(
                local_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )
            return testloader

    def print_data_stats(self, trainloader: torch.utils.data.DataLoader) -> None: #TODO
        """Debug function used to print stats about the loaded datasets.
        Args:
            trainloader (torch.utils.data.DataLoader): training set
        """
        num_examples = {
            "trainset": len(self.training_set),
            "testset": len(self.test_set),
        }
        targets = []
        for _, data in enumerate(trainloader, 0):
            targets.append(data[1])
        targets = [item.item() for sublist in targets for item in sublist]
        logging.warning(f"{self.node_name}, {Counter(targets)}")
        logging.info(f"Training set size: {num_examples['trainset']}")
        logging.info(f"Test set size: {num_examples['testset']}")

    def get_weights_list(self) -> list[float]:
        """Get the parameters of the network.
        Raises
        ------
            Exception: if the model is not initialized it raises an exception
        Returns
        -------
            List[float]: parameters of the network
        """
        if self.net:
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        raise NotYetInitializedFederatedLearningError

    def get_weights(self):
        """Get the weights of the network.
        Raises
        ------
            Exception: if the model is not initialized it raises an exception
        Returns
        -------
            _type_: weights of the network
        """
        if self.net:
            return self.net.state_dict()
        raise NotYetInitializedFederatedLearningError

    def update_weights(self, avg_tensors) -> None:
        """This function updates the weights of the network.
        Raises
        ------
            Exception: _description_
        Args:
            avg_tensors (_type_): tensors that we want to use in the network
        """
        if self.net:
            self.net.load_state_dict(avg_tensors, strict=True)
        else:
            raise NotYetInitializedFederatedLearningError

    def store_model_on_disk(self) -> None:
        """This function is used to store the trained model
        on disk.
        Raises
        ------
            Exception: if the model is not initialized it raises an exception
        """
        if self.net:
            torch.save(
                self.net.state_dict(),
                "../model_results/model_" + self.node_name + ".pt",
            )
        else:
            raise NotYetInitializedFederatedLearningError

    def train(self) -> tuple[float, torch.tensor]:
        """Train the network and computes loss and accuracy.
        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized
        Returns
        -------
            Tuple[float, float]: Loss and accuracy on the training set.
        """
        if self.net:
            criterion = nn.CrossEntropyLoss()
            running_loss = 0.0
            total_correct = 0
            total = 0
            #self.net = self.net.to(self.device)

            self.net.train()
            for _, (data, target) in enumerate(self.trainloader, 0):

                self.optimizer.zero_grad()

                if isinstance(data, list):
                    data = data[0]

                #data, target = data.to(self.device), target.to(self.device)
                # forward pass, backward pass and optimization
                outputs = self.net(data)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == target).float().sum()

                loss = criterion(outputs, target)
                running_loss += loss.item()
                total_correct += correct
                total += target.size(0)

                self.optimizer.zero_grad()
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.net.zero_grad()
        
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            loss = running_loss / len(self.trainloader)
            accuracy = total_correct / total
            logger.info(f"Training loss: {loss}, accuracy: {accuracy}")

            return loss, accuracy
        raise NotYetInitializedFederatedLearningError