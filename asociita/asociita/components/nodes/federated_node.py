from typing import Any
from datasets import arrow_dataset
import logging

from asociita.models.pytorch.federated_model import FederatedModel

class FederatedNode:
    def __init__(self, 
                 node_id: int,
                 settings: dict) -> None:
        """An abstract object representing a single node in the federated training.
        ------------
        Arguments:
        node_id (int): an int identifier of a node
        settings (dict): a dictionary containing settings for the node
        ------------
        Returns:
        None"""
        self.state = 1 # Attribute controlling the state of the object.
                         # 0 - initialized, resting
                         # 1 - initialized, in run-time
        
        self.node_id = node_id
        self.model = None
        self.train_data = None
        self.test_data = None
        self.settings = settings
        self.state = 0


    def prepare_node(self, 
                     model: Any, 
                     data: arrow_dataset.Dataset):
       """Prepares node for the training, given the passed model 
       and dataset.
       model (Any): compiled or pre-compiled model to be trained
       dataset (list[datasets.arrow_dataset.Dataset, 
                datasets.arrow_dataset.Dataset]): a list[train_data, test_data]
                wrapped in a pre-compiled HuggingFace object.
        ------------
        Returns:
        None"""
       
       self.state = 1
       self.train_data = data[0]
       self.test_data = data[1]
       
       self.model = FederatedModel(
           settings=self.settings["model_settings"]
       )

       if self.model != None and self.train_data != None \
        and self.test_data != None:
           self.state = 0
       else:
           # TODO: LOGGING INFO
           pass
    

    def train_local_model(
        self,
        # results: dict | None = None,
    ) -> tuple[list[float], list[float], list[float]]:
        """This function starts the server phase of the federated learning.
        In particular, it trains the model locally and then sends the weights.
        Then the updated weights are received and used to update
        the local model.
        Args:
            federated_model (FederatedModel): _description_
        Returns
        -------
            Tuple[List[float], List[float], List[float]]: _description_
        """
        logging.info(f"Starting training on node {self.node_id}")
        loss_list: list[float] = []
        accuracy_list: list[float] = []
        epsilon_list: list[float] = []

        local_epochs = self.node_settings['local_epochs'] # TODO
        
        for _ in range(local_epochs):
            metrics = self.local_training()
            loss_list.append(metrics["loss"])
            accuracy_list.append(metrics["accuracy"])
            if metrics.get("epsilon", None):
                epsilon_list.append(metrics["epsilon"])
        
        logging.debug("2")
        return Weights(
            weights=self.federated_model.get_weights(),
            sender=self.node_id,
            epsilon=metrics["epsilon"],
        )
    

    def local_training(
        self,
        differential_private_train: bool,
    ) -> dict:
        """_summary_.
        Args:
            differential_private_train (bool): _description_
            federated_model (FederatedModel): _description_
        Returns
        -------
            dict: _description_
        """
        loss, accuracy = self.federated_model.train()
        return {"loss": loss, "accuracy": accuracy, "epsilon": epsilon}
