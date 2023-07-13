from typing import Any
from datasets import arrow_dataset
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.loggers import Loggers

node_logger = Loggers.node_logger()

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
       ------------
       Arguments:
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
           settings=self.settings["model_settings"],
           net = model,
           local_dataset = data,
           node_name=self.node_id
       )

       if self.model != None and self.train_data != None \
        and self.test_data != None:
           self.state = 0
       else:
           # TODO: LOGGING INFO
           pass
    

    def train_local_model(
        self,
        mode: str
    ) -> tuple[list[float], list[float], list[float]]:
        """This function starts the server phase of the federated learning.
        In particular, it trains the model locally and then sends the weights.
        Then the updated weights are received and used to update
        the local model.
        -------------
        Args:
        node (FederatedNode object): Node that we want to train.
        mode (str): Mode of the training. 
            Mode = 'weights': Node will return model's weights.
            Mode = 'gradients': Node will return model's gradients.
        -------------
        Returns:
            Tuple[List[float], List[float], List[float]]: _description_
        """
        node_logger.info(f"Starting training on node {self.node_id}")
        loss_list: list[float] = []
        accuracy_list: list[float] = []

        local_epochs = self.settings['local_epochs']

        if mode == 'gradients':
            self.model.preserve_initial_model()
        
        for _ in range(local_epochs):
            metrics = self.local_training()
            loss_list.append(metrics["loss"])
            accuracy_list.append(metrics["accuracy"])
        
        node_logger.info(f"Results of training on node {self.node_id}: {accuracy_list}")
        if mode == 'weights:':
            return (
                self.node_id,
                self.model.get_weights()
                )
        elif mode == 'gradients':
            return (
                self.node_id,
                self.model.get_gradients()
            )
        else:
            node_logger.info("No mode was provided, returning only model's weights")
            return (
                self.node_id,
                self.model.get_weights()
                )

    def local_training(
        self,
    ) -> dict[int, int]:
        """Helper method for performing one epoch of local training.
        Performs one round of Federated Training and pack the
        results (metrics) into the appropiate data structure.
        Args:
            self
        Returns
        -------
            dict[int, int]: metrics from the training.
        """
        loss, accuracy = self.model.train()
        return {"loss": loss, "accuracy": accuracy}