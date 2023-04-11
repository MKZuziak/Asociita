from typing import Any
from datasets import arrow_dataset

class FederatedNode:
    def __init__(self, 
                 node_id: int) -> None:
        """An abstract object representing a single node in the federated training.
        ------------
        Arguments:
        node_id (int): an int identifier of a node
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
        self.state = 0


    def prepare_node(self, model, data):
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
       self.model = model
       self.train_data = data[0]
       self.test_data = data[1]


       if self.model != None and self.train_data != None \
        and self.test_data != None:
           self.state = 0
       else:
           # TODO: LOGGING INFO
           pass
