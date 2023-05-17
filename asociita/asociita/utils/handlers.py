from asociita.components.nodes.federated_node import FederatedNode
import logging, csv
from typing import Any
import os

class Handler:
    """Common class for various utilities handling the data logs."""
    @staticmethod
    def log_model_metrics(iteration: int, 
                          model: Any,
                          logger,
                          ) -> None:
        """Used to log the model's metrics (on the orchestrator level).
        
        Args:
            iteration (int): Current iteration of the training.
            node_id (FederatedNode object): Federated Node object that we want to evaluate metrics on.
        """
        try:
            (
                loss,
                accuracy,
                fscore,
                precision,
                recall,
                test_accuracy_per_class,
                true_positive_rate,
                false_positive_rate
            ) = model.evaluate_model()
            metrics = {"loss":loss, 
                        "accuracy": accuracy, 
                        "fscore": fscore, 
                        "precision": precision,
                        "recall": recall, 
                        "test_accuracy_per_class": test_accuracy_per_class, 
                        "true_positive_rate": true_positive_rate,
                        "false_positive_rate": false_positive_rate,
                        "epoch": iteration}
            logger.info(f"Evaluating model after iteration {iteration} on node {model.node_name}. Results: {metrics}")
        except Exception as e:
            logger.warning(f"Unable to compute metrics. {e}")
    

    @staticmethod
    def save_model_metrics(iteration: int, 
                           model: Any,
                           logger = None,
                           saving_path: str = None,
                           file_name: str = 'metrics.csv',
                           log_to_screen: bool = False) -> None:
        """Used to save the model metrics.
        Args:
            - iteration (int): Current iteration of the training.
            - node (FederatedNode): FederatedNode Object which metrics we want to save.
            - participants (list[int]): List of id's of participants
                that are participating in this simulation. By default,
                it will be equal to all available nodes -> self.nodes_id
            - saving_path (str or path-like object): the saving path of the
                csv file - if none, the file will be saved in the current 
                working directory.
            - file_name (str): a desired file name for the metrics, default to
                'metrics.csv'.
        Returns:
            None"""
        try:
            (
                loss,
                accuracy,
                fscore,
                precision,
                recall,
                test_accuracy_per_class,
                true_positive_rate,
                false_positive_rate
            ) = model.evaluate_model()
            metrics = {"epoch": iteration,
                       "node": model.node_name,
                        "loss":loss, 
                        "accuracy": accuracy, 
                        "fscore": fscore, 
                        "precision": precision,
                        "recall": recall, 
                        "test_accuracy_per_class": test_accuracy_per_class, 
                        "true_positive_rate": true_positive_rate,
                        "false_positive_rate": false_positive_rate,
                        "epoch": iteration}
            if log_to_screen == True:
                logger.info(f"Evaluating model after iteration {iteration} on node {model.node_name}. Results: {metrics}")
        except Exception as e:
            logger.warning(f"Unable to compute metrics. {e}")
        path = os.path.join(saving_path, file_name)
        with open(path, 'a+', newline='') as saved_file:
                writer = csv.DictWriter(saved_file, list(metrics.keys()))
                if os.path.getsize(path) == 0:
                    writer.writeheader()
                writer.writerow(metrics)
    
    
    @staticmethod
    def save_csv_file(file,
                      saving_path: str = None,
                      file_name: str = 'metrics.csv') -> None:
        """Used to preserve the content of a csv file."""
        path = os.path.join(saving_path, file_name)
        with open(path, 'a+', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, list(file[0].keys()))
            writer.writeheader()
            for row in file:
                writer.writerow(row)