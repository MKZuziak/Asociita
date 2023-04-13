from asociita.components.nodes.federated_node import FederatedNode
import logging


class Handler:
    """Common class for various utilities handling the data logs."""
    @staticmethod
    def log_model_metrics(iteration: int, 
                          model
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
            logging.warning(metrics)
        except Exception as e:
            logging.warning(f"Unable to compute metrics. {e}")