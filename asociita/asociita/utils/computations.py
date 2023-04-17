from collections import OrderedDict
from typing import Any, Mapping, TypeVar
import torch
from torch import Tensor

class Aggregators:
    """Define the Aggregators class."""

    @staticmethod
    def compute_average(shared_data: dict) -> dict[Any, Any]:
        """This function computes the average of the weights.
        Args:
            shared_data (Dict): weights received from the other nodes of the cluster
        Returns
        -------
            OrderedDict: the average of the weights
        """
        models = list(shared_data.values())
        results: OrderedDict = OrderedDict()

        for model in models:
            for key in model:
                if results.get(key) is None:
                    if torch.cuda.is_available():
                        model[key] = model[key].to("cuda:0")

                    results[key] = model[key]
                else:
                    if torch.cuda.is_available():
                        model[key] = model[key].to("cuda:0")
                    results[key] = results[key].add(model[key])

        for key in results:
            results[key] = torch.div(results[key], len(models))
        return results
    

    @staticmethod
    def add_gradients(model_weights: OrderedDict, gradient: OrderedDict) -> OrderedDict:
        """Adds gradients to the central weights, concluding one round of Federated Training."""
        updated_weights = OrderedDict.fromkeys(model_weights.keys(), 0)
        for key in model_weights:
            updated_weights[key] = model_weights[key] - gradient[key]
        return updated_weights



    @staticmethod
    def compute_distance_from_mean(shared_data: dict, average_weights: dict) -> dict:
        """This function takes as input the weights received from all the nodes and
        the average computed by the server.
        It computes the distance between the average and
        the weights received from each node per each layer.
        Then it computes the mean of the distances per each layers.
        The function returns a dictionary containing the mean of
        the distances per each node.
        Args:
            shared_data (Dict): the weights received from the other nodes of the cluster
            average_weights (Dict): the average of the weights
        Returns
        -------
            Dict: a dictionary containing the mean of the distances per each node
        """
        distances = {}
        for node_name, models in shared_data.items():
            mean = []
            for layer_name in models:

                mean.append(
                    torch.mean(
                        torch.subtract(models[layer_name], average_weights[layer_name]),
                    ),
                )

            distances[node_name] = torch.mean(torch.stack(mean))

        return 