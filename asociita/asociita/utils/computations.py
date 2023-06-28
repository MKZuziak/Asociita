from collections import OrderedDict
from typing import Any, Mapping, TypeVar
import torch
import itertools
from torch import Tensor

class Aggregators:
    """Defines the Aggregators class."""

    @staticmethod
    def compute_average(gradients: dict) -> dict[Any, Any]:
        """This function computes the average of the weights.
        Args:
            shared_data (Dict): weights received from the other nodes of the cluster
        Returns
        -------
            OrderedDict: the average of the weights
        """
        results = OrderedDict()
        for params in gradients.values():
            for key in params:
                if results.get(key) is None:
                    results[key] = params[key]
                else:
                    results[key] += params[key]

        for key in results:
            results[key] = torch.div(results[key], len(gradients))
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
    

class Subsets:
    """Defines the Subsets class - a set of static methods
    that helps with some set operations."""

    @staticmethod
    def form_superset(elements: list, 
                      remove_empty: bool = True,
                      return_dict: bool = True) -> list[list] | dict[tuple : None]:
        """Given a list of elements of a length N, forms a superset of a cardinality
        2^N, containing all the possible combinations of elements in the passed list.
        
        -------------
        Args
            elements (list): a list containing all the elements from which we want to form a superset.
            remove_empty (bool): if True, will remove an empty set that is formally also included in the superset.
            return_dict (bool): if True, will return the superset in a form {tuple: None} rather than [list].
       -------------
         Returns
            list[list] | dict[tuple : None]"""
        superset = list()
        for l in range(len(elements) + 1):
            for subset in itertools.combinations(elements, l):
                superset.append(list(subset))
        if remove_empty == True:
            superset.pop(0)
        
        if return_dict == True:
            superset = {tuple(coalition): None for coalition in superset}
            return superset
        else:
            return superset
    
    @staticmethod
    def form_loo_set(elements: list,
                     return_dict: bool = True) -> list[list] | dict[list : None]:
        """Given a list of elements of a length N, forms a set containing all the possible
        lists of N and N-1 length.
        -------------
        Args
            elements (list): a list containing all the elements from which we want to form a superset.
            return_dict (bool): if True, will return the superset in a form {tuple: None} rather than [list].
       -------------
         Returns
            list[list] | dict[tuple : None]"""
        loo_set = list()
        for l in range(len(elements) - 1, len(elements) + 1):
            for subset in itertools.combinations(elements, l):
                loo_set.append(list(subset))
        
        if return_dict == True:
            loo_set = {tuple(coalition) : None for coalition in loo_set}
            return loo_set
        else:
            return loo_set


    def select_subsets(coalitions: dict | list,
                       searched_node: int) -> dict[tuple : Any]:
        """Given a dict or list of possible coalitions and a searched node, will return
        every possible coalition which DO NOT CONTAIN the searche node.
        -------------
        Args
            coalitions (dict): a superset or a set of all possible coalitions, mapping
                each coalition to some value, e.g. {(1, 2, 3,): None}
            searched_node (int): an id of the searched node.
       -------------
         Returns
            dict[tuple : Any]"""
        if type(coalitions) == dict:
            subsets = {nodes: model for nodes, model in coalitions.items() 
                    if searched_node not in nodes}
        if type(coalitions) == list:
            subsets = [nodes for nodes in coalitions if searched_node not in nodes]
        return subsets
