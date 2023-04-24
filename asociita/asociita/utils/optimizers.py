from collections import OrderedDict
import numpy as np
from torch import zeros
from typing import Union


class Optimizers():
    def __init__(self,
                 weights: OrderedDict) -> None:
        self.previous_delta = OrderedDict((key, zeros(weights[key].size())) for key in weights.keys())
        self.previous_momentum = OrderedDict((key, zeros(weights[key].size())) for key in weights.keys())
    

    def fed_optimize(self,
                     optimizer: str,
                     settings: dict,
                     weights: OrderedDict,
                     delta: OrderedDict) -> OrderedDict:
        if optimizer == "Simple":
            learning_rate = settings['learning_rate']
            updated_weights = self.SimpleFedopt(weights=weights,
                                                delta = delta,
                                                learning_rate=learning_rate)
            return updated_weights
        elif optimizer == "FedAdagard":
            learning_rate = settings['learning_rate']
            b1 = settings['b1']
            tau = settings['tau']
            updated_weights = self.FedAdagard(weights=weights,
                                              delta=delta,
                                              b1=b1,
                                              tau=tau,
                                              learning_rate=learning_rate)
            return updated_weights
        elif optimizer == "FedYogi":
            raise "FedYogi optimizer is not optimized yet"
        elif optimizer == "FedAdam":
            raise "FedAdam optimizer is not optimizer yet."
        else:
            raise "Wrong optimizer was provided. Available optimizers: FedAdagard, FedYogi, FedAdam."

    @staticmethod
    def SimpleFedopt(weights: OrderedDict,
                     delta: OrderedDict,
                     learning_rate: float):
        """Adds gradients to the central weights, concluding one round of Federated Training."""
        updated_weights = OrderedDict.fromkeys(weights.keys(), 0)
        for key in weights:
            updated_weights[key] = weights[key] - (learning_rate * delta[key])
        return updated_weights
    

    def FedAdagard(self,
                   weights: OrderedDict,
                   delta: OrderedDict,
                   b1: float, 
                   tau: float,
                   learning_rate: float) -> OrderedDict:
        # Defining the current delta.
        current_delta = OrderedDict((key, zeros(weights[key].size())) for key in weights.keys())
        current_momentum = OrderedDict((key, zeros(weights[key].size())) for key in weights.keys())
        updated_weights = OrderedDict((key, zeros(weights[key].size())) for key in weights.keys())
        
        for row_key in current_delta.keys():
            current_delta[row_key] = b1 * self.previous_delta[row_key] - (1 - b1) * delta[row_key]
        
        for row_key in current_momentum.keys():
            current_momentum[row_key] = self.previous_momentum[row_key] + current_delta[row_key] ** 2

        for row_key in updated_weights.keys():
            updated_weights[row_key] = weights[row_key] + learning_rate * (current_delta[row_key] / (np.sqrt(current_momentum[row_key]) + tau ))
        
        self.previous_delta = current_delta
        self.previous_momentum = current_momentum

        return updated_weights


    def FedYogi(gradients,
                b1,
                b2,
                tau,
                learning_rate):
        ...
    


    def FedAdam(gradients,
                b1,
                b2,
                tau,
                learning_rate):
        ...
        