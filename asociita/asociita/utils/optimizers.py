from collections import OrderedDict
from torch import zeros
import torch


class Optimizers():
    def __init__(self,
                 weights: OrderedDict,
                 settings: dict) -> None:
        # Seting up a device for the Optimizer. Please note, that the device must be the same as this
        # that the nets were placed on. Otherwise, PyTorch will raise an exception trying to combine
        # data placec on CPU and GPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.previous_delta = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        self.previous_momentum = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        self.optimizer = settings['name']
        self.learning_rate = torch.tensor(settings['learning_rate'])

        # Selecting a proper centralised optimizer and placing all the tensors on the same device.
        if self.optimizer == 'Simple':
            self.learning_rate = self.learning_rate.to(self.device)
        elif self.optimizer == "FedAdagard":
            self.b1 = torch.tensor(settings['b1'])
            self.tau = torch.tensor(settings['tau'])
            self.b1, self.tau, self.learning_rate = self.b1.to(self.device), self.tau.to(self.device), self.learning_rate.to(self.device)
        elif self.optimizer == "FedYogi" or self.optimizer == "FedAdam":
            self.b1 = torch.tensor(settings['b1'])
            self.b2 = torch.tensor(settings['b2'])
            self.tau = torch.tensor(settings['tau'])
            self.b1, self.b2, self.tau, self.learning_rate = self.b1.to(self.device), self.b2.to(self.device), self.tau.to(self.device), self.learning_rate.to(self.device)
        else:
            "Wrong optimizer's name was provided. Unable to retrieve parameters!"


    def fed_optimize(self,
                     weights: OrderedDict,
                     delta: OrderedDict) -> OrderedDict:
        if self.optimizer == "Simple":
            updated_weights = self.SimpleFedopt(weights = weights,
                                                delta = delta,
                                                learning_rate = self.learning_rate)
            return updated_weights
        elif self.optimizer == "FedAdagard":
            updated_weights = self.FedAdagard(weights = weights,
                                              delta = delta,
                                              b1 = self.b1,
                                              tau = self.tau,
                                              learning_rate = self.learning_rate)
            return updated_weights
        elif self.optimizer == "FedYogi":
            updated_weights = self.FedYogi(weights = weights,
                                              delta = delta,
                                              b1 = self.b1,
                                              b2 = self.b2,
                                              tau = self.tau,
                                              learning_rate = self.learning_rate)
            return updated_weights
        elif self.optimizer == "FedAdam":
            updated_weights = self.FedAdam(weights = weights,
                                              delta = delta,
                                              b1 = self.b1,
                                              b2 = self.b2,
                                              tau = self.tau,
                                              learning_rate = self.learning_rate)
            return updated_weights
        else:
            raise "Wrong optimizer was provided. Available optimizers: FedAdagard, FedYogi, FedAdam."

    @staticmethod
    def SimpleFedopt(weights: OrderedDict,
                     delta: OrderedDict,
                     learning_rate: float):
        """Adds gradients to the central weights, concluding one round of Federated Training."""
        updated_weights = OrderedDict.fromkeys(weights.keys(), 0)
        for key in weights:
            updated_weights[key] = weights[key] + (learning_rate * delta[key])
        return updated_weights
    

    def FedAdagard(self,
                   weights: OrderedDict,
                   delta: OrderedDict,
                   b1: float, 
                   tau: float,
                   learning_rate: float) -> OrderedDict:
        # Defining the current delta.
        current_delta = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        current_momentum = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        updated_weights = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())


        for row_key in current_delta.keys():
            current_delta[row_key] = b1 * self.previous_delta[row_key] + (1 - b1) * delta[row_key]
        
        for row_key in current_momentum.keys():
            current_momentum[row_key] = self.previous_momentum[row_key] + (current_delta[row_key] ** 2)

        for row_key in updated_weights.keys():
            updated_weights[row_key] = (weights[row_key].to(self.device)) + (learning_rate * (current_delta[row_key] / (torch.sqrt(current_momentum[row_key]) + tau)))
        
        self.previous_delta = current_delta
        self.previous_momentum = current_momentum

        return updated_weights


    def FedYogi(self,
                weights: OrderedDict,
                delta: OrderedDict,
                b1: float,
                b2: float,
                tau: float,
                learning_rate: float):
        # Defining the current delta.
        current_delta = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        current_momentum = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        updated_weights = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())

        for row_key in current_delta.keys():
            current_delta[row_key] = b1 * self.previous_delta[row_key] + (1 - b1) * delta[row_key]
        
        for row_key in current_momentum.keys():
            current_momentum[row_key] = self.previous_momentum[row_key] - ((1 - b2) * (current_delta[row_key] ** 2) * torch.sign((self.previous_momentum[row_key] - (current_delta[row_key] ** 2))))

        for row_key in updated_weights.keys():
            updated_weights[row_key] = weights[row_key] + learning_rate * (current_delta[row_key] / (torch.sqrt(current_momentum[row_key]) + tau))
        
        self.previous_delta = current_delta
        self.previous_momentum = current_momentum
        
        return updated_weights

    
    def FedAdam(self,
                weights: OrderedDict,
                delta: OrderedDict,
                b1: float,
                b2: float,
                tau: float,
                learning_rate: float):
        # Defining the current delta.
        current_delta = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        current_momentum = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())
        updated_weights = OrderedDict((key, zeros(weights[key].size(), device=self.device)) for key in weights.keys())

        for row_key in current_delta.keys():
            current_delta[row_key] = b1 * self.previous_delta[row_key] + (1 - b1) * delta[row_key]
        
        for row_key in current_momentum.keys():
            current_momentum[row_key] = (b2 * self.previous_momentum[row_key]) + ((1 - b2) * (current_delta[row_key] ** 2))

        for row_key in updated_weights.keys():
            updated_weights[row_key] = weights[row_key] + (learning_rate * (current_delta[row_key] / (torch.sqrt(current_momentum[row_key]) + tau)))
        
        self.previous_delta = current_delta
        self.previous_momentum = current_momentum
        
        return updated_weights