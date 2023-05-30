import datasets
from datasets import load_dataset
from asociita.datasets.shard_transformation import Shard_Transformation
from asociita.datasets.shard_splits import Shard_Splits
from asociita.utils.showcase import save_random
from asociita.utils.handlers import Handler
import copy
import pandas as pd
import numpy as np

def load_fmnist(settings: dict) -> list[datasets.arrow_dataset.Dataset,
                                       list[list[list[datasets.arrow_dataset.Dataset]]]]:
    """Loads the MNIST dataset, splits it into the number of shards, pre-process selected
    shards (subsets) and returns in a following format:
    list[   
        "Orchestrator Data"[
            Dataset
            ],   
        "Agents Data"[
            "Agent N"[
                "Train Data"[
                Dataset
                ],
                "Test Data"[
                Dataset
                ]
            ]]]
    Where all 'Datasets' are an instances of hugging face container datasets.arrow_dataset.Dataset
    ---------
    Args:
        settings (dict) : A dictionary containing all the dataset settings.
    Returns:
        list[datasets.arrow_dataset.Dataset,
                                       list[list[list[datasets.arrow_dataset.Dataset]]]]"""
    
    # Using the 'test' data as a orchestrator validaiton set.
    orchestrator_data = load_dataset('fashion_mnist', split='test')
    # Using the 'train' data as a dataset reserved for agents
    dataset = load_dataset('fashion_mnist', split='train')

    # Type: Random Uniform (Sharding) -> Same size, random distribution
    if settings['split_type'] == 'random_uniform':
        return [orchestrator_data, Shard_Splits.random_uniform(dataset=dataset, settings=settings)]
    
    # Type: Uniform with Imbalanced Classes -> Samze size, different (random) distributions with heavy imbalance on selected clients
    if settings['split_type'] == 'random_imbalanced':
        return [orchestrator_data, Shard_Splits.random_imbalanced(dataset=dataset, settings=settings)]

    # Type: Same Dataset -> One dataset copied n times.
    elif settings['split_type'] == 'replicate_same_dataset':
        return [orchestrator_data, Shard_Splits.replicate_same_dataset(dataset=dataset, settings=settings)]
    
    # Type: Blocks - One dataset copied inside one block (cluster)
    elif settings['split_type'] == 'split_in_blocks':
        return [orchestrator_data, Shard_Splits.replicate_same_dataset(dataset=dataset, settings=settings)]
    
    else:
        raise "Unable to generate the dataset. Provided split-type does not exist."
            
