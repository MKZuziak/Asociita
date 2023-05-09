import logging
import datasets
from asociita.datasets.load_mnist import load_mnist
import json

def load_data(settings:dict) -> list[datasets.arrow_dataset.Dataset,
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
    
    dataset_name = settings['dataset_name']
    if dataset_name == 'mnist':
        loaded_dataset = load_mnist(settings=settings)
        if settings['save_dataset'] == True:
            json.dump(loaded_dataset)
            return loaded_dataset
        else:
            return loaded_dataset
    else:
        logging.warning("Wrong name of the dataset. Please provide a valid name.")
       