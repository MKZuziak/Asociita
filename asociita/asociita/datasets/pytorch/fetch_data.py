import logging
from asociita.datasets.pytorch.load_mnist import load_mnist

def load_dataset(dataset_name: str, settings:dict) -> list[list[tuple], list[tuple]]:
    if dataset_name == 'mnist':
        return load_mnist(settings=settings)
    else:
        logging.warning("Wrong name of the dataset. Please provide a valid name.")
       