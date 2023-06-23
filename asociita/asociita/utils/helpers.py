import json


class Helpers:
    @staticmethod
    def load_from_json(path: 'str', convert_keys: bool = False) -> dict:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        if convert_keys == True:
            if data.get('transformations'):
                data['transformations'] = {int(key): value for key, value in data['transformations'].items()}
            if data.get('imbalanced_clients'):
                data['imbalanced_clients'] = {int(key): value for key, value in data['imbalanced_clients'].items()}
        return data
    

    @staticmethod
    def chunker(seq: iter, size: int):
        """Helper function for splitting an iterable into a number
        of chunks each of size n. If the iterable can not be splitted
        into an equal number of chunks of size n, chunker will return the
        left-overs as the last chunk.
            
        Parameters
        ----------
        sqe: iter
            An iterable object that needs to be splitted into n chunks.
        size: int
            A size of individual chunk
        Returns
        -------
        Generator
            A generator object that can be iterated to obtain chunks of the original iterable.
        """
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))