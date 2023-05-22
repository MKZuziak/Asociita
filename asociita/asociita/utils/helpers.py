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