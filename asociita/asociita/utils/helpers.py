import json


class Helpers:
    @staticmethod
    def load_from_json(path: 'str') -> dict:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data