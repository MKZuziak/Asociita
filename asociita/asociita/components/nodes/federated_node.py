from typing import Any

class FederatedNode:
    def __init__(self, 
                 node_id: int,
                 model: Any,
                 data: list[list[tuple], list[tuple]]) -> None:
        self.node_id = node_id
        self.model = self.load_model(model)
        self.train_data, self.test_data = self.load_data(data)
    

    def load_model(self, model: Any) -> Any:
        return model
    

    def load_data(self, 
                  data: list[list[tuple], list[tuple]]) -> tuple[list[tuple], list[tuple]]:
        return data
