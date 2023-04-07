import types, logging
from typing import Any

class Orchestrator():
    def __init__(self, settings: dict) -> None:
        """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool.
        
        Parameters
        ----------
        settings : dict
            Dictionary object cotaining all the settings of the orchestrator.
        
        Returns
        ----------
        None"""
        self.state = 1 # Attribute controlling the state of the object.
                         # 0 - initialized, resting
                         # 1 - initialized, in run-time
        
        self.settings = settings # Settings attribute (dict)

        self.state = 0
    
    def load_model(self, model: Any) -> None:
        """Loads the global model that will be used as the orchestrator's main model
        In contrast to the client object, load_model and load_data are separated in the
        instance of the orchestrator class.
        
        Parameters
        ----------
        model : Any
            Compiled or pre-compiled model that will be used by the instance of the class.
        
        Returns
        ----------
        None"""
        assert self.state == 0, f"Object {self} is not resting, previous operation is still active."
        self.state = 1
        
        try:
            self.model = model
        except:
            logging.critical("Failed to load the model")
        
        if self.model != None:
            self.state = 0
