from asociita.components.settings.settings import Settings

def init_settings(orchestrator_type: str,
                  initialization_method: str = 'dict',
                  dict_settings: dict = None,
                  allow_default: bool = True,
                  **kwargs):
    """Factory function for initializing instance of an appropiate settings object.
    Parameters
    ----------
    orchestrator_type: str
        The type of the orchestrator for which the settings object should be returned.
    initialization_method: str, default to 'dict' 
        The method of initialization. Either 'dict' or 'kwargs'.
    dict_settings: dict, default to None
        A dictionary containing all the relevant settings if the initialization is made from dir. 
    allow_default: bool, default to True
        A logical switch to allow using default values in absence of passed values.
    Returns
    -------
    None"""
    if orchestrator_type == 'general':
        return Settings(allow_default=allow_default,
                        initialization_method=initialization_method,
                        dict_settings=dict_settings)