from asociita.exceptions.settings.settingexception import SettingsObjectException

class InitMethodException(SettingsObjectException):
    'Provided method of initializaiton is not supported. Settings object can be either initialized' \
          'from the passed dictionary containing appropiate key-words or by passing all the necessary key-words' \
          'explicitly in the function call.'

class OrchestratorSettingsMissing(SettingsObjectException):
    "The provided orchestrator settings are incomplete. The orchestrator settings should contain" \
    "the following parameters: the number of iterations ('iterations': int), number of nodes ('number_of_nodes': int)," \
    "local warm start ('local_warm_start': bool), sample size ('sample_size':int) and metrics save path ('metrics_save_path' : str or Path"