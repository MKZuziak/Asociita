class SimulationSettingsException(Exception):
    """The Simulation Configuration is missing key-arguments."""

class DatasetSettingsException(Exception):
    """The Dataset Configuration is missing key-arguments"""

class ArchiverSettingsException(Exception):
    """The Archiver Configuration is missing key-arguments"""

class SettingsObjectException(Exception):
    """The settings object is missing key-arguments"""