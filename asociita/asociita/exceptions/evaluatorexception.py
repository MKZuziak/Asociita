class Evaluation_Manager_Exception(Exception):
    """An error has occured on the level of Evaluation Manager"""

class Sample_Evaluator_Init_Exception(Evaluation_Manager_Exception):
    """Unable to initialize an instance of Sample_Evaluator. If the flag_sample_evaluator is set to True, Evaluation_Manager instance should receive the list of 
    all nodes and the number of iterations."""