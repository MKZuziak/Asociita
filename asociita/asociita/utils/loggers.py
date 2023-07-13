import logging

#Orchestrator_logger = logging.getLogger("")

# create loggers
class Loggers:
    @staticmethod
    def orchestrator_logger():
        # Creating a head logger for the orchestrator
        orchestrator_logger = logging.getLogger("orchestrator_logger")
        orchestrator_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        orchestrator_logger.addHandler(ch)
        orchestrator_logger.propagate = False
        return orchestrator_logger
    

    @staticmethod
    def node_logger():
        # Creating a head logger for the nodes
        node_logger = logging.getLogger("node_logger")
        node_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        zh = logging.StreamHandler()
        zh.setLevel(logging.DEBUG)
        zh.setFormatter(formatter)
        node_logger.addHandler(zh)
        node_logger.propagate = False
        return node_logger
    

    @staticmethod
    def model_logger():
        # Creating a head loger for the models
        model_logger = logging.getLogger("model_logger")
        model_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        model_logger.addHandler(sh)
        model_logger.propagate = False
        return model_logger
