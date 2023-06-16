from components.settings.settings import unit_test_settings
from components.orchestrator.generic_orchestrator import unit_test_genorchestrator
from components.orchestrator.generic_orchestrator import unit_test_genorchestrator_warchiver
from dataset.dataset_generation_hetsize import unit_test_hetgen

def test_suite():
    # Test 1: Generic Setting Class Object Initialization
    test_settings = unit_test_settings()
    # Test 2: Generic Orchestrator Class Object Initialization
    test_genorch = unit_test_genorchestrator()
    print("The simulation of generic orchestrator training and managing nodes was passed successfully.")

def test_suite2():
    # Test 2: Generic Orchestrator Class Object Initialization
    test_genorch = unit_test_genorchestrator_warchiver()
    print("The simulation of generic orchestrator training and managing nodes was passed successfully.")

if __name__ == "__main__":
    test_suite()
    test_suite2()