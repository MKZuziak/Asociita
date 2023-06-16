from components.settings.settings import unit_test_settings
from components.orchestrator.generic_orchestrator import unit_test_genorchestrator
from dataset.dataset_generation_hetsize import unit_test_hetgen

def test_suite():
    # Test 1: Generic Setting Class Object Initialization
    test_settings = unit_test_settings()
    # Test 2: Generic Orchestrator Class Object Initialization
    test_genorch = unit_test_genorchestrator()

if __name__ == "__main__":
    test_suite()