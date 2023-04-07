from asociita.components.orchestrator.orchestrator import Orchestrator
import unittest

class Orchestrator_tests(unittest.TestCase):

    def init_test(self):
        settings = dict()
        test_orchestrator = Orchestrator(settings=settings)
        self.assertEqual(0, test_orchestrator.state)

if __name__ == "__main__":
    tests_inst = Orchestrator_tests()
    tests_inst.init_test()