from asociita.components.nodes.federated_node import FederatedNode
import unittest
import logging

class Node_Tests(unittest.TestCase):

    def init_test(self):
        node_id = 0
        model = []
        dataset = []
        test_node = FederatedNode(node_id=node_id,
                                  model=model,
                                  dataset=dataset)
        
        self.assertIsNotNone(test_node)
        self.assertEqual(node_id, test_node.node_id)
        self.assertEqual(0, test_node.state)
        print("Initialization tests passed successfully")


if __name__ == "__main__":
    test_instance = Node_Tests()
    test_instance.init_test()