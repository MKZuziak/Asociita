import sys, os
from pathlib import Path

p = Path(__file__).parents[1]
p = os.path.join(p, 'asociita')
sys.path.insert(0, p)

from asociita.components.nodes.federated_node import FederatedNode