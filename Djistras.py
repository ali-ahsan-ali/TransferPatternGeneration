from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import heapq
from datetime import timedelta
from enum import Enum
import networkx as nx
from Graph import *
from typing import Dict, List, Set, Tuple, NamedTuple, TypeVar
from typing import Dict, List, Set, Tuple, NamedTuple, TypeVar
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta

from datetime import timedelta
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Set, Dict
import networkx as nx
import heapq

@dataclass(frozen=True)
class Path:
    cost: timedelta
    penalty: int
    nodes: List[Node]

    def __lt__(self, other):
        # Compare paths lexicographically (first by cost, then by penalty)
        return (self.cost, self.penalty) < (other.cost, other.penalty)

    def __repr__(self):
        return f"Path(cost={self.cost}, penalty={self.penalty}, nodes={self.nodes})"

    def __hash__(self):
        # Hash based on immutable attributes and nodes as a tuple
        return hash((self.cost, self.penalty, tuple(self.nodes)))
    
def is_not_dominated(path: Path, pareto_set: Set[Path]) -> bool:
    """
    Check if the given path is not dominated by any path in the Pareto set.
    """
    for p in pareto_set:
        if p.cost <= path.cost and p.penalty <= path.penalty:
            return False
    return True


def insert_and_clean(path: Path, pareto_set: Set[Path]) -> Set[Path]:
    """
    Insert a path into the Pareto set and remove dominated paths.
    """
    non_dominated = {p for p in pareto_set if not (path.cost <= p.cost and path.penalty <= p.penalty)}
    non_dominated.add(path)
    return non_dominated

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def mc_dijkstra(graph: nx.Graph, source: Node, target_station: str) -> Dict[Node, Set[Path]]:
    """
    Multicriteria Dijkstra algorithm to compute Pareto-optimal paths between two nodes.

    Parameters:
        graph: A networkx.Graph where nodes are instances of Node and edges contain 'penalty'.
        source: The source node.
        target_station: The target station name.

    Returns:
        A dictionary mapping each node to its set of Pareto-optimal paths.
    """
    logger.info("Starting MC Dijkstra algorithm")
    logger.debug("Source: %s, Target Station: %s", source, target_station)

    pareto_sets: Dict[Node, Set[Path]] = {node: set() for node in graph.nodes}
    candidate_paths = []
    heapq.heappush(candidate_paths, Path(cost=timedelta(0), penalty=0, nodes=[source]))

    while candidate_paths:
        # Select the path with the lexicographically smallest cost
        current_path = heapq.heappop(candidate_paths)
        current_node = current_path.nodes[-1]

        logger.debug("Processing path: %s", current_path)
        logger.debug("Current node: %s", current_node)

        # Add the current path to the Pareto set if it's not dominated
        if is_not_dominated(current_path, pareto_sets[current_node]):
            logger.debug("Path %s is not dominated. Adding to Pareto set for node %s.", current_path, current_node)
            pareto_sets[current_node] = insert_and_clean(current_path, pareto_sets[current_node])

            # If the current node is the target, continue to find all Pareto paths
            if current_node.station == target_station:
                logger.debug("Reached target station: %s with path: %s", target_station, current_path)
                continue

            # Explore neighbors
            for neighbor in graph.neighbors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor)
                cost = neighbor.time - current_node.time  # Should match edge_data['cost']
                penalty = edge_data.get("penalty", 0)
                new_cost = current_path.cost + cost
                new_penalty = current_path.penalty + penalty
                new_path = Path(cost=new_cost, penalty=new_penalty, nodes=current_path.nodes + [neighbor])

                if is_not_dominated(new_path, pareto_sets[neighbor]):
                    logger.debug("New path %s is not dominated for neighbor %s. Adding to candidates.", new_path, neighbor)
                    heapq.heappush(candidate_paths, new_path)

    logger.info("MC Dijkstra algorithm finished.")
    return pareto_sets


# Example Usage
if __name__ == "__main__":
    # Create a graph
    G = nx.Graph()

    # Define nodes
    node_s = Node("StationS", timedelta(hours=8), NODE_TYPE.DEPARTURE)
    node_a = Node("StationA", timedelta(hours=9), NODE_TYPE.ARRIVAL)
    node_b = Node("StationB", timedelta(hours=9, minutes=30), NODE_TYPE.TRANSFER)
    node_t = Node("StationT", timedelta(hours=10), NODE_TYPE.ARRIVAL)

    # Add nodes
    G.add_node(node_s)
    G.add_node(node_a)
    G.add_node(node_b)
    G.add_node(node_t)

    # Add edges with penalties
    G.add_edge(node_s, node_a, penalty=2)
    G.add_edge(node_a, node_b, penalty=3)
    G.add_edge(node_b, node_t, penalty=1)
    G.add_edge(node_a, node_t, penalty=4)

    # Run MC Dijkstra
    pareto_paths = mc_dijkstra(G, node_s, node_t)

    # Print results
    print("Pareto-optimal paths:")
    for path in pareto_paths:
        print(path)

