import sys
from Graph import Node
import heapq
from collections import namedtuple
import logging
import networkx as nx
from typing import Set, List, Dict
from datetime import datetime, timedelta
from Graph import Node, NODE_TYPE
import pickle

# Set up logging with only critical messages
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/run/media/ali/42daa914-34fa-4444-9ad9-f80f804fcb11/train/multiobjective_dijkstra.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

Label = namedtuple("Label", ["node", "cost", "pred"])

class MultiobjectiveDijkstra:
    def __init__(
        self,
        graph: nx.DiGraph,
        source: str,
        target: str,
        lower_bounds=None,
        upper_bound=None,
    ):
        """
        Initialize the Multiobjective Dijkstra Algorithm using NetworkX DiGraph.
        """
        logger.critical(f"Initializing MultiobjectiveDijkstra from {source} to {target}")
        logger.critical(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        self.graph = graph
        self.source = source
        self.target = target
        self.lower_bounds = lower_bounds if lower_bounds else {}
        self.upper_bound = upper_bound

        # Get the dimension of the cost vectors from the first edge
        self.d = 2

        # Initialize data structures
        self.H = []  # Priority queue
        self.L = {}  # Efficient labels for all nodes
        self.lastProcessedLabel = {}  # Last processed label index for each arc

        # Initialize efficient labels for all nodes
        for v in graph.nodes():
            self.L[v] = []

        # Initialize lastProcessedLabel for all arcs
        for u, v in graph.edges():
            self.lastProcessedLabel[(u, v)] = 0

    def dominates(self, cost1, cost2):
        """Check if cost1 dominates cost2 (is better in all dimensions)."""
        return all(cost1[i] <= cost2[i] for i in range(len(cost1))) and any(
            cost1[i] < cost2[i] for i in range(len(cost1))
        )

    def is_dominated(self, cost, label_set):
        """Check if cost is dominated by any cost in label_set."""
        if not label_set:
            return False
            
        return any(self.dominates(label.cost, cost) for label in label_set)

    def insert_priority_queue(self, label):
        """Insert a label into the priority queue (ordered lexicographically)."""
        heapq.heappush(self.H, (label.cost, id(label), label))

    def extract_lexmin(self):
        """Extract the lexicographically smallest label from the priority queue."""
        if not self.H:
            return None
        _, _, label = heapq.heappop(self.H)
        return label

    def nextCandidateLabel(
        self, v, L, lower_bound=None, upper_bound=None
    ):
        """
        Find the next candidate label for node v by processing unprocessed labels from predecessors.
        """
        # Initialize with max possible label
        label_v = Label(node=v, cost=(timedelta.max, float('inf')), pred=None)

        # For each incoming neighbor u
        incoming_neighbors = list(self.graph.predecessors(v))
        
        for u in incoming_neighbors:
            # Get the range of unprocessed labels
            start_idx = self.lastProcessedLabel.get((u, v), 0)
            end_idx = len(L[u])
            
            # Process all unprocessed labels from u
            for k in range(start_idx, end_idx):
                label_u = L[u][k]
                
                # Compute the new cost by adding the arc cost
                label_new = Label(
                    node=v, 
                    cost=(
                        label_u.cost[0] + self.graph[u][v]["cost"], 
                        label_u.cost[1] + self.graph[u][v]["penalty"]
                    ), 
                    pred=label_u
                )
                
                # Update last processed label index
                self.lastProcessedLabel[(u, v)] = k
                
                # Prune
                if label_new.cost[0] < self.upper_bound[0] and label_new.cost[1] < self.upper_bound[1]:
                    # Check if the new cost is dominated by any efficient label at v
                    if not self.is_dominated(label_new.cost, L[v]):
                        is_lexographically_smaller = True
                        for label in L[v]:
                            if not all(label_new.cost[i] < label.cost[i] for i in range(len(label_new.cost))):
                                is_lexographically_smaller = False
                                break
                        
                        if is_lexographically_smaller:
                            label_v = label_new
                            break  # Stop after finding first suitable label
        
        # Return None if no suitable label found
        if label_v.cost[0] == timedelta.max and label_v.cost[1] == float('inf'):
            return None
        
        return label_v

    def propagate(
        self, label: Label, w: Node, priority_queue, L, lower_bound=None, upper_bound=None
    ):
        """
        Propagate a label to node w.
        """
        v = label.node

        # Get the arc cost
        arc_cost = self.graph[v][w]["cost"]
        arc_penalty = self.graph[v][w]["penalty"]

        # Compute the new cost
        new_cost = (label.cost[0] + arc_cost, label.cost[1] + arc_penalty)
        
        # Prune
        if new_cost[0] > self.upper_bound[0] or new_cost[1] > self.upper_bound[1]:
            return priority_queue
        
        # Check if the new label is dominated
        if not self.is_dominated(new_cost, L[w]):
            new_label = Label(node=w, cost=new_cost, pred=label)
            acted = False
            for (index, label) in enumerate(self.H):
                if label[2].node == w:
                    self.H[index] = (new_cost, id(new_label), new_label)
                    acted = True
            
            if not acted:
                # Insert into priority queue
                self.insert_priority_queue(new_label)
        
        return priority_queue

    def run(self):
        for node in self.graph.nodes():
            if node.station == self.source and node.node_type == NODE_TYPE.TRANSFER:
                logger.critical(node)
                self.run_for_source(source_node=node)
    
    def run_for_source(self, source_node):
        """Run the Multiobjective Dijkstra Algorithm."""
        logger.critical("Starting MultiobjectiveDijkstra algorithm")
        
        # Initialize source label
        source_label = Label(
            node=source_node, cost=(timedelta(), 0), pred=None
        )
        self.insert_priority_queue(source_label)

        iteration = 0
        # Main loop
        while self.H:
            iteration += 1
            
            # Extract lexicographically smallest label
            current_label = self.extract_lexmin()
            if current_label is None:
                logger.critical("No more labels to process, terminating")
                break

            v = current_label.node

            # Add to efficient labels
            self.L[v].append(current_label)

            # Find next candidate label for v
            next_label = self.nextCandidateLabel(
                v, self.L, upper_bound=self.upper_bound
            )

            if next_label:
                self.insert_priority_queue(next_label)

            # Propagate to outgoing neighbors
            outgoing_neighbors = list(self.graph.successors(v))
            
            for w in outgoing_neighbors:
                self.H = self.propagate(
                    current_label, w, self.H, self.L, upper_bound=self.upper_bound
                )

            # Periodic logging
            if iteration % 1000 == 0:
                logger.critical(f"Iteration {iteration}, Queue size: {len(self.H)}")

        logger.critical(f"Algorithm terminated after {iteration} iterations")
        logger.critical(f"Total efficient labels found: {sum(len(labels) for labels in self.L.values())}")
        
        return 
    
    def arrival_chain_algorithm(labels: List[Label]):
        """Reconstruct the path from a label."""
        logger.critical(f"Reconstructing path from label with cost {label.cost}")
        path = [label.node]
        current = label
        while current:
            if current.node.node_type == NODE_TYPE.TRANSFER and current.pred != None and current.pred.node.node_type != NODE_TYPE.TRANSFER:
                path.append(current.node)
            if current.pred == None:
                path.append(current.node)
            current = current.pred
        path.reverse()
        
        return path

    def find_target_labels(self, efficient_labels: Dict):
        """
        Find labels for nodes matching the target station and optional node type.
        """
        logger.critical("Finding target labels across efficient labels")
        target_labels = []
        for node, labels in efficient_labels.items():
            if self.is_target_node(node):
                target_labels.extend(labels)
        
        # Sort target labels lexicographically
        target_labels.sort(key=lambda label: label.cost)
        
        logger.critical(f"Found {len(target_labels)} labels matching target criteria")
        
        return target_labels
    
    def is_target_node(self, node: Node) -> bool:
        """
        Check if a node matches the target criteria.
        """
        return node.station == self.target and node.node_type == NODE_TYPE.ARRIVAL