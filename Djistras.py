import sys
from Graph import Node
import heapq
from collections import namedtuple
import logging
import networkx as nx
from typing import Set, List, Dict
from datetime import datetime, timedelta


# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("multiobjective_dijkstra.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

Label = namedtuple("Label", ["node", "cost", "pred"])

class MultiobjectiveDijkstra:
    def __init__(
        self,
        graph: nx.DiGraph,
        source: Node,
        target: str,
        lower_bounds=None,
        upper_bound=None,
    ):
        """
        Initialize the Multiobjective Dijkstra Algorithm using NetworkX DiGraph.

        Parameters:
        - graph: NetworkX DiGraph with 'cost' attribute for each edge representing d-dimensional cost vector
        - source: Source node
        - target: Target node (for one-to-one version)
        - lower_bounds: Dictionary with nodes as keys and lower bound vectors as values (for one-to-one)
        - upper_bound: Upper bound label at target (for one-to-one)
        """
        logger.info(f"Initializing MultiobjectiveDijkstra from {source} to {target}")
        logger.info(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        
        self.graph = graph
        self.source = source
        self.target = target
        self.lower_bounds = lower_bounds if lower_bounds else {}
        self.upper_bound = upper_bound

        # Get the dimension of the cost vectors from the first edge
        self.d = 2
        logger.info(f"Cost dimension: {self.d}")

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
            
        logger.debug(f"Initial lower bounds: {self.lower_bounds}")
        logger.debug(f"Initial upper bound: {self.upper_bound}")
        logger.info("Initialization complete")

    def dominates(self, cost1, cost2):
        """Check if cost1 dominates cost2 (is better in all dimensions)."""
        logger.debug(f"Checking dominance: {cost1} vs {cost2}")
        result = all(cost1[i] <= cost2[i] for i in range(len(cost1))) and any(
            cost1[i] < cost2[i] for i in range(len(cost1))
        )
        logger.debug(f"Dominance result: {result}")
        return result

    def is_dominated(self, cost, label_set):
        """Check if cost is dominated by any cost in label_set."""
        logger.debug(f"Checking if cost {cost} is dominated, label set size: {len(label_set)}")
        if not label_set:
            logger.debug(f"Cost {cost} is not dominated (empty label set)")
            return False
            
        result = any(self.dominates(label.cost, cost) for label in label_set)
        
        logger.debug(f"Is dominated result: {result}")
        return result

    def insert_priority_queue(self, label):
        """Insert a label into the priority queue (ordered lexicographically)."""
        heapq.heappush(self.H, (label.cost, id(label), label))
        logger.debug(f"Inserted label into queue: node={label.node}, cost={label.cost}, queue size={len(self.H)}")

    def extract_lexmin(self):
        """Extract the lexicographically smallest label from the priority queue."""
        if not self.H:
            logger.debug("Priority queue is empty, returning None")
            return None
        _, _, label = heapq.heappop(self.H)
        logger.debug(f"Extracted label: node={label.node}, cost={label.cost}")
        return label

    def nextCandidateLabel(
        self, v, L, lower_bound=None, upper_bound=None
    ):
        """
        Find the next candidate label for node v by processing unprocessed labels from predecessors.

        Parameters:
        - v: Current node
        - lastProcessedLabel: Dictionary tracking last processed label index
        - L: Dictionary of efficient labels
        - lower_bound: Lower bound vector for node v (one-to-one version)
        - upper_bound: Upper bound label at target (one-to-one version)

        Returns:
        - The next candidate label for v, or None if none exists
        """
        logger.debug(f"Finding next candidate label for node {v}")
        logger.debug(f"Current upper bound: {self.upper_bound}")
        
        # Initialize with max possible label
        label_v = Label(node=v, cost=(timedelta.max, float('inf')), pred=None)

        # For each incoming neighbor u
        incoming_neighbors = list(self.graph.predecessors(v))
        logger.debug(f"Node {v} has {len(incoming_neighbors)} incoming neighbors: {incoming_neighbors}")
        
        candidate_labels_found = 0
        
        for u in incoming_neighbors:
            # Get the range of unprocessed labels (could be empty initially)
            start_idx = self.lastProcessedLabel.get((u, v), 0)
            end_idx = len(L[u])
            
            logger.debug(f"Processing labels for incoming neighbor {u}: start_idx={start_idx}, end_idx={end_idx}")
            
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
                
                # Log detailed label computation
                logger.debug(f"  Candidate label computation:")
                logger.debug(f"    From node {u}: original cost {label_u.cost}")
                logger.debug(f"    Arc cost from {u} to {v}: cost={self.graph[u][v]['cost']}, penalty={self.graph[u][v]['penalty']}")
                logger.debug(f"    New label cost: {label_new.cost}")
                
                # Update last processed label index
                self.lastProcessedLabel[(u, v)] = k
                
                # Prune
                if label_new.cost[0] < self.upper_bound[0] and label_new.cost[1] < self.upper_bound[1]:
                    logger.debug(f"Candidate label within upper bound")
                    
                    # Check if the new cost is dominated by any efficient label at v
                    if not self.is_dominated(label_new.cost, L[v]):
                        logger.debug(f"Candidate label not dominated")
                        
                        is_lexographically_smaller = True
                        for label in L[v]:
                            logger.debug(f"{all(label_new.cost[i] < label.cost[i] for i in range(len(label_new.cost)))}")
                            if not all(label_new.cost[i] < label.cost[i] for i in range(len(label_new.cost))):
                                is_lexographically_smaller = False
                                logger.debug(f"  Not lexicographically smaller compared to existing label {label.cost}")
                                break
                        
                        if is_lexographically_smaller:
                            label_v = label_new
                            candidate_labels_found += 1
                            logger.debug(f"Selected new candidate label: {label_v.cost}")
                            break  # Stop after finding first suitable label
        
        # Logging for no candidate label found
        if label_v.cost[0] == timedelta.max and label_v.cost[1] == float('inf'):
            logger.debug(f"No candidate label found for node {v}")
            return None
        
        # Final logging summary
        logger.debug(f"Candidate label selection for node {v}:")
        logger.debug(f"  Candidate labels processed: {candidate_labels_found}")
        logger.debug(f"  Selected label cost: {label_v.cost}")
        
        return label_v

    def propagate(
        self, label: Label, w: Node, priority_queue, L, lower_bound=None, upper_bound=None
    ):
        """
        Propagate a label to node w.

        Parameters:
        - label: The label to propagate
        - w: The target node for propagation
        - priority_queue: The priority queue
        - L: Dictionary of efficient labels
        - lower_bound: Lower bound vector for node w (one-to-one version)
        - upper_bound: Upper bound label at target (one-to-one version)

        Returns:
        - Updated priority queue
        """
        v = label.node
        logger.debug(f"Propagating label from {v} to {w}")

        # Get the arc cost
        arc_cost = self.graph[v][w]["cost"]
        arc_penalty = self.graph[v][w]["penalty"]
        logger.debug(f"Arc cost from {v} to {w}: cost={arc_cost}, penalty={arc_penalty}")

        # Compute the new cost
        new_cost = (label.cost[0] + arc_cost, label.cost[1] + arc_penalty)
        logger.debug(f"New cost at {w}: {new_cost} (from {label.cost})")
        
        # Prune
        if new_cost[0] > self.upper_bound[0] or new_cost[1] > self.upper_bound[1]:
            logger.debug(f"New cost at {w} is greater than upper bound, skipping propagation {new_cost[0]} {new_cost[1]}")
            return priority_queue
        
        # Check if the new label is dominated
        if not self.is_dominated(new_cost, L[w]):
            logger.debug(f"New label at {w} is not dominated")
            new_label = Label(node=w, cost=new_cost, pred=label)
            acted = False
            for (index, label) in enumerate(self.H):
                if label[2].node == w:
                    self.H[index] = (new_cost, id(new_label), new_label)
                    acted = True
            logger.debug(f"Label update action: {acted}")
            if not acted:
                # Insert into priority queue
                self.insert_priority_queue(new_label)
                logger.debug(f"Propagation to {w} complete")
        
        return priority_queue

    def run(self):
        """Run the Multiobjective Dijkstra Algorithm."""
        logger.info("Starting MultiobjectiveDijkstra algorithm")
        
        # Initialize source label
        source_label = Label(
            node=self.source, cost=(timedelta(), 0), pred=None
        )
        logger.info(f"Initializing source label at {self.source} with cost {source_label.cost}")
        self.insert_priority_queue(source_label)

        iteration = 0
        # Main loop
        while self.H:
            iteration += 1
            logger.info(f"Iteration {iteration}, queue size: {len(self.H)}")
            
            # Extract lexicographically smallest label
            current_label = self.extract_lexmin()
            if current_label is None:
                logger.info("No more labels to process, terminating")
                break

            v = current_label.node
            logger.info(f"Processing node {v} with cost {current_label.cost}")

            # Add to efficient labels
            self.L[v].append(current_label)
            logger.debug(f"Added label to efficient labels at {v}, now has {len(self.L[v])} labels")

            # Find next candidate label for v
            next_label = self.nextCandidateLabel(
                v, self.L, upper_bound=self.upper_bound
            )

            if next_label:
                logger.debug(f"Found next candidate label for {v}: {next_label.cost}")
                self.insert_priority_queue(next_label)
            else:
                logger.debug(f"No next candidate label found for {v}")

            # Propagate to outgoing neighbors
            outgoing_neighbors = list(self.graph.successors(v))
            logger.debug(f"Node {v} has {outgoing_neighbors} outgoing neighbors")
            
            for w in outgoing_neighbors:
                self.H = self.propagate(
                    current_label, w, self.H, self.L, upper_bound=self.upper_bound
                )

            # Log summary of current state
            if iteration % 100 == 0 or len(self.H) == 0:
                logger.info(f"Iteration {iteration} summary:")
                logger.info(f"  Queue size: {len(self.H)}")
                logger.info(f"  Total efficient labels: {sum(len(labels) for labels in self.L.values())}")
                if self.target:
                    logger.info(f"  Current target labels: {len(self.L.get(self.target, []))}")
                    logger.info(f"  Current upper bound: {self.upper_bound}")

        logger.info(f"Algorithm terminated after {iteration} iterations")
        logger.info(f"Total efficient labels found: {sum(len(labels) for labels in self.L.values())}")
        
        # Return efficient labels
        return self.find_target_labels(self.L)

    def reconstruct_path(self, label):
        """Reconstruct the path from a label."""
        logger.info(f"Reconstructing path from label with cost {label.cost}")
        path = []
        current = label
        while current:
            path.append(current.node)
            current = current.pred
        path.reverse()
        
        # Log the full path
        path_str = " -> ".join(str(node) for node in path)
        logger.info(f"Path: {path_str}")
        logger.info(f"Path length: {len(path)} nodes, cost: {label.cost}")
        
        return path

    def find_target_labels(self, efficient_labels: Dict):
        """
        Find labels for nodes matching the target station and optional node type.
        
        Parameters:
        - efficient_labels: Dictionary of efficient labels for all nodes
        
        Returns:
        - List of labels matching target criteria, sorted by cost
        """
        logger.info("Finding target labels across efficient labels")
        target_labels = []
        for node, labels in efficient_labels.items():
            if self.is_target_node(node):
                target_labels.extend(labels)
        
        # Sort target labels lexicographically
        target_labels.sort(key=lambda label: label.cost)
        
        logger.info(f"Found {len(target_labels)} labels matching target criteria")
        for i, label in enumerate(target_labels):
            logger.info(f"  Target label {i}: node={label.node}, cost={label.cost}")
        
        return target_labels
    
    def is_target_node(self, node: Node) -> bool:
        """
        Check if a node matches the target criteria.
        
        Parameters:
        - node: Node to check
        
        Returns:
        - Boolean indicating if the node matches target station and optional node type
        """
        logger.debug(f"Checking if node {node} matches target {self.target}")
        result = node.station == self.target
        logger.debug(f"Node matches target: {result}")
        return result