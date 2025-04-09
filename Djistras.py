import copy
import sys
from Graph import Node
import heapq
from collections import defaultdict, namedtuple
import logging
import networkx as nx
from typing import Any, Set, List, Dict, Tuple
from datetime import datetime, timedelta
from Graph import Node, NODE_TYPE
import pickle


currentDT = datetime.now()
FileName = currentDT.strftime("%Y%m%d%H%M%S")

# Set up logging with only critical messages
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"/home/ali/dev/TransferPatternGeneration/multiobjective_dijkstra.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

class Path:
    def __init__(self, path: List[Node], cost: Tuple, short_path: List[Node], ):
        self.path = path
        self.cost = cost
        self.short_path = short_path
    
    def __repr__(self):
        path_str = " -> ".join(str(node) for node in self.short_path)
        return f"Cost: {self.cost}, Path: {path_str}"
    
    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return (self.path == other.path and self.short_path == other.short_path and 
                self.cost[0] == other.cost[0]and 
                self.cost[1] == other.cost[1])
    
    def __hash__(self):
        hashable_path = tuple(self.path)
        hashable_short_path = tuple(self.short_path)
        # For timedelta objects
        if isinstance(self.cost[0], timedelta):
            hashable_cost = (self.cost[0].total_seconds(), self.cost[1])
        else:
            hashable_cost = self.cost
        return hash((hashable_path, hashable_cost, hashable_short_path))
    
class Label:
    def __init__(self, node: Node, cost: Tuple, pred: 'Label' = None):
        self.node = node
        self.cost = cost
        self.pred = pred
    
    def __repr__(self):
        """Pretty print the label with its predecessor chain."""
        return self._format_label()
    
    def _format_label(self, indent=0):
        """Format this label with proper indentation."""
        cost_str = f"({self.cost[0]}, {self.cost[1]})" if isinstance(self.cost, tuple) else str(self.cost)
        result = f"Label(node={self.node}, cost={cost_str}"
        
        if self.pred is not None:
            pred_str = self.pred._format_label(indent + 2)
            result += f",\npred={pred_str}"
        else:
            result += ", pred=None"
        
        return result
    
    def get_path_string(self):
        """Generate a string showing the path from source to this label."""
        path = []
        current = self
        while current:
            node_info = f"{current.node}"
            cost_info = f"({current.cost[0]}, {current.cost[1]})" if isinstance(current.cost, tuple) else str(current.cost)
            path.append(f"{node_info} - Cost: {cost_info}")
            current = current.pred
        
        path.reverse()
        return "\n→ ".join(path)
    
    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.cost < other.cost
    
    def __eq__(self, other):
        if not isinstance(other, Label):
            return NotImplemented
        return (self.node == other.node and 
                self.cost == other.cost and 
            self.pred == other.pred)

    def __hash__(self):
        return hash((self.node, self.cost, id(self.pred)))
    
class MultiobjectiveDijkstra:
    def __init__(
        self,
        graph: nx.DiGraph,
        source: str,
        lower_bounds=None,
        upper_bound=None,
    ):
        """
        Initialize the Multiobjective Dijkstra Algorithm using NetworkX DiGraph.
        """
        logger.critical(f"Initializing MultiobjectiveDijkstra from {source}")
        logger.critical(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        logger.debug(f"Lower bounds: {lower_bounds}")
        logger.debug(f"Upper bound: {upper_bound}")
        
        self.graph = graph
        self.source = source
        self.lower_bounds = lower_bounds if lower_bounds else {}
        self.upper_bound = upper_bound

        # Get the dimension of the cost vectors from the first edge
        self.d = 2
        logger.debug(f"Cost vector dimension: {self.d}")

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
        
        logger.debug("Data structures initialized successfully")

    def dominates(self, cost1, cost2):
        """Check if cost1 dominates cost2 (is better in all dimensions)."""
        result = all(cost1[i] <= cost2[i] for i in range(len(cost1))) and any(
            cost1[i] < cost2[i] for i in range(len(cost1))
        )
        logger.debug(f"Dominance check: {cost1} {'dominates' if result else 'does not dominate'} {cost2}")
        return result

    def is_dominated(self, cost, label_set):
        """Check if cost is dominated by any cost in label_set."""
        if not label_set:
            logger.debug(f"Label set empty, cost {cost} is not dominated")
            return False
        
        result = any(self.dominates(label.cost, cost) for label in label_set)
        logger.debug(f"Cost {cost} is {'dominated' if result else 'not dominated'} by label set of size {len(label_set)}")    
        return result
    
    def is_path_dominated(self, cost, path_set):
        """Check if cost is dominated by any cost in label_set."""
        if not path_set:
            logger.debug(f"Label set empty, cost {cost} is not dominated")
            return False

        result = any(self.dominates(path.cost, cost) for path in path_set)
        logger.debug(f"Cost {cost} is {'dominated' if result else 'not dominated'} by label set of size {len(path_set)}")    
        return result
    
    def insert_priority_queue(self, label):
        """Insert a label into the priority queue (ordered lexicographically)."""
        logger.debug(f"Inserting label for node {label.node} with cost {label.cost} into priority queue")
        heapq.heappush(self.H, (label.cost, id(label), label))
        logger.debug(f"Priority queue size after insertion: {len(self.H)}")

    def extract_lexmin(self):
        """Extract the lexicographically smallest label from the priority queue."""
        if not self.H:
            logger.debug("Priority queue is empty, returning None")
            return None
        _, _, label = heapq.heappop(self.H)
        logger.debug(f"Extracted label for node {label.node} with cost {label.cost} from priority queue")
        logger.debug(f"Priority queue size after extraction: {len(self.H)}")
        return label

    def nextCandidateLabel(
        self, v, L, lower_bound=None, upper_bound=None
    ):
        """
        Find the next candidate label for node v by processing unprocessed labels from predecessors.
        """
        logger.debug(f"Finding next candidate label for node {v}")
        
        # Initialize with max possible label
        label_v = Label(node=v, cost=(timedelta.max, float('inf')), pred=None)

        # For each incoming neighbor u
        incoming_neighbors = list(self.graph.predecessors(v))
        logger.debug(f"Node {v} has {len(incoming_neighbors)} incoming neighbors")
        
        for u in incoming_neighbors:
            # Get the range of unprocessed labels
            start_idx = self.lastProcessedLabel.get((u, v), 0)
            end_idx = len(L[u])
            
            logger.debug(f"Processing labels from neighbor {u} (index range: {start_idx}-{end_idx})")
            
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
                
                logger.debug(f"Generated new label with cost {label_new.cost} from label at node {u} with cost {label_u.cost}")
                
                # Update last processed label index
                self.lastProcessedLabel[(u, v)] = k
                
                # Prune
                if label_new.cost[0] < self.upper_bound[0] and label_new.cost[1] < self.upper_bound[1]:
                    logger.debug(f"Label satisfies upper bound constraints: {label_new.cost} < {self.upper_bound}")
                    
                    # Check if the new cost is dominated by any efficient label at v
                    if not self.is_dominated(label_new.cost, L[v]):
                        logger.debug(f"Label is not dominated by existing labels at node {v}")
                        
                        is_lexographically_smaller = True
                        for label in L[v]:
                            if not all(label_new.cost[i] < label.cost[i] for i in range(len(label_new.cost))):
                                is_lexographically_smaller = False
                                break
                        
                        if is_lexographically_smaller:
                            logger.debug(f"Found lexicographically smaller label with cost {label_new.cost}")
                            label_v = label_new
                            break  # Stop after finding first suitable label
                else:
                    logger.debug(f"Label pruned: {label_new.cost} exceeds upper bound {self.upper_bound}")
        
        # Return None if no suitable label found
        if label_v.cost[0] == timedelta.max and label_v.cost[1] == float('inf'):
            logger.debug(f"No suitable candidate label found for node {v}")
            return None
        
        logger.debug(f"Selected candidate label for node {v} with cost {label_v.cost}")
        return label_v

    def propagate(
        self, label: Label, w: Node, priority_queue, L, lower_bound=None, upper_bound=None
    ):
        """
        Propagate a label to node w.
        """
        v = label.node
        logger.debug(f"Propagating label from node {v} to node {w}")

        # Get the arc cost
        arc_cost = self.graph[v][w]["cost"]
        arc_penalty = self.graph[v][w]["penalty"]
        logger.debug(f"Arc cost: {arc_cost}, penalty: {arc_penalty}")

        # Compute the new cost
        new_cost = (label.cost[0] + arc_cost, label.cost[1] + arc_penalty)
        logger.debug(f"New cost after propagation: {new_cost}")            
        new_label = Label(node=w, cost=new_cost, pred=label)
        
        # Prune based on cost 
        if new_cost[0] > self.upper_bound[0] or new_cost[1] > self.upper_bound[1]:
            logger.debug(f"Label pruned: {new_cost} exceeds upper bound {self.upper_bound}")
            return priority_queue
        
        # Check if the new label is dominated
        if not self.is_dominated(new_cost, L[w]):
            logger.debug(f"New label is not dominated by existing labels at node {w}")
            acted = False
            for (index, label) in enumerate(self.H):
                if label[2].node == w:
                    logger.debug(f"Updating existing label for node {w} in priority queue")
                    self.H[index] = (new_cost, id(new_label), new_label)
                    acted = True
                    break
            
            if not acted:
                # Insert into priority queue
                logger.debug(f"Inserting new label for node {w} into priority queue")
                self.insert_priority_queue(new_label)
            
        else:
            logger.debug(f"New label is dominated by existing labels at node {w}")
            
        return priority_queue

    def run(self):
        logger.debug("Starting run method to find matching source node")
        for node in self.graph.nodes():
            if node.station == self.source and node.node_type == NODE_TYPE.TRANSFER:
                # if str(node) == "Pymble Station Platform 2|TRANSFER|None@22:47:00":
                self.run_for_source(source_node=node)
    
    def run_for_source(self, source_node):
        """Run the Multiobjective Dijkstra Algorithm."""
        logger.critical("Starting MultiobjectiveDijkstra algorithm")
        logger.debug(f"Source node: {source_node}")
        
        # Initialize source label
        source_label = Label(
            node=source_node, cost=(timedelta(), 0), pred=None
        )
        logger.debug(f"Created initial source label with cost {source_label.cost}")
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
            logger.debug(f"Processing node {v} with cost {current_label.cost} at iteration {iteration}")

            # Add to efficient labels
            self.L[v].append(current_label)
            logger.debug(f"Added label to efficient labels for node {v}. Total labels: {len(self.L[v])}")

            # Find next candidate label for v
            logger.debug(f"Finding next candidate label for node {v}")
            next_label = self.nextCandidateLabel(
                v, self.L, upper_bound=self.upper_bound
            )

            if next_label:
                logger.debug(f"Found next candidate label for node {v} with cost {next_label.cost}")
                self.insert_priority_queue(next_label)
            else:
                logger.debug(f"No next candidate label found for node {v}")

            # Propagate to outgoing neighbors
            outgoing_neighbors = list(self.graph.successors(v))
            logger.debug(f"Node {v} has {len(outgoing_neighbors)} outgoing neighbors")
            
            for w in outgoing_neighbors:
                logger.debug(f"Propagating to neighbor {w}")
                self.H = self.propagate(
                    current_label, w, self.H, self.L, upper_bound=self.upper_bound
                )

            # Periodic logging
            if iteration % 1000 == 0:
                logger.critical(f"Iteration {iteration}, Queue size: {len(self.H)}")

        logger.critical(f"Algorithm terminated after {iteration} iterations")
        logger.critical(f"Total efficient labels found: {sum(len(labels) for labels in self.L.values())}")
        logger.debug("Run complete")
        
        return 

    def find_target_labels(self, efficient_labels: Dict, target: str):
        """
        Find labels for nodes matching the target station and optional node type.
        """
        logger.critical(f"Target station: {target}")
        target_labels = []
        for node, labels in efficient_labels.items():
            if self.is_target_node(node, target):
                target_labels.extend(labels)
        
        # Sort target labels lexicographically
        target_labels.sort(key=lambda label: label.cost)
        logger.critical(f"Target station: {target}. Found {len(target_labels)} labels matching target criteria")
        
        return target_labels
    
    def is_target_node(self, node: Node, target: str) -> bool:
        """
        Check if a node matches the target criteria.
        """
        result = node.station == target and node.node_type == NODE_TYPE.ARRIVAL
        return result

    def arrival_chain_algorithm(self, optimal_labels: List[Label]):
        # Group labels by arrival time
        arrival_times = defaultdict(list)
        for label in optimal_labels:
            arrival_time = label.node.time
            arrival_times[arrival_time].append(label)
        
        # Sort arrival times
        sorted_times = sorted(arrival_times.keys())
        
        # Apply the arrival chain algorithm
        final_optimal_path =  {}
        prev_labels = []
        for current_time in sorted_times:
            current_labels = arrival_times[current_time]

            for label in current_labels:
                logger.critical(self.reconstruct_complete_path(label))

            # Step 1: Create extended previous labels (waiting passengers)
            extended_prev_labels = []
            if prev_labels:
                prev_time = sorted_times[sorted_times.index(current_time)-1]
                wait_time = current_time - prev_time
                
                for prev_label in prev_labels:
                    # Create new label with increased duration
                    new_duration = prev_label.cost[0] + wait_time
                    new_cost = (new_duration, prev_label.cost[1])
                    extended_label = Label(prev_label.node, new_cost, prev_label.pred)
                    extended_prev_labels.append(extended_label)
            
            # Step 2: Combine and filter (with preference to extended_prev_labels)
            time_optimal_labels = []
            
            # First pass: add extended_prev_labels if not dominated
            for label in extended_prev_labels:
                if not self.is_dominated(label.cost, time_optimal_labels):
                    time_optimal_labels = [
                        existing_label for existing_label in time_optimal_labels
                        if not self.dominates(label.cost, existing_label.cost)
                    ]
                    time_optimal_labels.append(label)
            
            # Second pass: add current_labels if they dominate existing or are new
            for label in current_labels:
                if not self.is_dominated(label.cost, time_optimal_labels):
                    # Remove dominated existing labels
                    time_optimal_labels = [
                        existing_label for existing_label in time_optimal_labels
                        if not self.dominates(label.cost, existing_label.cost)
                    ]
                    time_optimal_labels.append(label)
            
           
            for label in time_optimal_labels:
                path = self.reconstruct_complete_path(label)
                logger.critical(path)

            # Step 3: Add to the best item to the optimal list. Best is either the latest leaving, or the better cost in that order.  
            latest_departure = timedelta.min
            best_cost = (timedelta.max, float("inf"))
            best = []
            for label in time_optimal_labels:
                path: Path = self.reconstruct_complete_path(label)
                for (i, node) in enumerate(path.path):
                    if node.station == self.source and node.node_type == NODE_TYPE.DEPARTURE:
                        if node.time > latest_departure:
                            latest_departure = node.time
                            best_cost = label.cost
                            best = [path]
                        elif node.time == latest_departure and label.cost[0] < best_cost[0]:
                            best_cost = label.cost
                            best = [path]
                        elif node.time == latest_departure and self.dominates(label.cost, best_cost):
                            best_cost = label.cost
                            best = [path]
                        elif node.time == latest_departure:
                            best.append(path)
                        break 
                
            final_optimal_path[current_time] = best
        
            # Step 4: Update for next iteration
            prev_labels = current_labels
        
        logger.critical("\n\n\n\n\n\n\n\n\n\n\n\n")

        final_optimal_paths = []
        for (i, current_time) in enumerate(sorted_times):
            paths = final_optimal_path[current_time]
            
            copy_path = paths

            for j in range(len(paths) - 1, 0):
                if i <= 0: continue 

                for k in range(0, i - 1):
                    previous_time_to_compare_with = sorted_times[k]
                    previous_paths= final_optimal_path[previous_time_to_compare_with]
                    for previous_path in previous_paths: 
                        previous_start_time = previous_path.path[0].time
                        
                        # is there an earlier arrival time that leaves after this "optimal" label. If so, its not optimal and remove this.
                        if path.path[0].time > previous_start_time:
                            logger.critical(path)
                            logger.critical(copy_path)
                            logger.critical(paths)
                            del copy_path[j]

                        if len(copy_path) == 0: break
                    if len(copy_path) == 0: break
                if len(copy_path) == 0: break

            final_optimal_paths.extend(copy_path)

        for path in final_optimal_paths:
            logger.critical(path)

        logger.critical("POST CUNT\n\n\n\n\n\n\n\n\n\n\n\n")

        # Group labels by arrival time and then find dominantlabels
        departure_times = defaultdict(list)
        for path in final_optimal_paths:
            arrival_time = label.node.time
            for node in path.path:
                if node.station == self.source and node.node_type == NODE_TYPE.DEPARTURE:
                    if node.time not in departure_times:
                        departure_times[node.time] = []
                    
                    if not self.is_dominated(path.cost, departure_times[node.time]):
                        # Remove dominated existing labels
                        departure_times[node.time] = [
                            existing_path for existing_path in departure_times[node.time]
                            if not self.dominates(path.cost, existing_path.cost)
                        ]
                        departure_times[node.time].append(path)

        transfer_pattern = set()
        for time in departure_times:
            for path in departure_times[time]:
                logger.critical(path)
                node_list = [node.station_string_name for node in path.short_path]
                transfer_pattern.add(tuple(node_list))

        return final_optimal_path, transfer_pattern
    
    def reconstruct_complete_path(self, label: Label) -> Path:
        """Reconstruct the path from a label."""
        path = [label.node]
        short_path = [label.node]
        current = label
        while current:
            if current.node.node_type == NODE_TYPE.DEPARTURE and current.pred != None and current.pred.node.node_type == NODE_TYPE.TRANSFER:
                short_path.append(current.node)
            elif current.pred == None:
                short_path.append(current.node)

            path.append(current.node)
            current = current.pred
        path.reverse()
        short_path.reverse()
        
        return Path(path, label.cost,short_path)