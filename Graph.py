import networkx as nx
from enum import Enum
import logging
import sys
from datetime import timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import defaultdict
import bisect
import pickle

class TRAVEL_TYPE(Enum):
    NORMAL = 1
    STAYINGONTRAIN = 2
    TRANSFER = 3
    POSTTRANSFER = 4
    
    def __str__(self):
        return self.name

class NODE_TYPE(Enum):
    DEPARTURE = 1
    ARRIVAL = 2
    TRANSFER = 3
    
    def __str__(self):
        return self.name

@dataclass(frozen=True)
class Node:
    station: str
    platform: str
    time: timedelta
    node_type: NODE_TYPE

    def __str__(self):
        return f"{self.station}{self.node_type}@{self.time}"

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.time < other.time

class NestedDefaultDict:
    """A nested defaultdict that can be pickled."""
    def __init__(self):
        self._dict = {}
        
    def __getitem__(self, key):
        if key not in self._dict:
            self._dict[key] = defaultdict(list)
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value
        
    def items(self):
        return self._dict.items()
        
    def clear(self):
        self._dict.clear()

class TimeExpandedGraph(object):
    def __init__(self):
        """Initialize an optimized time-expanded graph using NetworkX."""
        self.graph = nx.DiGraph()
        self._station_nodes: Dict[str, Dict[NODE_TYPE, List[Node]]] = NestedDefaultDict()
        self._transfer_times: Dict[str, timedelta] = {}

        # Logger setup
        logging.basicConfig(filename="graph_processing.log",
                        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
                        filemode='w',
                        level=logging.INFO)
        self.logger = logging.getLogger("graph")
        self.logger.propagate = False

    def add_node(self, node: Node) -> None:
        """Add node to both NetworkX graph and indexed structure."""
        if not self.graph.has_node(node):
            self.graph.add_node(node)
            bisect.insort(self._station_nodes[node.station][node.node_type], node)
            self.logger.debug(f"Added node: {node}")

    def add_edge(self, source: Node, destination: Node, travel_type: TRAVEL_TYPE):
        """Add an edge between nodes with optimized indexing."""
        self.add_node(source)
        self.add_node(destination)
        penalty = 1 if travel_type == TRAVEL_TYPE.TRANSFER else 0
        self.graph.add_edge(
            source,
            destination,
            edge_type=travel_type,
            cost=destination.time - source.time,
            penalty=penalty
        )
        self.logger.debug(f"Added edge: {source} -> {destination} ({travel_type})")

    def get_next_transfer_node(self, station: str, after_time: timedelta) -> Optional[Node]:
        """Find next transfer node after a given time using binary search."""
        transfer_nodes = self._station_nodes[station][NODE_TYPE.TRANSFER]
        if not transfer_nodes:
            return None
        dummy = Node(station, after_time, NODE_TYPE.TRANSFER)
        idx = bisect.bisect_right(transfer_nodes, dummy)
        return transfer_nodes[idx] if idx < len(transfer_nodes) else None

    def get_station_nodes(self, station: str, node_type: NODE_TYPE,
                          start_time: timedelta, end_time: timedelta) -> List[Node]:
        """Get all nodes of a specific type within a time range using binary search."""
        nodes = self._station_nodes[station][node_type]
        start_idx = bisect.bisect_left(nodes, Node(station, start_time, node_type))
        end_idx = bisect.bisect_right(nodes, Node(station, end_time, node_type))
        return nodes[start_idx:end_idx]

    def set_transfer_time(self, station: str, transfer_time: timedelta):
        """Set minimum transfer time for a station."""
        self._transfer_times[station] = transfer_time
        self.logger.debug(f"Set transfer time for {station}: {transfer_time}")

    def get_transfer_time(self, station: str) -> timedelta:
        """Get minimum transfer time for a station."""
        return self._transfer_times.get(station, timedelta(minutes=2))

    def save_to_pickle(self, file_path: str):
        """Save the graph and metadata to a pickle file."""
        try:
            with open(file_path, 'wb') as f:
                # Save graph and index structures
                data = {
                    "graph": self.graph,
                    "_station_nodes": self._station_nodes,
                    "_transfer_times": self._transfer_times
                }
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Graph saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save graph: {e}")

    def load_from_pickle(self, file_path: str):
        """Load the graph and metadata from a pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data["graph"]
            self._station_nodes = data["_station_nodes"]
            self._transfer_times = data["_transfer_times"]
        self.logger.info(f"Graph loaded from {file_path}")