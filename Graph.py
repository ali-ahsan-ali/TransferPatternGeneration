from typing import Optional
import networkx as nx
from enum import Enum
import logging
from datetime import timedelta
from dataclasses import dataclass
import pickle


class TRAVEL_TYPE(Enum):
    NORMAL = 1
    TRANSFER = 2
    WAITINGCHAIN = 3
    STAYINGONTRAIN = 4

    def __str__(self):
        return self.name


class NODE_TYPE(Enum):
    ARRIVAL = 1
    TRANSFER = 2
    DEPARTURE=3

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Node:
    station: str
    station_string_name: str
    platform: str
    platform_string_name: str
    time: timedelta
    node_type: NODE_TYPE
    drop_off_type: Optional[int]

    def __str__(self):
        return f"{self.platform_string_name}|{self.node_type}@{self.time}"

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.time < other.time
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return (self.station == other.station and
                self.station_string_name == other.station_string_name and
                self.platform == other.platform and
                self.platform_string_name == other.platform_string_name and
                self.time == other.time and
                self.node_type == other.node_type and 
                self.drop_off_type == other.drop_off_type)


class TimeExpandedGraph(object):
    def __init__(self):
        """Initialize an optimized time-expanded graph using NetworkX."""
        self.graph = nx.DiGraph()

        # Logger setup
        logging.basicConfig(
            filename="graph_processing.log",
            format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
            filemode="w",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("graph")
        self.logger.propagate = False

    def add_node(self, node: Node) -> None:
        """Add node to both NetworkX graph and indexed structure."""
        if not self.graph.has_node(node):
            self.graph.add_node(node)
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
            penalty=penalty,
        )
        self.logger.debug(f"Added edge: {source} -> {destination} ({travel_type})")

    def save_to_pickle(self, file_path: str):
        """Save the graph and metadata to a pickle file."""
        try:
            with open(file_path, "wb") as f:
                # Save graph and index structures
                data = {
                    "graph": self.graph,
                }
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Graph saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save graph: {e}")

    def load_from_pickle(self, file_path: str):
        """Load the graph and metadata from a pickle file."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.graph = data["graph"]
        self.logger.info(f"Graph loaded from {file_path}")
