from Parser import GTFSParser
import pickle
import logging
import sys
from datetime import datetime, timedelta
from utilities import *
from typing import Dict, Optional
from Graph import *
from Djistras import *
from Line import *

class Main:
    def __init__(self, gtfs_path: str):
        """Initialize Main class with GTFS data path."""
        self.graph = TimeExpandedGraph()
        self.lines = Lines()
        self.parser = GTFSParser(gtfs_path)
        self.parser.load_gtfs_data()
        
        # Set up logging
        logging.basicConfig(filename="main.log",
                        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
                        filemode='w',
                        level=logging.INFO)
        self.logger = logging.getLogger("main")
        self.logger.propagate = False
        
        self._load_transfer_times()
        self.logger.info("Initialized Main class and loaded GTFS data.")

    def _load_transfer_times(self):
        """Load transfer times from GTFS transfers.txt if available."""
        try:
            for transfer in self.parser.transfers:
                self.graph.set_transfer_time(
                    transfer['from_stop_id'],
                    timedelta(seconds=int(transfer['min_transfer_time']))
                )
        except AttributeError:
            self.logger.warning("No transfer times found in GTFS data. Using defaults.")

    def add_vehicle_connection(self, 
                             station_a: str, 
                             child_stop_a: str, 
                             station_a_departure_time: str, 
                             station_b: str, 
                             child_stop_b: str, 
                             arrival_time: str,
                             station_b_departure_time: Optional[str] = None) -> None:
        """Add nodes and edges for a vehicle connection between stations."""
        try:
            # Parse times
            dep_time = parse_time_with_overflow(station_a_departure_time)
            arr_time = parse_time_with_overflow(arrival_time)
            
            # Create nodes
            dep_node = Node(station_a, child_stop_a, dep_time, NODE_TYPE.DEPARTURE)
            arr_node = Node(station_b, child_stop_b, arr_time, NODE_TYPE.ARRIVAL)
            
            # Add riding edge
            self.graph.add_edge(dep_node, arr_node, TRAVEL_TYPE.NORMAL)
            
            # If vehicle continues, add staying edge
            if station_b_departure_time:
                next_dep_time = parse_time_with_overflow(station_b_departure_time)
                next_dep_node = Node(station_b, child_stop_b, next_dep_time, NODE_TYPE.DEPARTURE)
                self.graph.add_edge(arr_node, next_dep_node, TRAVEL_TYPE.STAYINGONTRAIN)
                
                # Create transfer node at arrival time
                transfer_node = Node(station_b, child_stop_b, arr_time, NODE_TYPE.TRANSFER)
                self.graph.add_edge(transfer_node, next_dep_node, TRAVEL_TYPE.TRANSFER)
                
            self.logger.debug(f"Added vehicle connection from {station_a} to {station_b}")
            
        except Exception as e:
            self.logger.error(f"Error adding vehicle connection: {e}")
            raise

    def process_transfers(self):
        """Process all transfer connections in the graph."""
        nodes = sorted(list(self.graph.graph.nodes()), key=lambda x: x.time)
        total_nodes = len(nodes)
        processed = 0
        
        for i in range(total_nodes): 
            processed += 1
            if processed % 1000 == 0:  # Log progress every 1000 nodes
                self.logger.info(f"Processing transfers: {processed}/{total_nodes} nodes ({processed/total_nodes*100:.2f}%)")
            
            if nodes[i].node_type == NODE_TYPE.TRANSFER:
                for j in range(i + 1, total_nodes): 
                    if nodes[j].node_type == NODE_TYPE.DEPARTURE:
                        self.graph.add_edge(nodes[i], nodes[j], TRAVEL_TYPE.POSTTRANSFER)
                    if nodes[j].node_type == NODE_TYPE.TRANSFER:
                        self.graph.add_edge(nodes[i], nodes[j], TRAVEL_TYPE.POSTTRANSFER)
                        break
        
    def compact_graph(self) -> nx.DiGraph:
        """
        Creates a compact representation of a graph by removing departure nodes and 
        linking their predecessors directly to their successors.

        Parameters:
            original_graph: A directed graph (DiGraph) where nodes are instances of Node.

        Returns:
            A new compacted directed graph.
        """
        # Create a new graph to store the compact representation
        compacted_graph = nx.DiGraph()

        # Copy all nodes and edges except for departure nodes
        for node in self.graph.graph.nodes:
            if node.node_type != NODE_TYPE.DEPARTURE:
                compacted_graph.add_node(node)
        
        # Traverse each edge and rewire around departure nodes
        for node in self.graph.graph.nodes:
            if node.node_type == NODE_TYPE.DEPARTURE:
                # Get all predecessors and successors of the departure node
                predecessors = list(self.graph.graph.predecessors(node))
                successors = list(self.graph.graph.successors(node))

                for pred in predecessors:
                    for succ in successors:
                        # Get the cost and penalty for the new direct edge
                        pred_to_dep = self.graph.graph.get_edge_data(pred, node, default={})
                        dep_to_succ = self.graph.graph.get_edge_data(node, succ, default={})

                        # Calculate cumulative cost and penalty
                        new_cost = pred_to_dep.get("cost", timedelta(0)) + dep_to_succ.get("cost", timedelta(0))
                        new_penalty = pred_to_dep.get("penalty", 0) + dep_to_succ.get("penalty", 0)

                        # Add the direct edge
                        compacted_graph.add_edge(
                            pred,
                            succ,
                            cost=new_cost,
                            penalty=new_penalty,
                        )
            else:
                # Copy edges for non-departure nodes
                for neighbor in self.graph.graph.successors(node):
                    edge_data = self.graph.graph.get_edge_data(node, neighbor)
                    compacted_graph.add_edge(node, neighbor, **edge_data)

        return compacted_graph    

    def _connect_transfer_chain(self, transfer_node: Node):
        """Connect transfer node to the next transfer node in time sequence."""
        next_transfer = self.graph.get_next_transfer_node(
            transfer_node.station,
            transfer_node.time
        )
        if next_transfer:
            self.graph.add_edge(transfer_node, next_transfer, TRAVEL_TYPE.TRANSFER)

def build_graph(gtfs_path: str, output_pickle: str = "graph.pickle", output_line_pickle: str = "lines.pickle") -> TimeExpandedGraph:
    """Build and save the time-expanded graph from GTFS data."""
    try:
        # Try to load existing graph
        main = Main(gtfs_path)

        # main.graph.load_from_pickle(output_pickle)
        # print(f"Loaded existing graph from {output_pickle}")
        
        main.lines = load_lines(output_line_pickle)

        print(f"Loaded existing lines from {output_line_pickle}")
        
        if not main.lines:
            raise EOFError


        for line in main.lines.lines:
            print(line)

        return main.graph
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        print(f"Building new graph from GTFS data at {gtfs_path}")
        
        # Initialize main processor
        main = Main(gtfs_path)

        # Process all trips
        total_trips = len(main.parser.stop_times)
        for index, (trip_id, stop_times_array) in enumerate(main.parser.stop_times.items()):
            print(f"Processing trip {index}/{total_trips} ({index/total_trips*100:.2f}%)")
            length = len(stop_times_array)
            if length <= 1: continue 
            
            # Process each stop in the trip
            # for i in range(length - 2):
            #     main.add_vehicle_connection(
            #         main.parser.stops[stop_times[i]["stop_id"]]["parent_station"],
            #         stop_times[i]["stop_id"],
            #         stop_times[i]["departure_time"],
            #         main.parser.stops[stop_times[i + 1]["stop_id"]]["parent_station"],
            #         stop_times[i + 1]["stop_id"],
            #         stop_times[i + 1]["arrival_time"],
            #         stop_times[i + 1].get("departure_time")  # Some stops might not have departure times
            #     )

            # service_id = main.parser.trips[trip_id]["service_id"]
            # days_running = main.parser.calendar.loc[main.parser.calendar['service_id'] == service_id].to_dict("records")[0]

            main.lines.add_trip(stop_times_array, {})
        
        # Process transfer connections
        # print("Processing transfer connections...")
        # main.process_transfers()
        
        # Save the graph
        # print(f"Saving graph to {output_pickle}")
        # main.graph.save_to_pickle(output_pickle)
        
        print(f"Saving lines to {output_line_pickle}")
        save_lines(main.lines, output_line_pickle)

        return main.graph

def validate_transit_network(G: nx.DiGraph) -> bool:
    """
    Validate network connectivity rules using NetworkX.
    
    Rules:
    - TRANSFER node can connect to TRANSFER or DEPARTURE nodes
    - ARRIVAL node can connect to DEPARTURE or TRANSFER nodes
    - DEPARTURE node can connect to ARRIVAL nodes
    """
    for nodeView in G.nodes(data=True):
        node, _ = nodeView
        node_type = node.node_type
        
        for _, successor, _ in G.out_edges(node, data=True):
            succ_type = successor.node_type
            
            valid_connections = {
                NODE_TYPE.TRANSFER: [NODE_TYPE.TRANSFER, NODE_TYPE.DEPARTURE],
                NODE_TYPE.ARRIVAL: [NODE_TYPE.DEPARTURE, NODE_TYPE.TRANSFER],
                NODE_TYPE.DEPARTURE: [NODE_TYPE.ARRIVAL]
            }
            
            if succ_type not in valid_connections.get(node_type, []):
                return False
    
    return True

if __name__ == "__main__":
    # Configuration
    GTFS_PATH = '/home/ali/dev/sydneytrains_GTFS_PROD_20241208140100'
    PICKLE_PATH = "/home/ali/dev/Train/graph.pickle"
    LINES_PICKE_PATH = "/home/ali/dev/Train/lines.pickle"

    try:
        print("Start building graph")
        graph = build_graph(GTFS_PATH, PICKLE_PATH, LINES_PICKE_PATH)
        print("Graph processing completed successfully")
    
        # Print some statistics
        print(f"Total nodes: {len(graph.graph.nodes())}")
        print(f"Total edges: {len(graph.graph.edges())}")

        print("validating", validate_transit_network(graph.graph))

        # print(mc_dijkstra(graph.graph, Node("207310", timedelta(hours=4, minutes=16, seconds=36), node_type=NODE_TYPE.DEPARTURE), "200060"))
        
    except Exception as e:
        print(f"Error building graph: {e}")
        raise