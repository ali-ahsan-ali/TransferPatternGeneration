from Parser import GTFSParser
import pickle
import logging
from datetime import timedelta
from typing import Optional
from Graph import Node, TimeExpandedGraph, NODE_TYPE, TRAVEL_TYPE
from Djistras import Label, MultiobjectiveDijkstra
from utilities import parse_time_with_overflow
import networkx as nx
from pandas import *



# Set up logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("main.log", mode="w")
    ]
)
logger = logging.getLogger("main")

logger.info("Initialized Main class and loaded GTFS data.")
        
class Main:
    def __init__(self, gtfs_path: str):
        """Initialize Main class with GTFS data path."""
        self.graph = TimeExpandedGraph()
        self.parser = GTFSParser(gtfs_path)
        self.parser.load_gtfs_data()

    def add_vehicle_connection(
        self,
        station_a: str,
        child_stop_a: str,
        station_a_departure_time: str,
        station_b: str,
        child_stop_b: str,
        arrival_time: str,
        station_b_departure_time: Optional[str] = None,
    ) -> None:
        """Add nodes and edges for a vehicle connection between stations."""
        try:
            # Parse times
            dep_time = parse_time_with_overflow(station_a_departure_time)
            arr_time = parse_time_with_overflow(arrival_time)

            # Create nodes
            dep_node = Node(station_a, self.parser.stops[station_a]["stop_name"], child_stop_a, self.parser.stops[child_stop_a]["stop_name"], dep_time, NODE_TYPE.DEPARTURE)
            arr_node = Node(station_b, self.parser.stops[station_b]["stop_name"], child_stop_b, self.parser.stops[child_stop_b]["stop_name"], arr_time, NODE_TYPE.ARRIVAL)

            # Add riding edge
            self.graph.add_edge(dep_node, arr_node, TRAVEL_TYPE.NORMAL)

            # If vehicle continues, add staying edge
            if station_b_departure_time:
                next_dep_time = parse_time_with_overflow(station_b_departure_time)
                next_dep_node = Node(
                    station_b, self.parser.stops[station_b]["stop_name"], child_stop_b, self.parser.stops[child_stop_b]["stop_name"], next_dep_time, NODE_TYPE.DEPARTURE
                )
                self.graph.add_edge(arr_node, next_dep_node, TRAVEL_TYPE.STAYINGONTRAIN)

                # Create transfer node at arrival time
                transfer_node = Node(
                    station_b, self.parser.stops[station_b]["stop_name"], child_stop_b,  self.parser.stops[child_stop_b]["stop_name"], next_dep_time, NODE_TYPE.TRANSFER
                )
                self.graph.add_edge(transfer_node, next_dep_node, TRAVEL_TYPE.WAITINGCHAIN)
            logger.debug(
                f"Added vehicle connection from {station_a} to {station_b}"
            )

        except Exception as e:
            logger.error(f"Error adding vehicle connection: {e}")
            raise

    def process_transfers(self):
        """Process all transfer connections in the graph."""
        nodes = sorted(list(self.graph.graph.nodes()), key=lambda x: x.time)
        total_nodes = len(nodes)
        processed = 0
        
        last_station_transfer = {}

        for i in range(total_nodes):
            processed += 1
            if processed % 10000 == 0:  # Log progress every 1000 nodes
                logger.critical(
                    f"Processing transfers: {processed}/{total_nodes} nodes ({processed / total_nodes * 100:.2f}%)"
                )
                
            if nodes[i].station not in last_station_transfer and nodes[i].node_type == NODE_TYPE.TRANSFER:
                last_station_transfer[nodes[i].station] = i
            elif nodes[i].node_type == NODE_TYPE.TRANSFER:
                self.graph.add_edge(
                    nodes[last_station_transfer[nodes[i].station]], nodes[i], TRAVEL_TYPE.WAITINGCHAIN
                )
                last_station_transfer[nodes[i].station] = i
            
            elif nodes[i].node_type == NODE_TYPE.ARRIVAL:
                j = i + 1
                while j < total_nodes:
                    if nodes[j].station == nodes[i].station and nodes[j].node_type == NODE_TYPE.TRANSFER and nodes[j].time - nodes[i].time > timedelta(minutes=2):
                        self.graph.add_edge(
                            nodes[i], nodes[j], TRAVEL_TYPE.TRANSFER
                        )
                        break
                    j += 1


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
                NODE_TYPE.DEPARTURE: [NODE_TYPE.ARRIVAL],
            }

            if succ_type not in valid_connections.get(node_type, []):
                return False

    return True

def build_graph(
    gtfs_path: str, output_pickle: str = "graph.pickle"
) -> TimeExpandedGraph:
    """Build and save the time-expanded graph from GTFS data."""
    try:
        # Try to load existing graph
        main = Main(gtfs_path)

        main.graph.load_from_pickle(output_pickle)
        print(f"Loaded existing graph from {output_pickle}")

        return main.graph
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        print(f"Building new graph from GTFS data at {gtfs_path}")

        # Initialize main processor
        main = Main(gtfs_path)

        # Process all trips
        total_trips = len(main.parser.stop_times)
        for index, (trip_id, stop_times) in enumerate(main.parser.stop_times.items()):
            if index % 1000 == 0:  # Log progress every 1000 nodes
                logger.debug(
                    f"Processing trip {index}/{total_trips} ({index / total_trips * 100:.2f}%)"
                )
            length = len(stop_times)
            if length <= 1:
                continue
            
            service_id = main.parser.trips[stop_times[0]["trip_id"]]["service_id"]
            days_running = main.parser.calendar.loc[
                main.parser.calendar["service_id"] == service_id
            ].to_dict("records")[0]
            if days_running["monday"] != 1 or days_running["start_date"] < 20250310:
                continue

            # Process each stop in the trip
            logger.warning(f"{stop_times[0]["trip_id"]}, {trip_id}")
            for i in range(length - 1):
                j = i+1
                if stop_times[i]["pickup_type"] == 0 or isna(stop_times[i]["pickup_type"]):
                    while j < length - 2 and stop_times[j]["drop_off_type"] != 0 and isna(stop_times[j]["drop_off_type"]):
                        j += 1
                    
                    if j == length - 1 : continue 
                    
                    logger.debug(f"{main.parser.stops[stop_times[i]["stop_id"]]["stop_name"]}, {main.parser.stops[stop_times[j]["stop_id"]]["stop_name"]}")
                    if stop_times[i]["stop_id"] == 2073161 and stop_times[j]["stop_id"] == 217391:
                        exit()
                    main.add_vehicle_connection(
                        main.parser.stops[stop_times[i]["stop_id"]]["parent_station"],
                        stop_times[i]["stop_id"],
                        stop_times[i]["departure_time"],
                        main.parser.stops[stop_times[j]["stop_id"]]["parent_station"],
                        stop_times[j]["stop_id"],
                        stop_times[j]["arrival_time"],
                        stop_times[j].get(
                            "departure_time"
                        ) 
                    )

        # Process transfer connections
        print("Processing transfer connections...")
        main.process_transfers()
        
        # Validate        
        if validate_transit_network(main.graph.graph):

            # Save the graph
            print(f"Validation successful. Saving graph to {output_pickle}")
            main.graph.save_to_pickle(output_pickle)
        else :
            print(f"Validation unsuccessful. Not Saving graph!!!")
        
        return main.graph


def reconstruct_path(label: Label):
    """Reconstruct the path from a label."""
    path = [label.node]
    current = label
    while current:
        if current.node.node_type == NODE_TYPE.TRANSFER and  current.pred != None and current.pred.node.node_type != NODE_TYPE.TRANSFER:
            path.append(current.node)
        if current.pred == None:
            path.append(current.node)
        current = current.pred
    path.reverse()
   
    return (path, label.cost)
    
if __name__ == "__main__":
    # Configuration
    GTFS_PATH = "/home/ali/dev/TransferPatternGeneration/gtfs"
    PICKLE_PATH = "graph.pickle"

    try:
        logger.critical("Start building graph")
        graph = build_graph(GTFS_PATH, PICKLE_PATH).graph
        logger.critical("Graph processing completed successfully")

        # Print some statistics
        logger.critical(f"Total nodes: {len(graph.nodes())}")
        logger.critical(f"Total edges: {len(graph.edges())}")
        
        upper_bound = (timedelta(hours=24), 5)
        algorithm = MultiobjectiveDijkstra(graph, source="207310", target="200060", upper_bound=upper_bound)
        try:
            optimal_labels = pickle.load(open("/second/train/optimal_labels.pickle", "rb"))
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            algorithm.run()
            optimal_labels = algorithm.find_target_labels(algorithm.L)
            pickle.dump(optimal_labels, open("/second/train/optimal_labels.pickle", "wb"))
        
        algorithm.arrival_chain_algorithm(optimal_labels)
        
    except Exception as e:
        print(f"Error building graph: {e}")
        raise
