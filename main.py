from Parser import GTFSParser
import pickle
import logging
from datetime import timedelta, datetime
from typing import List, Optional
from Graph import Node, TimeExpandedGraph, NODE_TYPE, TRAVEL_TYPE
from Djistras import Label, MultiobjectiveDijkstra
from utilities import parse_time_with_overflow
import networkx as nx
from pandas import *

currentDT = datetime.now()
FileName = currentDT.strftime("%Y%m%d%H%M%S")

# Set up logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("log/" + FileName + "main.log", mode="w")
    ]
)
logger = logging.getLogger("main")

logger.info("Initialized Main class and loaded GTFS data.")
        
class Main:
    def __init__(self, gtfs_paths: List[str]):
        """Initialize Main class with GTFS data path."""
        self.graph = TimeExpandedGraph()
        self.parser = GTFSParser(gtfs_paths)
        self.parser.load_gtfs_data()

    def add_vehicle_connection(
        self,
        station_a: str,
        child_stop_a: str,
        station_a_departure_time: str,
        station_b: str,
        child_stop_b: str,
        station_b_arrival_time: str,
        station_b_departure_time: str,
        station_b_pickup_type: int = 0, 
        station_b_dropoff_type: int = 0,
        isInitial: bool = False
    ) -> None:
        """Add nodes and edges for a vehicle connection between stations."""
        try:
            # Parse times
            dep_time = parse_time_with_overflow(station_a_departure_time)
            arr_time: timedelta = parse_time_with_overflow(station_b_arrival_time)
                
            # Create nodes
            dep_node = Node(station_a, self.parser.stops[station_a]["stop_name"], child_stop_a, self.parser.stops[child_stop_a]["stop_name"], dep_time, NODE_TYPE.DEPARTURE, None)
            arr_node = Node(station_b, self.parser.stops[station_b]["stop_name"], child_stop_b, self.parser.stops[child_stop_b]["stop_name"], arr_time, NODE_TYPE.ARRIVAL, station_b_dropoff_type)
            
            # Add riding edge
            self.graph.add_edge(dep_node, arr_node, TRAVEL_TYPE.NORMAL)

            if isInitial:
                transfer_node_initial = Node(station_a, self.parser.stops[station_a]["stop_name"], child_stop_a, self.parser.stops[child_stop_a]["stop_name"], dep_time, NODE_TYPE.TRANSFER, None)
                self.graph.add_edge(transfer_node_initial, dep_node, TRAVEL_TYPE.WAITINGCHAIN)

            # If vehicle continues, add staying edge
            if station_b_departure_time:
                next_dep_time = parse_time_with_overflow(station_b_departure_time)
                next_dep_node = Node(
                    station_b, self.parser.stops[station_b]["stop_name"], child_stop_b, self.parser.stops[child_stop_b]["stop_name"], next_dep_time, NODE_TYPE.DEPARTURE, None
                )
                self.graph.add_edge(arr_node, next_dep_node, TRAVEL_TYPE.STAYINGONTRAIN)

                # Create transfer node at arrival time
                if station_b_pickup_type == 0:
                    transfer_node = Node(
                        station_b, self.parser.stops[station_b]["stop_name"], child_stop_b,  self.parser.stops[child_stop_b]["stop_name"], next_dep_time, NODE_TYPE.TRANSFER, None
                    )
                    self.graph.add_edge(transfer_node, next_dep_node, TRAVEL_TYPE.WAITINGCHAIN)
                    
                else:
                    logger.debug(f"{station_a}, {self.parser.stops[station_a]["stop_name"]}, {child_stop_a} {self.parser.stops[station_b]["stop_name"]} {station_b}, {self.parser.stops[station_b]["stop_name"]} {child_stop_b} shits fucked for {station_b_pickup_type} at {next_dep_time}")
            
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
            if processed % 1000 == 0:  # Log progress every 1000 nodes
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
            elif nodes[i].node_type == NODE_TYPE.ARRIVAL and nodes[i].drop_off_type == 0:
                j = i + 1
                while j < total_nodes:
                    if nodes[j].station == nodes[i].station and nodes[j].node_type == NODE_TYPE.TRANSFER and nodes[j].time - nodes[i].time > timedelta(minutes=2):
                        self.graph.add_edge(
                            nodes[i], nodes[j], TRAVEL_TYPE.TRANSFER
                        )
                        break
                    j += 1
            # elif nodes[i].node_type == NODE_TYPE.DEPARTURE:
            #     # Connect predecessor to successor. Now transfers are straight to arrival nodes.
            #     for predecessor in self.graph.graph.predecessors(nodes[i]):
            #         for successor in self.graph.graph.successors(nodes[i]):
            #             self.graph.add_edge(predecessor, successor, TRAVEL_TYPE.NORMAL)
                
            #     self.graph.graph.remove_node(nodes[i])
    
if __name__ == "__main__":
    # Configuration
    GTFS_PATHS = ["/home/ali/dev/TransferPatternGeneration/gtfs/metro_GTFS_PROD_20250401121700", "/home/ali/dev/TransferPatternGeneration/gtfs/sydneytrains_GTFS_PROD_20250402140100"]
    PICKLE_PATH = "graph.pickle"

    try:
        logger.critical("Start building graph")
        try:
            # Try to load existing graph
            main = Main(GTFS_PATHS)

            main.graph.load_from_pickle(PICKLE_PATH)
            print(f"Loaded existing graph from {PICKLE_PATH}")

        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            print(f"Building new graph from GTFS data at {GTFS_PATHS}")

            # Initialize main processor
            main = Main(GTFS_PATHS)

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
                
                if days_running["monday"] != 1 or days_running["start_date"] < 20250403:
                    continue

                # Process each stop in the trip
                logger.warning(f"{stop_times[0]["trip_id"]}, {trip_id}")
                isInitial = True
                for i in range(length - 1):

                    # Ignore the start of a trip if it's not picking anyone up and keep ignoring until it does. 
                    if i == 0 :
                        while i < length - 1 and stop_times[i]["pickup_type"] != 0:
                            i+= 1
                            
                    if i == length - 1 :  continue
                    
                    j = i+1
                        
                    main.add_vehicle_connection(
                        main.parser.stops[stop_times[i]["stop_id"]]["parent_station"],
                        stop_times[i]["stop_id"],
                        stop_times[i]["departure_time"],
                        main.parser.stops[stop_times[j]["stop_id"]]["parent_station"],
                        stop_times[j]["stop_id"],
                        stop_times[j]["arrival_time"],
                        stop_times[j]["departure_time"],
                        stop_times[j]["pickup_type"],
                        stop_times[j]["drop_off_type"],
                        isInitial
                    )

                    isInitial = False

            # Process transfer connections
            print("Processing transfer connections...")
            main.process_transfers()
            
            # Save the graph
            print(f"Saving graph to {PICKLE_PATH}")
            main.graph.save_to_pickle(PICKLE_PATH)
            logger.critical("Graph processing completed successfully")
                    
        # Print some statistics
        logger.critical(f"Total nodes: {len(main.graph.graph.nodes())}")
        logger.critical(f"Total edges: {len(main.graph.graph.edges())}")
            
        upper_bound = (timedelta(hours=24), 5)
        source = "207310"

        algorithm = MultiobjectiveDijkstra(main.graph.graph, source=source, upper_bound=upper_bound)
        try:
            optimal_labels = pickle.load(open(f"/home/ali/dev/TransferPatternGeneration/optimal_labels_{source}.pickle", "rb"))
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            algorithm.run()
            optimal_labels = algorithm.L
            pickle.dump(algorithm.L, open(f"/home/ali/dev/TransferPatternGeneration/optimal_labels_{source}.pickle", "wb"))
        
        stops = main.parser.stops.keys()
        for stop_id in stops:
            # if stop_id != "200060": continue

            if main.parser.stops[stop_id]["parent_station"] == "" or main.parser.stops[stop_id]["parent_station"] == None:
                target_labels = algorithm.find_target_labels(optimal_labels, stop_id)
                # try:
                    # transfer_pattern = pickle.load(open(f"/home/ali/dev/TransferPatternGeneration/transfer_pattern/transfer_pattern_{source}_{stop_id}.pickle", "rb"))
                # except (FileNotFoundError, pickle.UnpicklingError, EOFError):
                transfer_pattern = algorithm.arrival_chain_algorithm(target_labels)
                    # pickle.dump(transfer_pattern, open(f"/home/ali/dev/TransferPatternGeneration/transfer_pattern/transfer_pattern_{source}_{stop_id}.pickle", "wb"))
                
                logger.critical(f"{source} to {stop_id} is below: ")
                for time in transfer_pattern:
                    for trip in transfer_pattern[time]:
                        logger.critical(trip)
                logger.critical("\n")

    except Exception as e:
        print(f"Error building graph: {e}")
        raise
