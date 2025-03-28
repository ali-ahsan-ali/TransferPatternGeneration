import logging
from datetime import datetime
import sys
import os
import pickle
import pandas as pd


class GTFSParser:
    def __init__(
        self,
        gtfs_directory,
        debug_level=logging.DEBUG,
        pickle_path="gtfs_parser.pickle",
    ):
        self.gtfs_dir = gtfs_directory
        self.stop_times = dict()
        self.calendar = []
        self.stops = dict()
        self.trips = dict()
        self.pickle_path = pickle_path

        # Set up logging
        self.logger = logging.getLogger("GTFSParser")
        self.logger.setLevel(debug_level)

        # Create console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def load_gtfs_data(self, force_reload=False):
        """Load and preprocess GTFS data with pickle support"""
        if not force_reload and os.path.exists(self.pickle_path):
            self.logger.info("Found existing pickle file at %s", self.pickle_path)
            try:
                self._load_from_pickle()
                self.logger.info("Successfully loaded stop times from pickle file")
                return
            except Exception as e:
                self.logger.warning("Failed to load pickle file: %s", str(e))
                self.logger.info("Falling back to loading GTFS data from CSV files")

        self.logger.info("Loading GTFS data from %s", self.gtfs_dir)

        try:
            load_start = datetime.now()
            stop_dtype_dict = {
                "stop_id": "string",
                "parent_station": "string",
                "stop_name": "string",
                "location_type": "string",
            }
            self.logger.debug("Loading stops ...")
            self.stops = (
                pd.read_csv(
                    filepath_or_buffer=os.path.join(self.gtfs_dir, "stops.txt"),
                    dtype=stop_dtype_dict,
                )
                .set_index("stop_id")
                .to_dict("index")
            )

            stop_times_dtype_dict = {
                "trip_id": "string",
                "stop_id": "string",
                "arrival_time": "string",
                "stop_sequence": "string",
                "departure_time": "string",
            }

            trips_dtype_dict = {
                "trip_id": "string",
                "service_id": "string",
                "block_id": "string",
            }

            self.logger.debug("Loading trips...")
            initial_trips = (
                pd.read_csv(
                    filepath_or_buffer=os.path.join(self.gtfs_dir, "trips.txt"),
                    dtype=trips_dtype_dict,
                )
                .set_index("trip_id")
                .to_dict("index")
            )

            # Process stop times
            for key, value in initial_trips.items():
                self.trips[key] = value

            self.logger.debug("Loading stop times...")
            stop_times_initial = pd.read_csv(
                os.path.join(self.gtfs_dir, "stop_times.txt"),
                dtype=stop_times_dtype_dict,
            )
            self.logger.debug("Processing stop times...")

            # Process stop times
            for index, row in stop_times_initial.iterrows():
                if row["trip_id"] not in self.stop_times:
                    self.stop_times[row["trip_id"]] = []
                self.stop_times[row["trip_id"]].append(row.to_dict())
                
            for (key, value) in self.stop_times.items():
                self.stop_times[key] = sorted(value, key=lambda x: int(x['stop_sequence']))
            self.logger.debug("Loaded stop times")

            self.logger.debug("Loading Calendar...")
            self.calendar = pd.read_csv(os.path.join(self.gtfs_dir, "calendar.txt"))
            self.logger.debug("Loaded Calendar")

            load_time = datetime.now() - load_start
            self.logger.info("GTFS data loading completed in %s", str(load_time))

            # Save to pickle file
            self._save_to_pickle()
        except Exception as e:
            self.logger.error("Error loading GTFS data: %s", str(e))
            raise

    def _save_to_pickle(self):
        """Save the stop_times dictionary to a pickle file"""
        try:
            save_start = datetime.now()
            self.logger.info("Saving GTFS dictionary to pickle file...")

            with open(self.pickle_path, "wb") as f:
                data = {
                    "stop_times": self.stop_times,
                    "stops": self.stops,
                    "calendar": self.calendar,
                    "trips": self.trips,
                }

                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            save_time = datetime.now() - save_start
            self.logger.info(
                "All dictionary saved to %s in %s", self.pickle_path, str(save_time)
            )
        except Exception as e:
            self.logger.error("Failed to save pickle file: %s", str(e))

    def _load_from_pickle(self):
        """Load the stop_times dictionary from a pickle file"""
        load_start = datetime.now()
        self.logger.info("Loading stop_times dictionary from pickle file...")

        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
            self.stop_times = data["stop_times"]
            self.stops = data["stops"]
            self.calendar = data["calendar"]
            self.trips = data["trips"]

        load_time = datetime.now() - load_start
        self.logger.info(" All dictionary loaded in %s", str(load_time))
