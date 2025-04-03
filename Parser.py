import logging
from datetime import datetime
import sys
import os
import pickle
import pandas as pd


class GTFSParser:
    def __init__(
        self,
        gtfs_directories,  # Now accepts a list of directories
        debug_level=logging.DEBUG,
        pickle_path="gtfs_parser.pickle",
    ):
        # Ensure gtfs_directories is a list
        self.gtfs_dirs = gtfs_directories if isinstance(gtfs_directories, list) else [gtfs_directories]
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
        """Load and preprocess GTFS data from multiple directories with pickle support"""
        if not force_reload and os.path.exists(self.pickle_path):
            self.logger.info("Found existing pickle file at %s", self.pickle_path)
            try:
                self._load_from_pickle()
                self.logger.info("Successfully loaded data from pickle file")
                return
            except Exception as e:
                self.logger.warning("Failed to load pickle file: %s", str(e))
                self.logger.info("Falling back to loading GTFS data from CSV files")

        self.logger.info("Loading GTFS data from %d directories", len(self.gtfs_dirs))
        
        try:
            load_start = datetime.now()
            
            # Initialize empty DataFrames to merge data from multiple sources
            all_stops_df = pd.DataFrame()
            all_stop_times_df = pd.DataFrame()
            all_trips_df = pd.DataFrame()
            all_calendar_df = pd.DataFrame()
            
            # Process each GTFS directory
            for dir_index, gtfs_dir in enumerate(self.gtfs_dirs):
                self.logger.info("Processing GTFS directory %d/%d: %s", 
                                dir_index + 1, len(self.gtfs_dirs), gtfs_dir)
                
                # Define datatypes for each file
                stop_dtype_dict = {
                    "stop_id": "string",
                    "parent_station": "string",
                    "stop_name": "string",
                    "location_type": "string",
                }
                
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
                
                # Load stops
                stops_path = os.path.join(gtfs_dir, "stops.txt")
                if os.path.exists(stops_path):
                    self.logger.debug("Loading stops from %s...", gtfs_dir)
                    stops_df = pd.read_csv(stops_path, dtype=stop_dtype_dict)
                    stops_df["source_dir"] = gtfs_dir
                    stops_df["source_index"] = dir_index
                    self.logger.info("Loaded %d stops from %s", len(stops_df), gtfs_dir)
                    all_stops_df = pd.concat([all_stops_df, stops_df])
                else:
                    self.logger.warning("Stops file not found in %s", gtfs_dir)
                
                # Load trips
                trips_path = os.path.join(gtfs_dir, "trips.txt")
                if os.path.exists(trips_path):
                    self.logger.debug("Loading trips from %s...", gtfs_dir)
                    trips_df = pd.read_csv(trips_path, dtype=trips_dtype_dict)
                    trips_df["source_dir"] = gtfs_dir
                    trips_df["source_index"] = dir_index
                    self.logger.info("Loaded %d trips from %s", len(trips_df), gtfs_dir)
                    all_trips_df = pd.concat([all_trips_df, trips_df])
                else:
                    self.logger.warning("Trips file not found in %s", gtfs_dir)
                
                # Load stop times
                stop_times_path = os.path.join(gtfs_dir, "stop_times.txt")
                if os.path.exists(stop_times_path):
                    self.logger.debug("Loading stop times from %s...", gtfs_dir)
                    stop_times_df = pd.read_csv(stop_times_path, dtype=stop_times_dtype_dict)
                    stop_times_df["source_dir"] = gtfs_dir
                    stop_times_df["source_index"] = dir_index
                    self.logger.info("Loaded %d stop times from %s", len(stop_times_df), gtfs_dir)
                    all_stop_times_df = pd.concat([all_stop_times_df, stop_times_df])
                else:
                    self.logger.warning("Stop times file not found in %s", gtfs_dir)
                
                # Load calendar
                calendar_path = os.path.join(gtfs_dir, "calendar.txt")
                if os.path.exists(calendar_path):
                    self.logger.debug("Loading calendar from %s...", gtfs_dir)
                    calendar_df = pd.read_csv(calendar_path)
                    calendar_df["source_dir"] = gtfs_dir
                    calendar_df["source_index"] = dir_index
                    self.logger.info("Loaded %d calendar entries from %s", len(calendar_df), gtfs_dir)
                    all_calendar_df = pd.concat([all_calendar_df, calendar_df])
                else:
                    self.logger.warning("Calendar file not found in %s", gtfs_dir)
            
            # Remove duplicates from stops DataFrame by keeping first occurrence
            self.logger.info("Found %d total stops before deduplication", len(all_stops_df))
            duplicated_stops = all_stops_df.duplicated(subset=["stop_id"], keep="first")
            duplicate_count = duplicated_stops.sum()
            if duplicate_count > 0:
                self.logger.warning("Found %d duplicate stop_ids across sources", duplicate_count)
            
            # Drop duplicates, keeping first occurrence
            all_stops_df = all_stops_df[~all_stops_df.duplicated(subset=["stop_id"], keep="first")]
            self.logger.info("Kept %d unique stops after deduplication", len(all_stops_df))
            
            # Similarly handle duplicates in trips
            self.logger.info("Found %d total trips before deduplication", len(all_trips_df))
            duplicate_trips = all_trips_df.duplicated(subset=["trip_id"], keep="first").sum()
            if duplicate_trips > 0:
                self.logger.warning("Found %d duplicate trip_ids across sources", duplicate_trips)
            all_trips_df = all_trips_df[~all_trips_df.duplicated(subset=["trip_id"], keep="first")]
            self.logger.info("Kept %d unique trips after deduplication", len(all_trips_df))
            
            # Convert merged DataFrames to the required formats
            self.logger.info("Processing merged data with %d stops, %d trips, %d stop times, and %d calendar entries", 
                         len(all_stops_df), len(all_trips_df), len(all_stop_times_df), len(all_calendar_df))
            
            # Process stops
            self.logger.debug("Converting stops to dictionary...")
            self.stops = all_stops_df.set_index("stop_id").to_dict("index")
            self.logger.info("Processed %d unique stops", len(self.stops))
            
            # Process trips
            self.logger.debug("Converting trips to dictionary...")
            initial_trips = all_trips_df.set_index("trip_id").to_dict("index")
            
            # Copy initial trips
            for key, value in initial_trips.items():
                self.trips[key] = value
            self.logger.info("Processed %d unique trips", len(self.trips))
            
            # Process stop times
            self.logger.debug("Processing merged stop times...")
            progress_interval = max(1, len(all_stop_times_df) // 10)  # Log every 10%
            processed_rows = 0
            
            # Group stop times by trip_id for more efficient processing
            self.logger.debug("Grouping stop times by trip_id...")
            stop_times_grouped = dict(tuple(all_stop_times_df.groupby("trip_id")))
            self.logger.info("Stop times grouped into %d unique trip_ids", len(stop_times_grouped))
            
            # Process each trip group
            self.logger.debug("Processing stop times for each trip...")
            trip_count = 0
            total_trips = len(stop_times_grouped)
            
            for trip_id, group_df in stop_times_grouped.items():
                trip_count += 1
                if trip_count % max(1, total_trips // 20) == 0:  # Log progress at 5% intervals
                    self.logger.info("Processing stop times: %d/%d trips (%.1f%%)", 
                                    trip_count, total_trips, (trip_count / total_trips * 100))
                
                # Convert dataframe rows to list of dictionaries
                group_records = group_df.to_dict("records")
                processed_rows += len(group_records)
                
                # Check if trip has a block_id
                block_id = None
                if trip_id in self.trips and "block_id" in self.trips[trip_id]:
                    block_id = self.trips[trip_id]["block_id"]
                
                if not block_id or block_id == "":
                    # Use the trip_id as is
                    if trip_id not in self.stop_times:
                        self.stop_times[trip_id] = []
                    self.stop_times[trip_id].extend(group_records)
                else:
                    # Create a super trip using block_id
                    super_trip_id = f"super_trip_{block_id}"
                    if super_trip_id not in self.stop_times:
                        self.stop_times[super_trip_id] = []
                    self.stop_times[super_trip_id].extend(group_records)
            
            self.logger.info("Processed all %d stop time entries", processed_rows)
            
            # Process and clean up stop times
            self.logger.debug("Cleaning and sorting stop times...")
            total_keys = len(self.stop_times)
            processed_keys = 0
            
            for key, value in self.stop_times.items():
                processed_keys += 1
                if processed_keys % max(1, total_keys // 20) == 0:  # Log progress at 5% intervals
                    self.logger.info("Cleaning stop times: %d/%d trips (%.1f%%)", 
                                    processed_keys, total_keys, (processed_keys / total_keys * 100))
                
                # Sort by arrival time
                value = sorted(value, key=lambda x: x['arrival_time'])
                new_val = []
                prev = None
                
                for val in value:
                    # Skip stops that can't be used for pickup or drop-off
                    if val["pickup_type"] != 0 and val["drop_off_type"] != 0:
                        self.logger.debug("Removing useless stop time entry: %s", val)
                        continue
                    # Handle duplicate stops (same location, one for pickup, one for drop-off)
                    elif (prev and prev["pickup_type"] == 1 and prev["drop_off_type"] == 0 
                          and val["pickup_type"] == 0 and val["drop_off_type"] == 1 
                          and prev["stop_id"] == val["stop_id"]):
                        # Remove duplicate
                        self.logger.critical("Removing duplicate stop time entry: %s", val)
                        new_val.pop()
                        val["drop_off_type"] = 0
                        val["arrival_time"] = prev["arrival_time"]
                    
                    new_val.append(val)
                    prev = val
                
                self.stop_times[key] = new_val
            
            # Process calendar - also handle potential duplicates in calendar
            self.logger.info("Found %d total calendar entries before deduplication", len(all_calendar_df))
            if not all_calendar_df.empty and "service_id" in all_calendar_df.columns:
                duplicate_calendar = all_calendar_df.duplicated(subset=["service_id"], keep="first").sum()
                if duplicate_calendar > 0:
                    self.logger.warning("Found %d duplicate service_ids across sources", duplicate_calendar)
                all_calendar_df = all_calendar_df[~all_calendar_df.duplicated(subset=["service_id"], keep="first")]
            self.logger.info("Kept %d unique calendar entries after deduplication", len(all_calendar_df))
            
            all_calendar_df["service_id"] = all_calendar_df["service_id"].values.astype("str")
            self.calendar = all_calendar_df
            
            load_time = datetime.now() - load_start
            self.logger.info("GTFS data loading completed in %s", str(load_time))
            
            # Save to pickle file
            self._save_to_pickle()
        
        except Exception as e:
            self.logger.error("Error loading GTFS data: %s", str(e))
            raise

    def _save_to_pickle(self):
        """Save the merged data to a pickle file"""
        try:
            save_start = datetime.now()
            self.logger.info("Saving merged GTFS dictionary to pickle file...")

            with open(self.pickle_path, "wb") as f:
                data = {
                    "stop_times": self.stop_times,
                    "stops": self.stops,
                    "calendar": self.calendar,
                    "trips": self.trips,
                }

                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            save_time = datetime.now() - save_start
            pickle_size_mb = os.path.getsize(self.pickle_path) / (1024 * 1024)
            self.logger.info(
                "All merged dictionary saved to %s (%.2f MB) in %s", 
                self.pickle_path, pickle_size_mb, str(save_time)
            )
        except Exception as e:
            self.logger.error("Failed to save pickle file: %s", str(e))

    def _load_from_pickle(self):
        """Load the merged data dictionary from a pickle file"""
        load_start = datetime.now()
        self.logger.info("Loading merged dictionaries from pickle file...")

        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
            self.stop_times = data["stop_times"]
            self.stops = data["stops"]
            self.calendar = data["calendar"]
            self.trips = data["trips"]

        load_time = datetime.now() - load_start
        self.logger.info("All merged dictionaries loaded in %s", str(load_time))
        self.logger.info("Loaded %d stops, %d trips, %d trip stop times, %d calendar entries", 
                      len(self.stops), len(self.trips), len(self.stop_times), len(self.calendar))