import pickle
from utilities import parse_time_with_overflow
import logging 
from datetime import timedelta
import json
from json import *

# Logger setup
logging.basicConfig(
    filename="lines_processing.log",
    format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
    filemode='w',
    level=logging.DEBUG
)
logger = logging.getLogger("lines")
handler = logging.FileHandler("lines_processing.log")
handler.setFormatter(logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s'))
logger.addHandler(handler)

class Lines:
    def __init__(self):
        self.lines = []

    def add_trip(self, stop_times_array, calendar):
        matched = False
        for line in self.lines:
            if line.add_trip(stop_times_array, calendar):
                matched = True
                break
        if not matched:
            new_line = Line()
            new_line.force_add_trip(stop_times_array, calendar)
            self.lines.append(new_line)

    def default(self, obj):
        if isinstance(obj, Lines):
            return {
                "__type__": "Lines",
                "lines": obj.lines
            }
        elif isinstance(obj, Line):
            return {
                "__type__": "Line",
                "stops": obj._stops,
                "trip_times": obj._trip_times,
                "calendar": obj.calendar
            }
        return super().default(obj)

class Line:
    def __init__(self):
        self._stops = []
        self._trip_times = []
        self.calendar = dict()

        # Logger setup

    def force_add_trip(self, stop_times_array, calendar):
        self._stops = [d['stop_id'] for d in stop_times_array]

        self._trip_times.append(
            [(parse_time_with_overflow(d['arrival_time']), parse_time_with_overflow(d['departure_time'])) for d in stop_times_array]
        )

        return True
    
    def add_trip(self, stop_times_array, calendar):
        if self.candidate(stop_times_array, calendar):
            # logger.debug(stop_times_array)
            # logger.debug([d['stop_id'] for d in stop_times_array])
            self._trip_times.append(
                [(parse_time_with_overflow(d['arrival_time']), parse_time_with_overflow(d['departure_time'])) for d in stop_times_array]
            )

            return True

        return False

    def candidate(self, stop_times_array, calendar):
        num_stops = len(self._stops)
        if num_stops and num_stops != len(stop_times_array):
            return False

        for i in range(num_stops):
            stop = self._stops[i]
            if stop != stop_times_array[i]["stop_id"]:
                # The trip does not share the same sequence of stations.
                # logger.debug("Fucked stops {0} {1}".format(stop, stop_times_array[i]["stop_id"]))
                # logger.debug(self._stops)
                # logger.debug([d['stop_id'] for d in stop_times_array])
                return False
            elif i > 0:
                # Should be ordered, so last stop should be prior to this one
                (comparison_arrive_time, _) = self._trip_times[-1][i]
                (_, comparison_departure_time) = self._trip_times[-1][i-1]
                line_travel_time = comparison_arrive_time - comparison_departure_time
                new_arrival_time = parse_time_with_overflow(stop_times_array[i]["arrival_time"])
                new_departure_time = parse_time_with_overflow(stop_times_array[i-1]["departure_time"])
                trip_travel_time =new_arrival_time - new_departure_time
                
                # Overtakes? 
                # if line_travel_time != trip_travel_time and new_departure_time >= comparison_departure_time and new_arrival_time <= comparison_arrive_time:
                #     logger.debug("Fucked Overtakes {0} {1} {2} {3}".format( comparison_departure_time, comparison_arrive_time, \
                #                 new_departure_time, new_arrival_time ))
                #     return False

                # # Overtaken? 
                # if line_travel_time != trip_travel_time and comparison_departure_time >= new_departure_time and comparison_arrive_time <= new_arrival_time:
                #     logger.debug("Fucked Overtaken {0} {1} {2} {3}".format( comparison_departure_time, comparison_arrive_time, \
                #                 new_departure_time, new_arrival_time ))
                #     return False

        return True
    
    # Save the Line instance to a pickle file
    def save_line_to_pickle(line, filename):
        with open(filename, 'wb') as file:
            pickle.dump(line, file)

    # Load the Line instance from a pickle file
    def load_line_from_pickle(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def __str__(self):
        return f"Stops: {self._stops}" # , Trip Times: {self._trip_times}

import json
from json import JSONEncoder
from datetime import timedelta
from typing import List, Dict, Tuple

class LinesEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Lines):
            return {
                "__type__": "Lines",
                "lines": obj.lines
            }
        elif isinstance(obj, Line):
            return {
                "__type__": "Line",
                "stops": obj._stops,
                "trip_times": obj._trip_times,
                "calendar": obj.calendar
            }
        elif isinstance(obj, timedelta):
            # Convert timedelta to total seconds
            return {
                "__type__": "timedelta",
                "seconds": obj.total_seconds()
            }
        return super().default(obj)

def lines_decoder(obj):
    if "__type__" in obj:
        if obj["__type__"] == "Lines":
            lines_obj = Lines()
            lines_obj.lines = [lines_decoder(line) for line in obj["lines"]]
            return lines_obj
        elif obj["__type__"] == "Line":
            line_obj = Line()
            line_obj._stops = obj["stops"]
            line_obj._trip_times = obj["trip_times"]
            line_obj.calendar = obj["calendar"]
            return line_obj
        elif obj["__type__"] == "timedelta":
            return timedelta(seconds=obj["seconds"])
    return obj

def save_lines(lines_obj: Lines, filename: str):
    """Save Lines object to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(lines_obj, f, cls=LinesEncoder, indent=2)
        print(f"Successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving: {e}")
        raise  # Re-raise the exception to see the full error trace if needed

def load_lines(filename: str) -> Lines:
    """Load Lines object from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.loads(f.read(), object_hook=lines_decoder)
    except Exception as e:
        print(f"Error loading: {e}")
        return None

