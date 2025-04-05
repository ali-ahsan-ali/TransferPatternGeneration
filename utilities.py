from datetime import timedelta
import logging 

# Set up logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("log/" + "utilities.log", mode="w")
    ]
)
logger = logging.getLogger("utilities")

def parse_time_with_overflow(time_str):
    # Split the time string into hours, minutes, and seconds
    hours, minutes, seconds = map(int, time_str.split(":"))
    
    # Convert the time into a timedelta, accounting for overflow
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)