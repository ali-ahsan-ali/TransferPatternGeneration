from datetime import timedelta

def parse_time_with_overflow(time_str):
    # Split the time string into hours, minutes, and seconds
    hours, minutes, seconds = map(int, time_str.split(":"))
    
    # Convert the time into a timedelta, accounting for overflow
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)