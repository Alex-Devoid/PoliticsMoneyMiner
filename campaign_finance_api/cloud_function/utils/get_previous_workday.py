"""
Often you may need to get the date of the previous workday. 
This get_previous_workday() function finds the previous, non-federal-holidy workday.
"""
from datetime import datetime, timedelta
# Function to calculate federal holidays for the given year
def get_federal_holidays(year):
    holidays = [
        datetime(year, 1, 1),  # New Year's Day
        datetime(year, 6, 19),  # Juneteenth National Independence Day
        datetime(year, 7, 4),   # Independence Day
        datetime(year, 11, 11), # Veterans' Day
        datetime(year, 12, 25), # Christmas Day
    ]

    # Calculating dynamic holidays
    holidays.append(datetime(year, 1, 1) + timedelta(weeks=2, days=(0 - datetime(year, 1, 1).weekday())))  # Martin Luther King Jr. Birthday (3rd Monday of January)
    holidays.append(datetime(year, 2, 1) + timedelta(weeks=2, days=(0 - datetime(year, 2, 1).weekday())))  # Washington's Birthday (3rd Monday of February)
    holidays.append(datetime(year, 5, 31) - timedelta(days=datetime(year, 5, 31).weekday() + 1))  # Memorial Day (last Monday of May)
    holidays.append(datetime(year, 9, 1) + timedelta(weeks=0, days=(0 - datetime(year, 9, 1).weekday())))  # Labor Day (1st Monday in September)
    holidays.append(datetime(year, 10, 1) + timedelta(weeks=1, days=(0 - datetime(year, 10, 1).weekday())))  # Columbus Day (2nd Monday in October)
    holidays.append(datetime(year, 11, 1) + timedelta(weeks=3, days=(3 - datetime(year, 11, 1).weekday())))  # Thanksgiving Day (4th Thursday in November)

    return holidays


# Function to determine the last working day (skipping weekends and holidays)
def get_previous_workday():
    """
    Temporary override to return October 7, 2024 for testing purposes.
    """

    today = datetime.now()
    previous_day = today - timedelta(days=1)
    federal_holidays = get_federal_holidays(today.year)

    # Keep going back until we find a valid workday
    while previous_day.weekday() in [5, 6] or previous_day in federal_holidays:
        previous_day -= timedelta(days=1)

    return previous_day