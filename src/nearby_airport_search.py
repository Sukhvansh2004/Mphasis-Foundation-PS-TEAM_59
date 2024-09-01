import csv
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Open the provided text file and read the lines
with open('GlobalAirportDatabase.txt', 'r') as file:
    lines = file.readlines()

# Define the CSV file path
csv_file_path = 'GlobalAirportDatabase.csv'

# Open a new CSV file to write the data into
with open(csv_file_path, 'w', newline='') as csvfile:
    # Define the column headers for the CSV file
    fieldnames = [
        'ICAO', 'IATA', 'Airport Name', 'City', 'Country',
        'Latitude Degrees', 'Latitude Minutes', 'Latitude Seconds', 'Latitude Direction',
        'Longitude Degrees', 'Longitude Minutes', 'Longitude Seconds', 'Longitude Direction',
        'Elevation', 'Decimal Latitude', 'Decimal Longitude'
    ]

    # Create a CSV DictWriter object
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header to the CSV file
    writer.writeheader()

    # Process each line in the text file
    for line in lines:
        # Split the line into components based on the colon delimiter
        parts = line.strip().split(':')

        # Construct a dictionary for the row
        row_dict = {
            'ICAO': parts[0],
            'IATA': parts[1],
            'Airport Name': parts[2],
            'City': parts[3],
            'Country': parts[4],
            'Latitude Degrees': parts[5],
            'Latitude Minutes': parts[6],
            'Latitude Seconds': parts[7],
            'Latitude Direction': parts[8],
            'Longitude Degrees': parts[9],
            'Longitude Minutes': parts[10],
            'Longitude Seconds': parts[11],
            'Longitude Direction': parts[12],
            'Elevation': parts[13],
            'Decimal Latitude': parts[14],
            'Decimal Longitude': parts[15] if len(parts) > 15 else None  # Some lines may not have all fields
        }

        # Write the row to the CSV file
        writer.writerow(row_dict)

# The CSV file is now created at the specified path



# Function to calculate the distance between two points using the Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is approximately 6371
    distance_km = 6371 * c
    return distance_km

