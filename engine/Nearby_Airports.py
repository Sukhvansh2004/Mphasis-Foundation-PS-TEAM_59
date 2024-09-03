import pandas as pd
from nearby_airport_search import haversine


def nearby_airport(cancelled_flight_departure_airport, radius):
    df = pd.read_csv('GlobalAirportDatabase.csv')

    reference_airport_iata = cancelled_flight_departure_airport
    search_radius_km_D = radius
    
    df_airports = df
    
    # Check if the reference airport exists in the DataFrame
    if reference_airport_iata in df_airports['IATA'].values:
        # Get the reference airport's latitude and longitude
        reference_airport = df_airports[df_airports['IATA'] == reference_airport_iata].iloc[0]
        # print('INFORMATION ABOUT THE DESTINATION AIRPORT')
        # print(reference_airport,'\n')
        ref_lat, ref_lon = reference_airport['Decimal Latitude'], reference_airport['Decimal Longitude']
    
        # Calculate the distance of all airports from the reference airport
        df_airports['Distance_km'] = df_airports.apply(
            lambda row: haversine(ref_lon, ref_lat, row['Decimal Longitude'], row['Decimal Latitude']),
            axis=1
        )
        df_nearby_airports = df_airports[df_airports['Distance_km'] <= search_radius_km_D]
    
        # Drop the reference airport itself from the list
        df_nearby_airports = df_nearby_airports[df_nearby_airports['IATA'] != reference_airport_iata]
        # print('TOTAL NEARBY AIRPORTS FOUND: ',len(df_nearby_airports))
        # Print the nearby airports
        # print(df_nearby_airports[['ICAO', 'IATA', 'Airport Name', 'City', 'Country', 'Distance_km']])
        # print('\n')
        # print('IATA code for nearby airports: ')
        # print(df_nearby_airports)
        df_nearby_airports.dropna(subset=['IATA'], inplace=True)
        E_prime_d = df_nearby_airports['IATA'].tolist()
    else:
        return []
        # print(f"Airport with IATA code '{reference_airport_iata}' not found in the database.") 
        pass
    return E_prime_d

   
        
        
        
        
        
        