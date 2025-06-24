from pnr_reaccomodation import *
from dimod import BINARY
from docplex.mp.model import Model
from dimod.constrained import cqm_to_bqm
from neal import SimulatedAnnealingSampler
from qiskit_optimization.translators import from_docplex_mp
import numpy as np
import math
import pandas as pd
from datetime import timedelta
from nearby_airport_search import haversine
from path_scoring import *
import os

METHOD = 'SIMULATE'
moduleDir = os.path.dirname(os.path.abspath(__file__))
def next_72hrsflights(flight_number, canceled_departure_datetime, sorted_dataframe):
    canceled_flight_index = sorted_dataframe[
        (sorted_dataframe['FlightNumber'] == flight_number) &
        (sorted_dataframe['DepartureDateTime'] == canceled_departure_datetime)
    ].index[0]

    canceled_flight_departure_datetime = sorted_dataframe.loc[canceled_flight_index, 'DepartureDateTime']
    window_end = canceled_flight_departure_datetime + timedelta(hours=72)


    sorted_dataframe['ArrivalDateTime'] = pd.to_datetime(sorted_dataframe['ArrivalDateTime'])

    nearby_flights = sorted_dataframe[
        (sorted_dataframe['ArrivalDateTime'] <= pd.to_datetime(window_end)) &
        (sorted_dataframe['DepartureDateTime'] > pd.to_datetime(canceled_departure_datetime))
    ]

    return nearby_flights

def get_direct_flights(E_prime_d, E_prime, dataframe):

    filtered_flights = pd.DataFrame()

    # Loop through each departure airport in E_prime_d
    for departure_airport in E_prime_d:
        # Loop through each destination airport in E_prime
        for arrival_airport in E_prime:
            # Filter the dataframe for flights from the current departure airport to the current destination airport
            current_flights = dataframe[
                (dataframe['DepartureAirport'] == departure_airport) &
                (dataframe['ArrivalAirport'] == arrival_airport)
            ]
            # Append the filtered flights to the main DataFrame
            filtered_flights = pd.concat([filtered_flights, current_flights], ignore_index=True)

    return filtered_flights

def get_1_interconnecting(E_prime_d, E_prime, flights_df):
    solutions = []
    # for each possible start and end
    for dep in E_prime_d:
        a_flights = flights_df[flights_df["DepartureAirport"] == dep]
        for arr in E_prime:
            b_flights = flights_df[flights_df["ArrivalAirport"] == arr]
            # look for any a->X and X->b
            for _, a_row in a_flights.iterrows():
                mid = a_row["ArrivalAirport"]
                # all B’s that depart from that mid-point
                matches = b_flights[b_flights["DepartureAirport"] == mid]
                for _, b_row in matches.iterrows():
                    # build a 2-row DataFrame
                    sol = pd.DataFrame([a_row, b_row])
                    solutions.append(sol)

    if solutions:
        return pd.concat(solutions, ignore_index=True)
    else:
        # empty with same columns
        return flights_df.iloc[0:0].copy()

def get_2_interconnecting(E_prime_d, E_prime, flights_df):
    solutions = []
    for dep in E_prime_d:
        a_flights = flights_df[flights_df["DepartureAirport"] == dep]
        for arr in E_prime:
            b_flights = flights_df[flights_df["ArrivalAirport"] == arr]
            # for every a-leg and b-leg, see if there’s an actual connecting c-leg
            for _, a_row in a_flights.iterrows():
                for _, b_row in b_flights.iterrows():
                    mid1 = a_row["ArrivalAirport"]
                    mid2 = b_row["DepartureAirport"]
                    # we want a_row.dep -> mid1, then mid1->mid2, then mid2->b_row.arr
                    c_matches = flights_df[
                        (flights_df["DepartureAirport"] == mid1) &
                        (flights_df["ArrivalAirport"] == mid2) &
                        (flights_df["DepartureAirport"] != dep) &
                        (flights_df["ArrivalAirport"] != arr)
                    ]
                    for _, c_row in c_matches.iterrows():
                        # build a 3-row DataFrame
                        sol = pd.DataFrame([a_row, c_row, b_row])
                        solutions.append(sol)

    if solutions:
        return pd.concat(solutions, ignore_index=True)
    else:
        return flights_df.iloc[0:0].copy()


"""NOW WE HAVE DEFINED THE BINARY VARIABLES OF THE FORM 'Q_ikjl' THAT REPRESENTS WHETHER WE SELECT A FLIGHT IN OUR PATH OR NOT. HERE THE FIRST INDEX REPRESENTS THE START NODE OF OUR FLIGHT, THE SECOND INDEX REPRESENTS THE END NODE, THE THIRD INDEX REPRESENTS AIRCRAFT ID AND THE FOURTH INDEX REPRESENTS THE DEPARTURE TIME.WE HAVE ENCODED THE POSSIBLE VALUES FOR THESE INDICES WITHE THE HELP OF THE FUNCTION GIVEN BELOW.

    ALL THESE INDICES UNIQULY IDENTIFIES A FLIGHT.

    THE FUNCTION IS CREATING  3 4-DIMENSIONAL TENSORS:
    1) THE FIRST TENSOR IS A REFERENCE TENSOR THAT CONTAINS 1 AT THE INDICES WHERE BINARY VARIABLES ACTUALLY EXISTS AND 0 WHERE IT DOESN'T.
    2) THE SECOND AND THIRD TENSOR CONTAIN DEPARTURE AND ARRIVAL TIME OF FLIGHTS AT THEIR RESPECTIVE INDICES AND 0 WHERE NO SUCH FLIGHT EXISTS.(WE WILL USE THESE TENSORS AHEAD)
    """

def encode_flights(dataframe, unique_airports, flight_numbers, departure_times):

    departure_airport_dict = {airport: idx for idx, airport in enumerate(unique_airports)}
    arrival_airport_dict   = {airport: idx for idx, airport in enumerate(unique_airports)}
    flight_number_dict     = {flight:  idx for idx, flight   in enumerate(flight_numbers)}
    departure_time_dict    = {pd.to_datetime(time): idx for idx, time   in enumerate(departure_times)}

    # existence mask stays the same
    flight_existence_array = np.zeros(
        (len(unique_airports),
         len(unique_airports),
         len(flight_numbers),
         len(departure_times)),
        dtype=int
    )

    # create empty object arrays and fill with None
    shape = flight_existence_array.shape
    departure_datetime_array = np.empty(shape, dtype=object)
    arrival_datetime_array   = np.empty(shape, dtype=object)
    departure_datetime_array[:] = None
    arrival_datetime_array[:]   = None

    for _, row in dataframe.iterrows():
        i = departure_airport_dict[row["DepartureAirport"]]
        j = arrival_airport_dict[row["ArrivalAirport"]]
        k = flight_number_dict[row["FlightNumber"]]
        dt = pd.to_datetime(row["DepartureDateTime"])
        l  = departure_time_dict[dt]

        flight_existence_array[i, j, k, l]   = 1
        departure_datetime_array[i, j, k, l] = row["DepartureDateTime"]
        arrival_datetime_array[i, j, k, l]   = row["ArrivalDateTime"]

    return (
        flight_existence_array,
        departure_datetime_array,
        arrival_datetime_array,
        arrival_airport_dict,
        departure_airport_dict,
    )


def decode(variable, unique_airports, flight_numbers, departure_time, df):
    sol = []
    for i in range(len(variable)):
        depart_airport = unique_airports[variable[i][0]]
        arriv_airport = unique_airports[variable[i][1]]
        flght_no = flight_numbers[variable[i][2]]
        depart_time = departure_time[variable[i][3]]

        decoded_data = df[
            (df["DepartureAirport"] == depart_airport) &
            (df["ArrivalAirport"] == arriv_airport) &
            (df["FlightNumber"] == flght_no) &
            (df["DepartureDateTime"] == depart_time)
        ]

        if not decoded_data.empty:
            # Assuming "InventoryId" is a string column, use iloc[0] to get the first value
            sol.append(decoded_data["InventoryId"].iloc[0])

    return sol

def main(*disruptions, INVENTORY_FILE=os.path.join(moduleDir, "Files", "inv.csv"), AIRPORT_FILE = os.path.join(moduleDir, 'GlobalAirportDatabase.csv'), PNR_FILE = os.path.join(moduleDir, "Files", "pnrb.csv"), PASSENGER_LIST = os.path.join(moduleDir, "Files", "pnrp.csv"), TOKEN = 'DEV-6ddf205adb6761bc0018a65f2496245457fe977f', method=METHOD):
    inventory_dataframe=pd.read_csv(INVENTORY_FILE)

    for disrupt in disruptions:
        inventory_dataframe["DepartureDateTime"]=pd.to_datetime(inventory_dataframe["DepartureDateTime"])
        inventory_dataframe["ArrivalDateTime"]=pd.to_datetime(inventory_dataframe["ArrivalDateTime"])

        schedule_dataframe = inventory_dataframe.sort_values(by="DepartureDateTime").reset_index(drop=True)

        index = inventory_dataframe.index[inventory_dataframe['InventoryId'] == disrupt].tolist()[0]
        # print(schedule_dataframe.index[schedule_dataframe['InventoryId']=="INV-ZZ-2774494"].tolist())
        # print(schedule_dataframe.iloc[index])

        cancelled_flight=schedule_dataframe.iloc[index]
        reduced_flights=next_72hrsflights(cancelled_flight["FlightNumber"],cancelled_flight["DepartureDateTime"],schedule_dataframe)

        # print(cancelled_flight["DepartureDateTime"])
        # print(reduced_flights["DepartureDateTime"])

        """BELOW WE HAVE DEFINED THRREE FUNCTIONS THAT FINDS ALL THE DIRECT FLIGHTS WITHIN THAT 72 HOUR TIME WINDOW AND FINDS POSSIBLE INTERCONNECTING NODES WHEN CONSIDERING PATHS UPTO 2 INTERCONNECTING NODES.(THE FUNCTIONS BELOW DON'T TAKE TIME INTO ACCOUNT i.e the downline flights may be backward in time)"""



        cancelled_flight_departure_airport=cancelled_flight["DepartureAirport"]
        cancelled_flight_arrival_airport=cancelled_flight["ArrivalAirport"]

        """NOW I WILL ATTEMPT TO FIND AIRPORTS NEAR THE ARRIVAL AIRPORT WITHIN SOME EPSILON RADIUS AND AIRPORTS NEAR DEPARTURE AIRPORT ALSO WITHIN SOME EPSILON RADIUS WHICH CAN BE TUNED ACCORDING TO THE USER.
        THIS TAKES FLIGHTS WITH VARIABLE DESTINATION(NEAR THE ORIGINAL DESTINATION) AND FLIGHTS WITH VARIABLE DEPARTURE AIRPORT(NEAR ORIGINAL DEPARTURE AIRPORT)INTO ACCOUNT.
        """

        reference_airport_iata = cancelled_flight_arrival_airport
        search_radius_km_A = 2

        df_airports = pd.read_csv(AIRPORT_FILE)

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
            df_nearby_airports = df_airports[df_airports['Distance_km'] <= search_radius_km_A]

            # Drop the reference airport itself from the list
            df_nearby_airports = df_nearby_airports[df_nearby_airports['IATA'] != reference_airport_iata]
            # print('TOTAL NEARBY AIRPORTS FOUND: ',len(df_nearby_airports))
            # # Print the nearby airports
            # print(df_nearby_airports[['ICAO', 'IATA', 'Airport Name', 'City', 'Country', 'Distance_km']])
            # print('\n')
            # print('IATA code for nearby airports: ')
            E_prime=df_nearby_airports['IATA'].tolist()
            # print(E_prime)
        else:
            print(f"Airport with IATA code '{reference_airport_iata}' not found in the database.")

        reference_airport_iata = cancelled_flight_departure_airport
        search_radius_km_D = 5

        
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
            # # Print the nearby airports
            # print(df_nearby_airports[['ICAO', 'IATA', 'Airport Name', 'City', 'Country', 'Distance_km']])
            # print('\n')
            # print('IATA code for nearby airports: ')
            E_prime_d=df_nearby_airports['IATA'].tolist()
            # print(E_prime_d)
        else:
            raise AssertionError(f"Airport with IATA code '{reference_airport_iata}' not found in the database.")

        """NOW THE VARIABLE E_prime CONTAINS IATA CODE FOR POSSIBLE DESTINATIONS AND THE VARIABLE E_prime_d CONTAINS IATA CODE FOR POSSIBLE DEPARTURE AIRPORTS NEARBY THE ORIGINAL DEPARTURE AIRPORT"""

        E_prime.append(cancelled_flight_arrival_airport)
        E_prime_d.append(cancelled_flight_departure_airport)

        flight1=get_direct_flights(E_prime_d,E_prime,reduced_flights)
        flight2=get_1_interconnecting(E_prime_d,E_prime,reduced_flights)
        flight3=get_2_interconnecting(E_prime_d,E_prime,reduced_flights)
        reduced_data = pd.concat([flight1, flight2, flight3], ignore_index=True)
        reduced_data = reduced_data.drop_duplicates()

        # print(cancelled_flight_arrival_airport)
        # print(cancelled_flight_departure_airport)

        # print(len(flight2))
        # print(len(flight3))

        # print(reduced_flights[["DepartureAirport","ArrivalAirport"]])

        # """CHECKING IF THE REDUCED DATA HAS DIRECT FLIGHTS OR NOT"""

        # print(reduced_data[["DepartureAirport","ArrivalAirport"]])

        """BEGINNING TO ENCODE THE UNIQUE VALUES FOR EACH INDEX NUMERICALLY."""

        departure_airports = reduced_data["DepartureAirport"].unique()
        arrival_airports = reduced_data["ArrivalAirport"].unique()
        unique_airports =np.concatenate((departure_airports, arrival_airports))
        cancelled_flight_airports=np.array([cancelled_flight_arrival_airport,cancelled_flight_departure_airport])
        unique_airports = np.concatenate((unique_airports, cancelled_flight_airports))
        unique_airports = [code.strip() for code in unique_airports]
        # print(unique_airports)
        unique_airports=np.unique(unique_airports)
        # print(unique_airports)
        flight_no=reduced_data["FlightNumber"].unique()
        departure_time = pd.to_datetime(reduced_data["DepartureDateTime"]).unique()
        departure_time = sorted(departure_time)

        # """BELOW ARE ALL POSSIBLE ARRIVAL AND DEPARTURE AIRPORTS(INCLUDING INTERMEDIATE NODES AS WELL)"""

        # print(departure_airports)
        # print(arrival_airports)

        """NOW CREATING LIST OF ACTUAL STARTING AIRPORTS AND ACTUAL DEPARTURE AIRPORTS FOR THE WHOLE PATH WHOSE FLIGHTS ARE AVAILABLE"""

        final_dep_airports = list(set(E_prime_d) & set(departure_airports))
        final_arr_airports = list(set(E_prime) & set(arrival_airports))

        depart_no=arriv_no=len(unique_airports)
        flght_no=len(flight_no)
        time_no=len(departure_time)

        # print(depart_no,flght_no,time_no)

        

        """GETTING THE ENCODED VALUES OF ALL THE POSSIBLE ARRIVAL AIRPORTS AND DEPARTURE AIRPORTS"""

        flight_exist,departure_time_tensor,arrival_time_tensor,arrival_airport_dict,departure_airport_dict=encode_flights(reduced_data,unique_airports,flight_no,departure_time)
        arrival_airports_encoded_indices=[]
        departure_airports_encoded_indices=[]
        for i in final_arr_airports:
            if i in arrival_airport_dict.keys():
                arrival_airports_encoded_indices.append(arrival_airport_dict[i])
        # print(arrival_airports_encoded_indices)
        for i in final_dep_airports:
            if i in departure_airport_dict.keys():
                departure_airports_encoded_indices.append(departure_airport_dict[i])
        # print(departure_airports_encoded_indices)

        # #print(flight_exist)
        # print(departure_time_tensor)
        # print(arrival_time_tensor)

        """HERE, 'n' REPRESENTS THE MAXIMUM ALLOWED FLIGHTS IN A PATH.

        'r' REPRESENTS THE MINIMUM CONNECT TIME(IN HOURS) ALLOWED BETWEEN TWO CONSECUTIVE FLIGHTS.

        'm' REPRESENTS THE MAXIMUM CONNECT TIME(IN HOURS) ALLOWED BETWEEN TWO CONSECUTIVE FLIGHTS.

        'st' REPRESENTS DEPARTURE TIME OF THE CANCELLED FLIGHT.

        'Et' REPRESENTS ARRIVAL TIME OF THE CANCELLED FLIGHT.
        """

        start_node = np.argwhere(unique_airports== cancelled_flight_departure_airport).flatten()[0]
        end_node = np.argwhere(unique_airports== cancelled_flight_arrival_airport).flatten()[0]
        st=cancelled_flight["DepartureDateTime"]
        et=cancelled_flight["ArrivalDateTime"]
        n_ = 3
        r = 0.2
        m_ =2

        # print(start_node)
        # print(end_node)
        # print(st)
        # print(et)

        mdl = Model("docplex model")
        variables = {(i, j, k, l): mdl.binary_var(name=f'q({i},{j},{k},{l})')
                    for i in range(depart_no)
                    for j in range(arriv_no)
                    for k in range(flght_no)
                    for l in range(time_no) if flight_exist[i, j, k, l] == 1}

        """DEFINING LAGRANGE MULTIPLIERS FOR DIFFERENT CONSTRAINTS"""

        scm=1000
        ecm=1000
        stcm=1
        etcm=1
        ocm=1
        cfm=500
        tfm=1000
        sqrt_scm = np.sqrt(scm)
        sqrt_ecm = np.sqrt(ecm)
        sqrt_cfm = np.sqrt(cfm)
        sqrt_ocm = np.sqrt(ocm)
        total_seconds_per_hour = 3600

        var_vec = np.array(variables.values())
        print(var_vec)

        """WE DEFINED THE NECESSARY CONSTRAINTS AND AN OBJECTIVE FUNCTION THAT MINIMIZES THE DIFFERENCE BETWEEN THE ACTUAL ARRIVAL TIME AND ALTERNATE ARRIVAL TIME AND SIMILARLY MINIMIZES THE DIFFERENCE BETWEEN ACTUAL AND ALTERNATE DEPARTURE TI ME .

        THE CONSTRAINTS ENSURE THAT THE FLIGHTS FORM A PATH FROM THE START NODE TO END NODE(CAN BE MADE MULTIPLE END NODES); ALSO ENSURE THAT CONSECUTIVE FLIGHTS ARE ACTUALLY SORTED BY TIME(OVERLAY CONSTRAINTS) IF THE PATH HAS ATLEAST ONE INTERCONNECTING NODE.
        """

        total_flight_after_reduction=len(flight1)+len(flight2)+len(flight3)
        if(total_flight_after_reduction>0):
            start_constraint = mdl.sum(variables[i,j, k, l]*sqrt_scm
                                    for i in departure_airports_encoded_indices
                                    for j in range(arriv_no)
                                    for k in range(flght_no)
                                    for l in range(time_no) if flight_exist[i,j, k, l]==1)  # For example, sum equals 1
            mdl.add_constraint(start_constraint == 1*math.sqrt(scm), 'start_constraint')
            end_constraint = mdl.sum(variables[i, j, k, l]*sqrt_ecm
                                    for i in range(depart_no)
                                for j in arrival_airports_encoded_indices
                                for k in range(flght_no)
                                for l in range(time_no) if flight_exist[i, j, k, l]==1)
            mdl.add_constraint(end_constraint == 1*math.sqrt(ecm) ,'end_constraint')

            start_time_constraint = mdl.sum(stcm*variables[i, j, k, l] * ((departure_time_tensor[i, j, k, l] - st).total_seconds() / 3600)
                                for i in departure_airports_encoded_indices
                                for j in range(arriv_no)
                                for k in range(flght_no)
                                for l in range(time_no) if flight_exist[i, j, k, l]==1)
            
            end_time_constraint = mdl.sum(etcm*variables[i, j, k, l] * ((arrival_time_tensor[i, j, k, l] - et).total_seconds() / 3600)
                                for i in range(arriv_no)
                                for j in arrival_airports_encoded_indices
                                for k in range(flght_no)
                                for l in range(time_no) if flight_exist[i, j, k, l]==1)
            
            mdl.minimize(start_time_constraint + end_time_constraint)

            if((len(flight2)+len(flight3))>0):
                for j in range(arriv_no):
                    if j not in departure_airports_encoded_indices and j not in arrival_airports_encoded_indices:
                        connecting_flight = mdl.sum(math.sqrt(cfm)*variables[o, j, k, l] - math.sqrt(cfm)*variables[j, p, m, n]
                                                for p in range(arriv_no)
                                                for o in range(depart_no) if o != start_node

                                                for k in range(flght_no)
                                                for l in range(time_no)
                                                for m in range(flght_no)
                                                for n in range(time_no) if flight_exist[o,j, k, l]*flight_exist[j,p,m,n]==1)
                        mdl.add_constraint(connecting_flight == 0, f'connecting_constraint{j}')

                for i in range(depart_no):

                    overlay_constraint_expr_lb = mdl.sum(math.sqrt(ocm)*variables[i, j, k, l] * variables[m, i, n, o] * ((departure_time_tensor[i, j, k, l] -
                                                    arrival_time_tensor[m, i, n, o]).total_seconds() / 3600 - r * math.sqrt(ocm) )
                                                    for j in range(arriv_no)
                                                    for m in range(depart_no)
                                                    for k in range(flght_no)
                                                    for l in range(time_no)
                                                    for n in range(flght_no)
                                                    for o in range(time_no) if flight_exist[i,j, k, l]*flight_exist[m,i,n,o]==1)
                    overlay_constraint_expr_ub = mdl.sum(math.sqrt(ocm)*variables[i, j, k, l] * variables[m, i, n, o] * ((departure_time_tensor[i, j, k, l] -
                                                    arrival_time_tensor[m, i, n, o]).total_seconds() / 3600 - m * math.sqrt(ocm) )
                                                    for j in range(arriv_no)
                                                    for m in range(depart_no)
                                                    for k in range(flght_no)
                                                    for l in range(time_no)
                                                    for n in range(flght_no)
                                                    for o in range(time_no) if flight_exist[i,j, k, l]*flight_exist[m,i,n,o]==1)

                    mdl.add_constraint(overlay_constraint_expr_lb >= 0, f'overlay_constraint_lb{i}')
                    mdl.add_constraint(overlay_constraint_expr_ub <= 0, f'overlay_constraint_ub{i}')

            total_flight = mdl.sum(variables[i,j, k, l]*math.sqrt(tfm)
                                    for i in range(depart_no)
                                    for j in range(arriv_no)
                                    for k in range(flght_no)
                                    for l in range(time_no) if flight_exist[i,j,k,l]==1)
            mdl.add_constraint(total_flight <= n_ * math.sqrt(tfm), 'total_flight_constraint')
        else:
            print("No Alternate Flight Available")

        if(total_flight_after_reduction>0):
            mod = from_docplex_mp(mdl)
            # print(mod.prettyprint())
        else:
            raise AssertionError("No Alternate Flight Available")
        
        """NOW WE DEFINE A CONSTRAINT QUADRATIC MODEL AND USE D-WAVE'S HYBRID SOLVER TO SAMPLE SOLUTIONS."""

        

        # Transfer the objective function
        obj_linear = mod.objective.linear.to_dict()
        obj_quadratic = mod.objective.quadratic.to_dict()
        objective_bqm = BinaryQuadraticModel(obj_linear, obj_quadratic, mod.objective.constant, vartype=BINARY)

        #Initialize the Constrained Quadratic Model
        cqm = ConstrainedQuadraticModel().from_bqm(objective_bqm)

        # Transfer the linear constraints
        for constraint in mod.linear_constraints:
            linear_terms = {mod.get_variable(i).name: coeff for i, coeff in enumerate(constraint.linear.to_array())}
            linear_bqm = BinaryQuadraticModel(linear_terms, {}, 0.0, vartype=BINARY)

            if constraint.sense == constraint.sense.LE:
                cqm.add_constraint(linear_bqm <= constraint.rhs, label=constraint.name)
            elif constraint.sense == constraint.sense.GE:
                cqm.add_constraint(linear_bqm >= constraint.rhs, label=constraint.name)
            else:
                cqm.add_constraint(linear_bqm == constraint.rhs, label=constraint.name)

        # Transfer the quadratic constraints
        for constraint in mod.quadratic_constraints:
            linear_terms = {mod.get_variable(i).name: coeff for i, coeff in enumerate(constraint.linear.to_array())}
            quadratic_terms = {(mod.get_variable(i).name, mod.get_variable(j).name): coeff
                            for (i, j), coeff in constraint.quadratic.to_dict().items()}
            quadratic_bqm = BinaryQuadraticModel(linear_terms, quadratic_terms, 0.0, vartype=BINARY)

            if constraint.sense == constraint.sense.LE:
                cqm.add_constraint(quadratic_bqm <= constraint.rhs, label=constraint.name)
            elif constraint.sense == constraint.sense.GE:
                cqm.add_constraint(quadratic_bqm >= constraint.rhs, label=constraint.name)
            else:
                cqm.add_constraint(quadratic_bqm == constraint.rhs, label=constraint.name)


        """PRINTING THE TOP N UNIQUE SOLUTIONS."""

        num_top_solutions = 5  # Set the number of top unique solutions to print

        if total_flight_after_reduction > 0:
            # Initialize the Leap Hybrid CQM Sampler
            if method == 'SIMULATE':
                bqm = cqm_to_bqm(cqm)[0] # Will only work for linear constraints for non linear use LeapHybridCQMSampler
                sampler = SimulatedAnnealingSampler()
                print("Submitting to simulated annealing solver")
                result = sampler.sample(
                    bqm, 
                    num_reads=1000, 
                    # num_sweeps=num_sweeps,
                    # beta_range=beta_range
                )
                result = result.filter(lambda row: cqm.check_feasible(row.sample))  # Filter for feasible solutions
            else:
                sampler = LeapHybridCQMSampler(token=TOKEN)
                print("Submitting to Leap Hybrid CQM Sampler")
                result = sampler.sample_cqm(cqm)
                result = result.filter(lambda row: row.is_feasible)  # Filter for feasible solutions

            try:
                unique_solutions = {}
                for sample, energy, num_occurrences in result.data(['sample', 'energy', 'num_occurrences']):
                    # 1) Keep only the real variables
                    real_sample = {
                        k: v
                        for k, v in sample.items()
                        if not (isinstance(k, str) and k.startswith('slack_'))
                    }

                    # 2) Build a hashable repr by sorting on the string of the key
                    sample_repr = tuple(
                        sorted(
                            real_sample.items(),
                            key=lambda kv: str(kv[0])
                        )
                    )

                    # 3) Deduplicate
                    if sample_repr not in unique_solutions:
                        unique_solutions[sample_repr] = (energy, num_occurrences)
                        
            # Sort unique solutions by the number of occurrences (descending)
                sorted_unique_solutions = sorted(unique_solutions.items(), key=lambda x: x[1][1], reverse=True)

            # Print the top unique solutions or no feasible solutions if there are none
                if sorted_unique_solutions:
                    print(f"Top {num_top_solutions} unique solutions:")
                    for i, (sample_repr, (energy, num_occurrences)) in enumerate(sorted_unique_solutions[:num_top_solutions]):
                        print(f"\nUnique Solution {i+1}:")
                        print("Energy:", energy, "Occurrences:", num_occurrences)
                        print("Sample:", dict(sample_repr))
                else:
                #HERE WE NEED TO
                    print("NO FEASIBLE SOLUTIONS")
                    continue

            except Exception as e:
                print("An error occurred while solving the CQM:", e)
        else:
            print("No flights to process.")



        flight_solution = []
        for i, (config_tuple, (energy, num_occurrences)) in enumerate(sorted_unique_solutions[:num_top_solutions]):
            selected_tuples = []
            for key, val in config_tuple:  # unpack the (key, value) pairs
                if val != 1.0:
                    continue
                # key might already be a tuple, or a string repr of a tuple
                if isinstance(key, tuple):
                    indices = key
                else:
                    try:
                        indices = ast.literal_eval(key)
                        if not isinstance(indices, tuple):
                            # skip any weird non‐tuple keys
                            continue
                    except (ValueError, SyntaxError):
                        continue
                selected_tuples.append(indices)

            solution = decode(selected_tuples, unique_airports, flight_no, departure_time, reduced_data)
            flight_solution.append(solution)

        print(flight_solution)

        scorer = PathScoring(scoring_criteria_Flights_toggle, scoring_criteria_Flights)
        paths=flight_solution
        df=reduced_data
        alpha = []
        for i in range(len(paths)):
            alpha.append(scorer.calc_score(df, paths[i], cancelled_flight)/(len(paths[i]))**2)
            for j in range(len(paths[i])):
                paths[i][j] = Flight(df[df["InventoryId"]==paths[i][j]][['FC_AvailableInventory', 'BC_AvailableInventory', 'PC_AvailableInventory', 'EC_AvailableInventory']], paths[i][j], df[df["InventoryId"]==paths[i][j]]['DepartureAirport'].iloc[0], df[df["InventoryId"]==paths[i][j]]['ArrivalAirport'].iloc[0])

        alpha = np.array(alpha)
        alpha = alpha/np.max(alpha)
        
        PNR_list = pd.read_csv(PNR_FILE)
        passenger_details = pd.read_csv(PASSENGER_LIST)        
        PNR_scorer = impacted_PNR(scoring_criteria_PNRs, PNR_list, passenger_details, scoring_criteria_PNRs_toggle, inventory_dataframe, [disrupt])
        impacted_pax, PNRs, matrix_solved = PNR_scorer.solve()
        
        Passengers_flight_ungrouped = impacted_pax[disrupt]
        PNR = []
        PNRs = PNRs[disrupt]
        scores = matrix_solved[disrupt]
        row_index_list = PNRs.index.tolist()
        for i in range(len(PNRs)):
            PNR.append(Passenger(int(PNRs['PAX_CNT'].iloc[i]), row_index_list[i]))


        src=cancelled_flight_departure_airport
        dest=cancelled_flight_arrival_airport
        sampleset = reaccomodation(PNR, paths, scores, alpha, src, dest, Passengers_flight_ungrouped, disrupt, TOKEN, "Quantum", method=method)

        if sampleset is not None and sampleset.first.energy<0:
            df1 = pd.read_csv(os.path.join(moduleDir, "Solutions", "Quantum", f"Default_solution_{disrupt}.csv"))
            df2 = pd.read_csv(os.path.join(moduleDir, "Solutions", "Quantum", f"Exception_list_{disrupt}.csv"))

            for i in range(len(df1)):
                flight_id = df1["Flight ID"][i]
                PNR_ID = df1["PNR ID"][i]
                passenger_class = df1["Class"][i]
                inventory_id_condition = inventory_dataframe["InventoryId"] == flight_id

                if passenger_class == 'BC':
                    inventory_dataframe.loc[inventory_id_condition, "BC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]
                elif passenger_class == 'FC':
                    inventory_dataframe.loc[inventory_id_condition, "FC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]
                elif passenger_class == 'PC':
                    inventory_dataframe.loc[inventory_id_condition, "PC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]
                else:
                    inventory_dataframe.loc[inventory_id_condition, "EC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]

            for i in range(len(df2)):
                flight_id = df2["Flight ID"][i]  
                PNR_ID = df2["PNR ID"][i]
                passenger_class = df2["Class"][i]
                inventory_id_condition = inventory_dataframe["InventoryId"] == flight_id

                if passenger_class == 'BC':
                    inventory_dataframe.loc[inventory_id_condition, "BC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]
                elif passenger_class == 'FC':
                    inventory_dataframe.loc[inventory_id_condition, "FC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]
                elif passenger_class == 'PC':
                    inventory_dataframe.loc[inventory_id_condition, "PC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]
                else:
                    inventory_dataframe.loc[inventory_id_condition, "EC_AvailableInventory"] -= PNRs['PAX_CNT'].loc[PNR_ID]

if __name__ == '__main__':
    main("INV-ZZ-1409214", TOKEN='DEV-12b7e5b3bee7351638023f6bf954329397740cbe')

# if __name__ == '__main__':
#     import random
#     import time
#     INVENTORY_FILE = os.path.join(moduleDir, "Files", "inv.csv")
#     flight_network = pd.read_csv(INVENTORY_FILE)
#     inventory_ids = flight_network['InventoryId'].tolist()
    
#     num_sims = 150
#     results_df = pd.DataFrame(columns=[
#         'Simulation', 'Disruption', 'Num_Default_PNRs',
#         'Num_Exception_NonNull_PNRs', 'Num_Exception_Null_PNRs',
#         'Total_PNRs', 'Percentage_Default',
#         'Percentage_Exception_NonNull', 'Percentage_Exception_Null',
#         'Percentage_Solved', 'Time_Taken'
#     ])
#     results_path = os.path.join(moduleDir, "Simulation_Results_Quantum.csv")
#     for i, disruption in enumerate(inventory_ids):
#         try:
#             start = time.time()
#             main(disruption, TOKEN='DEV-12b7e5b3bee7351638023f6bf954329397740cbe')
#             end = time.time()
#             print(f"Time taken for simulation {i+1}: {end - start:.2f} seconds")

#             solution_path = os.path.join(moduleDir, "Solutions", "Quantum", f"Default_solution_{disruption}.csv")
#             exception_path = os.path.join(moduleDir, "Solutions", "Quantum", f"Exception_list_{disruption}.csv")
            
#             default_solutions = pd.read_csv(solution_path)        
#             exception_pnrs = pd.read_csv(exception_path)
            
#             num_default_pnrs = default_solutions['PNR ID'].nunique()
#             num_exception_pnrs = exception_pnrs[exception_pnrs['Path'].notnull()]['PNR ID'].nunique()
#             num_null_exception_pnrs = exception_pnrs[exception_pnrs['Path'].isnull()]['PNR ID'].nunique()
#             total_pnrs = num_default_pnrs + num_exception_pnrs + num_null_exception_pnrs
#             percentage_default = (num_default_pnrs / total_pnrs) * 100 if total_pnrs > 0 else 0
#             percentage_exception = (num_exception_pnrs / total_pnrs) * 100 if total_pnrs > 0 else 0
#             percentage_null_exception = (num_null_exception_pnrs / total_pnrs) * 100 if total_pnrs > 0 else 0
#             percentage_solved = (num_default_pnrs + num_exception_pnrs) / total_pnrs * 100 if total_pnrs > 0 else 0

#             new_row = pd.DataFrame([{
#                 'Simulation': i + 1,
#                 'Disruption': disruption,
#                 'Num_Default_PNRs': num_default_pnrs,
#                 'Num_Exception_NonNull_PNRs': num_exception_pnrs,
#                 'Num_Exception_Null_PNRs': num_null_exception_pnrs,
#                 'Total_PNRs': total_pnrs,
#                 'Percentage_Default': percentage_default,
#                 'Percentage_Exception_NonNull': percentage_exception,
#                 'Percentage_Exception_Null': percentage_null_exception,
#                 'Percentage_Solved': percentage_solved,
#                 "Time_Taken": end - start
#             }])
#             results_df = pd.concat([results_df, new_row], ignore_index=True)
#             # Save after each simulation
#             results_df.to_csv(results_path, index=False)

#             print(f"Simulation {i+1}:")
#             print(f"  Disruption: {disruption}")
#             print(f"  Number of PNRs in Default Solution: {num_default_pnrs}")
#             print(f"  Number of PNRs in Exception List with Non Null Path: {num_exception_pnrs}")
#             print(f"  Number of PNRs in Exception List with Null Path: {num_null_exception_pnrs}")
#             print(f"  Total PNRs: {total_pnrs}")
#             print(f"  Percentage of PNRs assigned to Default Solution: {percentage_default:.2f}%")
#             print(f"  Percentage of PNRs assigned to Exception List with Non Null Path: {percentage_exception:.2f}%")
#             print(f"  Percentage of PNRs assigned to Exception List with Null Path: {percentage_null_exception:.2f}%")
#             print(f"  Percentage of PNRs solved (Default + Exception): {percentage_solved:.2f}%")
#             print(f"  Time taken: {end - start:.2f} seconds")
#             print("-" * 40)
#         except Exception as e:
#             print(f"An error occurred during simulation {i+1}: {e}")
#             continue

#     print("Simulation results saved to 'Simulation_Results_Quantum.csv'.")