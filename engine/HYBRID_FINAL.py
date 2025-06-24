from pnr_reaccomodation import *
from classical_pathfinding import * 
import os

METHOD = 'SIMULATE'
moduleDir = os.path.dirname(os.path.abspath(__file__))
def check_time_diff(flight_network, disrupted_flights):
    current_net = flight_network[flight_network['InventoryId'].isin(disrupted_flights)].copy()
    current_net.sort_values(by="DepartureDateTime", inplace=True)
    
    # Convert DataFrame column to a set of IDs
    df_ids = set(current_net['InventoryId'])

    # Find IDs that are in the list but not in the DataFrame
    invalid_ids = [id_ for id_ in disrupted_flights if id_ not in df_ids]
    assert len(invalid_ids) == 0, f"The flight IDs {invalid_ids} are not valid."
    
    final = []
    while True:
        if len(current_net) == 0: break
        if len(current_net) == 1: 
            final.append([current_net.iloc[0,:]['InventoryId']]) 
            break
        current_batch = []
        lost_batch = []
        prev_row = current_net.iloc[0,:]
        for i in range(1, len(current_net)):
            current_row = current_net.iloc[i,:]
            curr_time_diff = (datetime.strptime(current_row['DepartureDateTime'], "%Y-%m-%d %H:%M:%S") - datetime.strptime(prev_row['DepartureDateTime'], "%Y-%m-%d %H:%M:%S")).total_seconds()/3600
            if curr_time_diff>72:
                if len(current_batch) == 0:
                    current_batch.append(prev_row['InventoryId']) 
                current_batch.append(current_row['InventoryId'])
                prev_row = current_net.iloc[i,:]
            else:
                lost_batch.append(current_row['InventoryId'])
        current_net = current_net[current_net['InventoryId'].isin(lost_batch)]
        if len(current_batch)>=1:final.append(current_batch)
        
    return final
     
def main(*disruptions_all, INVENTORY_FILE=os.path.join(moduleDir, "Files", "inv.csv"), PNR_FILE = os.path.join(moduleDir, "Files", "pnrb.csv"), PASSENGER_LIST = os.path.join(moduleDir, "Files", "pnrp.csv"), TOKEN = 'DEV-6ddf205adb6761bc0018a65f2496245457fe977f'):
    flight_network = pd.read_csv(INVENTORY_FILE)
    PNR_list = pd.read_csv(PNR_FILE)
    passenger_details = pd.read_csv(PASSENGER_LIST)
    disruptions_all = check_time_diff(flight_network, list(disruptions_all))
    
    defaults = []
    exceptions = []
    for disruptions in disruptions_all:
        # Create Object and run solve function
        solver = pathfind_recursion(flight_network, disruptions, scoring_criteria= scoring_criteria_Flights, toggle = scoring_criteria_Flights_toggle, verbose= 0, stopovers= 2)
        solutions, alphas, sources, destinations, _ = solver.solve()
        
        # Create Object and run solve function for PNRs
        PNR_list = impacted_PNR(scoring_criteria_PNRs, PNR_list, passenger_details, scoring_criteria_PNRs_toggle, flight_network, disruptions)
        impacted_pax, PNRs, matrix_solved = PNR_list.solve()
        
        for disrupt in disruptions:
            print(f"Solving for disruption of {disrupt} inventory ID flight")
            paths = solutions[disrupt]
            print(f"Paths for {disrupt}:", paths)
            alpha = alphas[disrupt]
            Passengers_flight = PNRs[disrupt]
            scores = matrix_solved[disrupt]
            abs_alpha = []
            
            PNR = []
            row_index_list = Passengers_flight.index.tolist()
            for i in range(len(Passengers_flight)):
                PNR.append(Passenger(int(Passengers_flight['PAX_CNT'].iloc[i]), row_index_list[i]))
            
            for i in range(len(paths)):
                abs_alpha.append(paths[i][-1])
                paths[i] = paths[i][:-1]
                for j in range(len(paths[i])):
                    paths[i][j] = Flight(flight_network[flight_network["InventoryId"]==paths[i][j]][['FC_AvailableInventory', 'BC_AvailableInventory', 'PC_AvailableInventory', 'EC_AvailableInventory']], paths[i][j], flight_network[flight_network["InventoryId"]==paths[i][j]]['DepartureAirport'].iloc[0], flight_network[flight_network["InventoryId"]==paths[i][j]]['ArrivalAirport'].iloc[0])
            
            sampleset =  reaccomodation(PNR, paths, scores, alpha, sources[disrupt], destinations[disrupt], impacted_pax[disrupt], disrupt, TOKEN, "Hybrid", method=METHOD)
        
            if sampleset is not None and sampleset.first.energy<0:
                default_path = os.path.join(moduleDir, "Solutions", "Hybrid", f"Default_solution_{disrupt}.csv")
                except_path = os.path.join(moduleDir, "Solutions", "Hybrid", f"Exception_list_{disrupt}.csv")
                defaults.append(default_path)
                exceptions.append(except_path)
                df1 = pd.read_csv(default_path)
                df2 = pd.read_csv(except_path)

                for i in range(len(df1)):
                    flight_id = df1["Flight ID"][i]
                    PNR_ID = df1["PNR ID"][i]
                    passenger_class = df1["Class"][i]
                    inventory_id_condition = flight_network["InventoryId"] == flight_id

                    if passenger_class == 'BC':
                        flight_network.loc[inventory_id_condition, "BC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif passenger_class == 'FC':
                        flight_network.loc[inventory_id_condition, "FC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif passenger_class == 'PC':
                        flight_network.loc[inventory_id_condition, "PC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    else:
                        flight_network.loc[inventory_id_condition, "EC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]

                for i in range(len(df2)):
                    flight_id = df2["Flight ID"][i]  
                    PNR_ID = df2["PNR ID"][i]
                    passenger_class = df2["Class"][i]
                    inventory_id_condition = flight_network["InventoryId"] == flight_id

                    if passenger_class == 'BC':
                        flight_network.loc[inventory_id_condition, "BC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif passenger_class == 'FC':
                        flight_network.loc[inventory_id_condition, "FC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif passenger_class == 'PC':
                        flight_network.loc[inventory_id_condition, "PC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    else:
                        flight_network.loc[inventory_id_condition, "EC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
    
    return [defaults, exceptions]
        
# if __name__ == '__main__':
#     main("INV-ZZ-3217115", TOKEN='DEV-12b7e5b3bee7351638023f6bf954329397740cbe')

if __name__ == '__main__':
    import random
    import time
    INVENTORY_FILE = os.path.join(moduleDir, "Files", "inv.csv")
    flight_network = pd.read_csv(INVENTORY_FILE)
    inventory_ids = flight_network['InventoryId'].tolist()
    
    results_df = pd.DataFrame(columns=[
        'Simulation', 'Disruption', 'Num_Default_PNRs',
        'Num_Exception_NonNull_PNRs', 'Num_Exception_Null_PNRs',
        'Total_PNRs', 'Percentage_Default',
        'Percentage_Exception_NonNull', 'Percentage_Exception_Null',
        'Percentage_Solved', 'Time_Taken'
    ])
    results_path = os.path.join(moduleDir, "Simulation_Results_Hybrid.csv")
    for i, disruption in enumerate(inventory_ids[850:]):
        try:
            start = time.time()
            main(disruption, TOKEN='DEV-12b7e5b3bee7351638023f6bf954329397740cbe')
            end = time.time()
            print(f"Time taken for simulation {i+1}: {end - start:.2f} seconds")

            solution_path = os.path.join(moduleDir, "Solutions", "Hybrid", f"Default_solution_{disruption}.csv")
            exception_path = os.path.join(moduleDir, "Solutions", "Hybrid", f"Exception_list_{disruption}.csv")
            
            default_solutions = pd.read_csv(solution_path)        
            exception_pnrs = pd.read_csv(exception_path)
            
            num_default_pnrs = default_solutions['PNR ID'].nunique()
            num_exception_pnrs = exception_pnrs[exception_pnrs['Path'].notnull()]['PNR ID'].nunique()
            num_null_exception_pnrs = exception_pnrs[exception_pnrs['Path'].isnull()]['PNR ID'].nunique()
            total_pnrs = num_default_pnrs + num_exception_pnrs + num_null_exception_pnrs
            percentage_default = (num_default_pnrs / total_pnrs) * 100 if total_pnrs > 0 else 0
            percentage_exception = (num_exception_pnrs / total_pnrs) * 100 if total_pnrs > 0 else 0
            percentage_null_exception = (num_null_exception_pnrs / total_pnrs) * 100 if total_pnrs > 0 else 0
            percentage_solved = (num_default_pnrs + num_exception_pnrs) / total_pnrs * 100 if total_pnrs > 0 else 0

            new_row = pd.DataFrame([{
                'Simulation': i + 1,
                'Disruption': disruption,
                'Num_Default_PNRs': num_default_pnrs,
                'Num_Exception_NonNull_PNRs': num_exception_pnrs,
                'Num_Exception_Null_PNRs': num_null_exception_pnrs,
                'Total_PNRs': total_pnrs,
                'Percentage_Default': percentage_default,
                'Percentage_Exception_NonNull': percentage_exception,
                'Percentage_Exception_Null': percentage_null_exception,
                'Percentage_Solved': percentage_solved,
                "Time_Taken": end - start
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            # Save after each simulation
            results_df.to_csv(results_path, index=False)

            print(f"Simulation {i+1}:")
            print(f"  Disruption: {disruption}")
            print(f"  Number of PNRs in Default Solution: {num_default_pnrs}")
            print(f"  Number of PNRs in Exception List with Non Null Path: {num_exception_pnrs}")
            print(f"  Number of PNRs in Exception List with Null Path: {num_null_exception_pnrs}")
            print(f"  Total PNRs: {total_pnrs}")
            print(f"  Percentage of PNRs assigned to Default Solution: {percentage_default:.2f}%")
            print(f"  Percentage of PNRs assigned to Exception List with Non Null Path: {percentage_exception:.2f}%")
            print(f"  Percentage of PNRs assigned to Exception List with Null Path: {percentage_null_exception:.2f}%")
            print(f"  Percentage of PNRs solved (Default + Exception): {percentage_solved:.2f}%")
            print(f"  Time taken: {end - start:.2f} seconds")
            print("-" * 40)
        except Exception as e:
            print(f"An error occurred during simulation {i+1}: {e}")
            continue

    print("Simulation results saved to 'Simulation_Results_Hybrid.csv'.")