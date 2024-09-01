from pnr_reaccomodation import *
from classical_pathfinding import * 

def check_time_diff(flight_network, disrupted_flights):
    current_net = flight_network[flight_network['InventoryId'].isin(disrupted_flights)].copy()
    current_net.sort_values(by="DepartureDateTime", inplace=True)

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
    
def main(*disruptions_all, INVENTORY_FILE="INV-ZZ-20231208_041852.csv", PNR_FILE = "PNRB-ZZ-20231208_062017.csv", PASSENGER_LIST = "PNRP-ZZ-20231208_111136.csv", TOKEN = 'DEV-6ddf205adb6761bc0018a65f2496245457fe977f'):
    flight_network = pd.read_csv(INVENTORY_FILE)
    PNR_list = pd.read_csv(PNR_FILE)
    passenger_details = pd.read_csv(PASSENGER_LIST)
    disruptions_all = check_time_diff(flight_network, list(disruptions_all))
    
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
            
            sampleset =  reaccomodation(PNR, paths, scores, alpha, sources[disrupt], destinations[disrupt], impacted_pax[disrupt], disrupt, TOKEN)
        
            if sampleset is not None and sampleset.first.energy<0:
                df1 = pd.read_csv(f"Default_solution_{disrupt}.csv")
                df2 = pd.read_csv(f"Exception_list_{disrupt}.csv")

                for i in range(len(df1)):
                    flight_id = df1["Flight ID"][i]
                    PNR_ID = df1["PNR ID"][i]
                    inventory_id_condition = flight_network["InventoryId"] == flight_id

                    if flight_id == 'BC':
                        flight_network.loc[inventory_id_condition, "BC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif flight_id == 'FC':
                        flight_network.loc[inventory_id_condition, "FC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif flight_id == 'PC':
                        flight_network.loc[inventory_id_condition, "PC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    else:
                        flight_network.loc[inventory_id_condition, "EC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]

                for i in range(len(df2)):
                    flight_id = df2["Flight ID"][i]  
                    PNR_ID = df2["PNR ID"][i]
                    inventory_id_condition = flight_network["InventoryId"] == flight_id

                    if flight_id == 'BC':
                        flight_network.loc[inventory_id_condition, "BC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif flight_id == 'FC':
                        flight_network.loc[inventory_id_condition, "FC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    elif flight_id == 'PC':
                        flight_network.loc[inventory_id_condition, "PC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                    else:
                        flight_network.loc[inventory_id_condition, "EC_AvailableInventory"] -= Passengers_flight['PAX_CNT'].loc[PNR_ID]
                        
        
if __name__ == '__main__':
    main("INV-ZZ-3174758", TOKEN='DEV-12b7e5b3bee7351638023f6bf954329397740cbe')