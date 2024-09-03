from datetime import datetime, timedelta
import time
import pandas as pd
import os

class PathScoring:
    def __init__(self, toggle, scoring_criterion):
        self.toggle = toggle
        self.scoring_criteria = scoring_criterion
            
    def calc_score(self, df, path_detailed, disrupted_flight_data):
        score = 0
        arrival_delay = (datetime.strptime(df[df["InventoryId"]==path_detailed[-1]]['ArrivalDateTime'].iloc[0].strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") - datetime.strptime(disrupted_flight_data['ArrivalDateTime'].strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")).total_seconds()/3600
        if arrival_delay < 6 and self.toggle['Arr_LT_6hrs']:
            score += self.scoring_criteria['Arr_LT_6hrs']
        elif arrival_delay < 12 and self.toggle['Arr_LT_12hrs']:
            score += self.scoring_criteria['Arr_LT_12hrs']
        elif arrival_delay < 24 and self.toggle['Arr_LT_24hrs']:
            score += self.scoring_criteria['Arr_LT_24hrs']
        elif arrival_delay < 48 and self.toggle['Arr_LT_48hrs']:
            score += self.scoring_criteria['Arr_LT_48hrs']
        
        
        departure_delay = (datetime.strptime(df[df["InventoryId"]==path_detailed[0]]['DepartureDateTime'].iloc[0].strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") - datetime.strptime(disrupted_flight_data['DepartureDateTime'].strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")).total_seconds()/3600
        if departure_delay < 6 and self.toggle['SPF_LT_6hrs']:
            score += self.scoring_criteria['SPF_LT_6hrs']
        elif departure_delay < 12 and self.toggle['SPF_LT_12hrs']:
            score += self.scoring_criteria['SPF_LT_12hrs']
        elif departure_delay < 24 and self.toggle['SPF_LT_24hrs']:
            score += self.scoring_criteria['SPF_LT_24hrs']
        elif departure_delay < 48 and self.toggle['SPF_LT_48hrs']:
            score += self.scoring_criteria['SPF_LT_48hrs']

        equipment_check = 1
        for flight in path_detailed:
            if df[df["InventoryId"]==flight]['AircraftType'].iloc[0] != disrupted_flight_data['AircraftType']:
                equipment_check = 0
                break
        if equipment_check and self.toggle['Equipment']: score += self.scoring_criteria['Equipment']
        if len(path_detailed) != 1 and self.toggle['IsStopover']: score += self.scoring_criteria['IsStopover']

        if df[df["InventoryId"]==path_detailed[0]]['ArrivalAirport'].iloc[0] == disrupted_flight_data['ArrivalAirport'] and self.toggle['Same_Citipairs']: score += self.scoring_criteria['Same_Citipairs']
        if df[df["InventoryId"]==path_detailed[0]]['ArrivalAirport'].iloc[0] != disrupted_flight_data['ArrivalAirport'] and self.toggle['Different_Citipairs']: score -= self.scoring_criteria['Different_Citipairs']

        return score
   
class impacted_PNR:
    def __init__(self, scoring_criteria, PNR_list, passenger_details, toggle, flight_network, disruption, verbose = 0):
        self.scoring_criteria = scoring_criteria
        self.PNR_list = PNR_list
        self.passenger_details = passenger_details
        self.flight_network = flight_network
        self.toggle = toggle
        self.disruption = disruption
        self.verbose = verbose

        self.flight_network.sort_values(by=["DepartureDateTime"], inplace = True, kind= "stable")
        self.flight_network = self.flight_network.reset_index().iloc[:,1:]
        pass

    def find_impacted_passengers(self, PNR_list, passenger_details, disrupted_flight_data):
        PNR_list = PNR_list[PNR_list['FLT_NUM'] == disrupted_flight_data["FlightNumber"]]
        dep_date = datetime.strptime(disrupted_flight_data['DepartureDateTime'].strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
        dep_date = dep_date.strftime("%#m/%#d/%#Y %#H:%M")
        PNR_list = PNR_list[PNR_list['DEP_DTML'] == dep_date]
        detailed_PNR = self.PNR_list[self.PNR_list['RECLOC'].isin(list(PNR_list['RECLOC']))]

        passenger_details = passenger_details.groupby('RECLOC')
        passenger_affected = passenger_details.get_group(PNR_list.iloc[0,:]['RECLOC']).copy()
        current_PNR = PNR_list.iloc[0,:]
        pax = current_PNR['PAX_CNT']
        booking_class = current_PNR['COS_CD']
        passenger_affected['COS_CD'] = [booking_class] * len(passenger_affected)
        passenger_affected['PAX_CNT'] = [pax] * len(passenger_affected)
        score = self.calc_score(passenger_affected, current_PNR)
        passenger_affected["Scores"] = [max(score)] * len(score)    

        for i in range(1,len(PNR_list)):
            passengers = passenger_details.get_group(PNR_list.iloc[i,:]['RECLOC']).copy()
            current_PNR = PNR_list.iloc[i,:]
            pax = current_PNR['PAX_CNT']
            booking_class = current_PNR['COS_CD']
            passengers['COS_CD'] = [booking_class] * len(passengers)
            passengers['PAX_CNT'] = [pax] * len(passengers)
            score = self.calc_score(passengers, current_PNR)
            passengers["Scores"] = [max(score)] * len(score)

            passenger_affected = pd.concat([passenger_affected, passengers])
            
        passenger_affected.sort_values(by= ["Scores"], inplace=True, ascending= False)
        passenger_affected = passenger_affected.reset_index().iloc[:,1:]

        return passenger_affected
    
    def calc_score(self, passengers, PNR):
        scores = []
        for i in range(len(passengers)):
            current_score = 0
            current_passenger = passengers.iloc[i,:]
            if type(current_passenger['SSR_CODE_CD1']) == str and self.toggle['SSR']: current_score += self.scoring_criteria['SSR']
            if self.toggle['Per_Pax']: current_score += PNR['PAX_CNT'] * self.scoring_criteria["Per_Pax"]
            if type(current_passenger['TierLevel'] == str):
                if current_passenger['TierLevel'] == "Gold" and self.toggle['Loyalty_Gold']: current_score += self.scoring_criteria["Loyalty_Gold"]
                if current_passenger['TierLevel'] == "Silver" and self.toggle['Loyalty_Silver']: current_score += self.scoring_criteria["Loyalty_Silver"]
                if current_passenger['TierLevel'] == "Platinum" and self.toggle['Loyalty_Platinum']: current_score += self.scoring_criteria["Loyalty_Platinum"]
                if current_passenger['TierLevel'] == "Presidential Platinum" and self.toggle['Loyalty_PPlatinum']: current_score += self.scoring_criteria["Loyalty_PPlatinum"]
            if self.toggle['EconomyClass'] and PNR['COS_CD'] == 'EconomyClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            if self.toggle['BusinessClass'] and PNR['COS_CD'] == 'BusinessClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            if self.toggle['PremiumEconomyClass'] and PNR['COS_CD'] == 'PremiumEconomyClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            if self.toggle['FirstClass'] and PNR['COS_CD'] == 'FirstClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            
            scores.append(current_score)
        return scores
    
    def solver(self, disruption):
        flight_disruption = self.flight_network[self.flight_network["InventoryId"] == disruption].iloc[0,:]
        passengers = self.find_impacted_passengers(self.PNR_list.copy(), self.passenger_details.copy(), flight_disruption)
        PNR_list = passengers.groupby('RECLOC').first()
        PNR_list.sort_values(by= ["Scores"], inplace=True, ascending= False)
        return passengers, PNR_list
    
    def solve(self):
        if len(self.disruption) == 1:
            solution_pax, solution_pnr = self.solver(self.disruption[0])
            mater = self.matrix_finder(solution_pnr)
            return {self.disruption[0]:solution_pax}, {self.disruption[0]:solution_pnr}, {self.disruption[0]:mater}
        else:
            solution_pax = {}
            solution_pnr = {}
            matrix_find = {}
            for disrupt in self.disruption:
                solution_pax[disrupt], solution_pnr[disrupt] = self.solver(disrupt)
                matrix_find[disrupt] = self.matrix_finder(solution_pnr[disrupt])
            return solution_pax, solution_pnr, matrix_find

    def matrix_finder(self, solution_pnr):
        bossman = {}

        for pnr in range(len(solution_pnr)):
            current_pnr = solution_pnr.iloc[pnr,:].copy()
            answer = {"FirstClass":0,
                      "BusinessClass":0,
                      "PremiumEconomyClass":0,
                      "EconomyClass":0,
                      }
            all_class = list(answer.keys())
            class_curr = current_pnr['COS_CD']
            curr_score = current_pnr['Scores']
            answer[current_pnr['COS_CD']] = curr_score
            
            if class_curr == 'FirstClass' and self.toggle['Downgrade_allow']:
                answer[all_class[1]] = curr_score - 100
            if class_curr == 'EconomyClass' and self.toggle['Upgrade_allow']:
                answer[all_class[2]] = curr_score - 100
            if class_curr == 'PremiumEconomyClass' and self.toggle['Upgrade_allow']:
                answer[all_class[1]] = curr_score - 100
            if class_curr == 'PremiumEconomyClass' and self.toggle['Downgrade_allow']:
                answer[all_class[3]] = curr_score - 100
            if class_curr == 'BusinessClass' and self.toggle['Upgrade_allow']:
                answer[all_class[0]] = curr_score - 100
            if class_curr == 'BusinessClass' and self.toggle['Downgrade_allow']:
                answer[all_class[2]] = curr_score - 100

            final = {}
            final["FC"] = answer["FirstClass"]
            final["BC"] = answer["BusinessClass"]
            final["PC"] = answer["PremiumEconomyClass"]
            final["EC"] = answer["EconomyClass"]
            bossman[current_pnr.name] = final
        
        return bossman

def csv_reader_buss(noel):
    scoring = {}
    scoring_toggle = {}
    for i in range(len(noel)):
        current = noel.iloc[i,:]
        scoring[current['Business Rules']] = current['Score']
        scoring_toggle[current['Business Rules']] = current['Enabled']
    return scoring, scoring_toggle

def csv_reader_plane(noel):
    scoring = {}
    scoring_toggle = {}
    for i in range(len(noel)):
        current = noel.iloc[i,:]
        scoring[current['Flight Rules']] = current['Score']
        scoring_toggle[current['Flight Rules']] = current['Enabled']
    return scoring, scoring_toggle

moduleDir = os.path.dirname(os.path.abspath(__file__))

scoring_criteria_PNRs, scoring_criteria_PNRs_toggle = csv_reader_buss(pd.read_csv(os.path.join(moduleDir, "Rules", "Business_Rules_PNR.csv")))

scoring_criteria_Flights, scoring_criteria_Flights_toggle = csv_reader_plane(pd.read_csv(os.path.join(moduleDir, "Rules", "Flight_Scoring.csv")))

