from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel
from dwave.system import LeapHybridCQMSampler
import numpy as np
import pandas as pd

#Defining a class Passenger for a particular PNR
class Passenger:
    def __init__(self, PAX, ID):
        """Args:
        PAX: No. of passengers on PNR
        ID: ID of a PNR"""
        self.PAX = PAX
        self.ID = ID

#Defining a class Flight for a particular Flight
class Flight:
    def __init__(self, classes, ID, src, dest):
        """Args:
        ID: ID of a flight
        src: Airport from which it is leaving 
        dest: Airport at which it is arriving
        classes: Pandas DataFrame having the no. of seats of each class"""
        self.ID = ID
        self.classes = {}
        self.src = src
        self.dest = dest
        for i in classes.columns:
            self.classes[str(i[:2])] = int(classes[i].iloc[0])
            
# FUNCTION TO MAP PASSENGER REACCOMODATION TO A MULTI KNAPSACK PROBLEM
def build_knapsack_cqm(PNR, paths, reward, alpha, src, dest):
    """Construct a CQM for the knapsack problem.

    Args:
        PNR (array-like):
            Array of passengers as a objects.
        paths :
            Array of paths and a path is a array of flights as objects.
        reward:
            Dict of dict of passengers for a score of each class
        alpha:
            Score for each path

    Returns:
        Constrained quadratic model instance that represents the knapsack problem.
    """
    num_PNR = len(PNR)
    
    print(f"\nBuilding a CQM for {num_PNR} PNRs and {len(paths)} paths")

    cqm = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    
    flights = {}  #Dictionary keeping Track of flights and their corresponding paths
    for c in range(len(paths)):
        for flt in paths[c]:
            if flt.ID not in flights:
                    flights[flt.ID] = [flt,[c]]
            else:
                flights[flt.ID][1].append(c)
    
    for ID in flights:
        for cls in flights[ID][0].classes:
            constraint = QuadraticModel()
            for passengers in PNR:
                if reward[passengers.ID][cls] != 0:
                    for c in flights[ID][1]:
                        constraint.add_variable('BINARY', (c, ID, passengers.ID, cls))
                        obj.add_linear((c, ID, passengers.ID, cls), alpha[c]*( -reward[passengers.ID][cls]))
                        constraint.set_linear((c, ID, passengers.ID, cls), passengers.PAX)
                    #Capacity Constrainst of a class in a particular flight
            cqm.add_constraint(constraint, sense="<=", rhs=flights[ID][0].classes[cls], label=f'Capacity of class {cls} in {ID}')

    
    for passengers in PNR:
        # Summation of outgoing and incoming flights from src and dest respectively for a particular passenger in all classes is less than equal to 1
        constraint1 = QuadraticModel()
        constraint2 = QuadraticModel()
        for ID in flights:
            if flights[ID][0].src in src:
                for cls in flights[ID][0].classes:
                    if reward[passengers.ID][cls] != 0: 
                        for c in flights[ID][1]: 
                            constraint1.add_variable('BINARY', (c, ID, passengers.ID, cls))
                            constraint1.set_linear((c, ID, passengers.ID, cls), 1)
            if flights[ID][0].dest in dest:
                for cls in flights[ID][0].classes:
                    if reward[passengers.ID][cls] != 0: 
                        for c in flights[ID][1]: 
                            constraint2.add_variable('BINARY', (c, ID, passengers.ID, cls))
                            constraint2.set_linear((c, ID, passengers.ID, cls), 1)
                            
        cqm.add_constraint(constraint1, sense="<=", rhs=1, label = f"Outgoing Flights from src for passenger {passengers.ID}")
        cqm.add_constraint(constraint2, sense="<=", rhs=1, label = f"Incoming Flights to dest for passenger {passengers.ID}")
    
    airports = {} #Dictionary Keeping track of Flights incoming and outgoing at a particular airport for a particular path
    for flight in flights:
        if flights[flight][0].src not in airports:
            incoming = [flight]
            outcoming = []
            airports[flights[flight][0].src] = (incoming, outcoming)
        else:
            airports[flights[flight][0].src][0].append(flight)
            
        if flights[flight][0].dest not in airports:
            incoming = []
            outcoming = [flight]
            airports[flights[flight][0].dest] = (incoming, outcoming)
        else:
            airports[flights[flight][0].dest][1].append(flight)
    
    for passengers in PNR: #Summation of a path is equal to its length if started on
        for c in range(len(paths)):
            constraint = QuadraticModel()
            for flt in range(1,len(paths[c])):
                flight = paths[c][flt]
                for cls in flight.classes:
                    if reward[passengers.ID][cls] != 0:
                        constraint.add_variable('BINARY', (c, flight.ID, passengers.ID, cls))
                        constraint.set_linear((c, flight.ID, passengers.ID, cls), 1)
            
            for cls in paths[c][0].classes:
                flight = paths[c][0]
                if reward[passengers.ID][cls] != 0:
                        constraint.add_variable('BINARY', (c, flight.ID, passengers.ID, cls))
                        constraint.set_linear((c, flight.ID, passengers.ID, cls), -(len(paths[c])-1))
                        
            cqm.add_constraint(constraint, sense = "==", rhs=0, label = f'Path Preservance of {passengers.ID} for {c} path')
                
    for passengers in PNR:
        #Summation of the incoming flights at a airport for a passenger is than equal to the summation of outgoing flights (Path Preservence)
        for station in airports:
            if station not in src and station not in dest:
                constraint = QuadraticModel()
                for flt in airports[station][0]:
                    for cls in flights[flt][0].classes:
                        if reward[passengers.ID][cls] != 0: 
                            for c in flights[flt][1]:
                                constraint.add_variable( 'BINARY', (c, flt, passengers.ID, cls))
                                constraint.set_linear((c, flt, passengers.ID, cls), 1)
                                
                for flt in airports[station][1]:
                    for cls in flights[flt][0].classes:
                        if reward[passengers.ID][cls] != 0: 
                            for c in flights[flt][1]:
                                constraint.add_variable( 'BINARY', (c, flt, passengers.ID, cls))
                                constraint.set_linear((c, flt, passengers.ID, cls), -1)
                                
                cqm.add_constraint(constraint, sense = "==", rhs=0, label = f'Path Preservance of {passengers.ID} at airport {station}')
        
    cqm.set_objective(obj)
    return cqm

# FUNCTION TO PARSE THE SOLUTION OF PASSENGER REACCOMODATION AND STORING RESULTS IN CSV FILE
def parse_solution(sampleset, passenger_flights, disrupt, abs_alpha, scores, paths):
    """Translate the sampler sample returned from solver to shipped items.

    Args:
        sampleset (dimod.Sampleset):
            Samples returned from the solver.
    """
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    if not len(feasible_sampleset):
        print("No feasible solution found")
        return None

    # Find the minimum energy
    min_energy = min(sampleset.data_vectors['energy'])

    # Collect all samples with the minimum energy
    min_energy_samples = [sample for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']) if energy == min_energy]

    #Converting normalised alpha to abs_alpha
    for i in range(len(abs_alpha)):
        abs_alpha[i] = abs_alpha[i] * (len(paths[i])**2)

    dataframes = []
    for sampler in min_energy_samples:  #Sampling samples with same energy
        arr = list()
        
        nos = 0
        for i in sampler:
            if sampler[i] == 1:
                arr.append(i)
                nos += 1
                
        if nos==0:
            continue
        
        df = pd.DataFrame(arr)
        
        df.rename(columns={0: 'Path', 1: 'Flight ID', 2: 'PNR ID', 3: 'Class'}, inplace=True)
            # Find the most frequent value(s)
        most_frequent_values = df['Path'].mode()
            
            # Getting all the unique paths
        unique_paths = most_frequent_values.unique()
        max_paths = [unique_paths[0]]  #Path with the highest score
        mpth_score = abs_alpha[unique_paths[0]] #Paths with the highest scores
        
        if len(unique_paths) > 1:
            for path in unique_paths[1:]:
                    if abs_alpha[path] > mpth_score: #Paths with higher score replace all low scores path
                        mpth_score = abs_alpha[path]
                        max_paths = [path]
                    elif abs_alpha[path] == mpth_score: #Paths with equal score are appended
                        max_paths.append(path)
        
        if len(max_paths) > 1: #If still there are more than one unqiue paths check PNRs
            selecter = []    
            for path in max_paths:
                check_df = df[df['Path'] == path]
                n = len(paths[path])  #Checking path length
                rank = 0
                for pnr in range(len(check_df)):  #Adding PNR scores together
                    rank += scores[check_df.iloc[pnr]['PNR_ID']][check_df.iloc[pnr]['Class']]
                rank /= n 
                selecter.append([path, rank])
            max_paths = [max(selecter, key = lambda x: x[1])[0]]  #Selecting the maximum ranked path
            
        most_frequent_values = df[df['Path'] == max_paths[0]]['Path']   
         
        # Filter DataFrame to remove rows with the most frequent values
        filtered_df = df[~df['Path'].isin(most_frequent_values)]

        # Create a DataFrame with the removed rows
        removed_df = df[df['Path'].isin(most_frequent_values)]
        
        absent = passenger_flights[~passenger_flights['RECLOC'].isin(df['PNR ID'])][['RECLOC']].drop_duplicates(subset='RECLOC')
    
        absent.rename(columns={'RECLOC': 'PNR ID'}, inplace=True)
        filtered_df = pd.concat([filtered_df, absent], ignore_index=True)
        
        dataframes.append([removed_df, filtered_df])
    
    
    if len(dataframes) > 1:  #If more than one sample with same energies
        subframes = [dataframes[0]]
        n = len(subframes[0][0]['PNR ID'].unique())
        for i in dataframes[1:]:   #Sample based on the no. of passengers accomodates
            if len(i[0]['PNR ID'].unique()) == n:
                subframes.append(i)
            elif len(i[0]['PNR ID'].unique()) > n:
                subframes = [i]
                n = len(i[0]['PNR ID'].unique())
        
        if len(subframes) > 1:  #If still equal check path score
            subframes = [max(subframes, key = lambda x: abs_alpha[x[0]['Path'].iloc[0]])]
        
        dataframes = subframes  #Most aprropriate solution
    
    if len(dataframes) != 0: 
        dataframes[0][0].to_csv(f"Default_solution_{disrupt}.csv")
        dataframes[0][1].to_csv(f"Exception_list_{disrupt}.csv")
        
        print('Accomodations done, check the default and the exception files')
    else:
        print("No accomodations available")
        
    return feasible_sampleset

#FUNCTION TO SOLVE THE KNAPSACK PROBLEM AND GIVE AN OUTPUT AS CSV FILES
def reaccomodation(PNR, paths, reward, alpha, src, dest, passenger_flights, disrupt, TOKEN):
    """Solve a knapsack problem using a CQM solver."""

    sampler = LeapHybridCQMSampler(token=TOKEN)

    cqm = build_knapsack_cqm(PNR, paths, reward, alpha, src, dest)
    
    print("Submitting CQM to solver {}.".format(sampler.solver.name))
    
    sampleset = sampler.sample_cqm(cqm, label='Multi-Knapsack')

    return parse_solution(sampleset, passenger_flights, disrupt, alpha, reward, paths)
