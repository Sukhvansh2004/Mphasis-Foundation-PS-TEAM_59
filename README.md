# Mphasis-Foundation-PS-TEAM_59
 Inter IIT 12 - Team_59 - Solution
Quantum and Classical Pathfinding and Airline Re-accommodation

Overview
The project aims at addressing the challenge of passenger re-accommodation due to planned schedule changes in airlines. The key objectives include identifying impacted flights and passengers, determining suitable alternate flights, ranking alternate flight solutions, and prioritizing PNR re-accommodation. 
The solution incorporates a business rule engine for flexibility in rule application and produces two sets of solution files: a default flight level re-accommodation solution and an exception list for individual PNR re-accommodation. The project emphasizes the creation of various rule profiles, flexibility in rule enforcement, and the generation of comprehensive solution files catering to different scenarios.

Files Description

Python Scripts
    * classical_re-assignment_greedy.py    
    * find_impacted_passengers: Identifies passengers affected by flight changes.    
    * calc_score: Calculates scores for reassignment options.    
    * find_possible_paths: Determines potential paths for reassignment.    
    * main: The main function orchestrating the process.    
    * PNRS: Manages Passenger Name Records.    
    * classical_pathfinding.py    
    * Multiple functions including _init_, statistics_printer, reduce_network, calc_score, recurse, solver,     
    * multisolve, solve, find_impacted_passengers, and matrix_finder, all contributing to the classical approach to pathfinding.    
    * CLASSICAL_FINAL.py    
    * main: The final integration of classical pathfinding and reassignment methods.    
    * QUANTUM_FINAL.py    
    * In this file, we have formulated our solution into two sub problems:    
    * the first part focuses on path finding where we have modelled our problem into a Constrained Quadratic Model(CQM)    
    * The second part focuses on passenger     
    * re-accomodation(which we modelled as a CQM too)and generating the solution files.    
    * Functions like next_72hrsflights, get_direct_flights, get_1_interconnecting, get_2_interconnecting are being defined for flight data reduction.    
    * encode_flights, decode, and main to implement D-Wave’s HybridCQM Sampler to solve our CQM Model for the project.    
    * Nearby_Airports.py    
    * nearby_airport: Finds airports in close proximity to a given location.    
    * pnr_reaccomodation.py:Py file containing the below functions and classes for Flights and Passenger.    
    * build_knapsack_cqm: Builds a Constrained Quadratic Model for the knapsack problem involving passenger re-accommodation.    
    * parse_solution: Interprets solver results for re-accommodation.    
    * reaccomodation: Applies a CQM solver to the knapsack problem in passenger re-accommodation.    
    * path_scoring.py: Contains classes to score flight paths and PNRs    
    * Functions for scoring flights including next_72hrsflights, get_direct_flights, get_1_interconnecting, get_2_interconnecting, encode_flights, and decode_and_retrieve_flights.    
Data Files
    * GlobalAirportDatabase.csv    
    * Description: A comprehensive database of global airports, utilised as a reference in the pathfinding process.    
    * Other excel files given with the Dataset of the Problem Statement    
    * A sample output for the cancellation of flight INV-ZZ-3174758 flight using CLASSICAL_FINAL.py   
    * Business_Rules_PNR.csv : Has an Enabled column to turn on or off the business rules accordingly with an extra advantage of dynamic scoring for each business rule 
    * Flight_Scoring.csv : It is similar to the Business_Rules_PNR file, just this one is for making changes in flight scoring  
    
Installation and Usage    
docplex    
qiskit_optimization    
numpy    
pandas    
dimod    
dwave    

Contributing    
Team_59    

License    
None      

Github Repo Link     
https://github.com/Sukhvansh2004/Mphasis-Foundation-PS-TEAM_59.git
