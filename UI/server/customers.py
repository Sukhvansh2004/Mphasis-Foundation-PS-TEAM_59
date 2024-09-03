import requests
from config import *

PNRP = 'https://docs.google.com/spreadsheets/d/11-Y7q3yHLTSS9n0wRPC3Wnrfm-mDmuot/edit#gid=947711678'
PNRB = 'https://docs.google.com/spreadsheets/d/14GZCdnhcSHWIZzSqmtukgbOqhmjig5W0/edit#gid=1259380578'







def get_sheet_data(sheet_url):
    """Returns a 2-dimensional array of data from the given sheet"""
    sheet_id = sheet_url.split('/')[5]
    gid = sheet_url.split('/')[6].split('=')[1]
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=tsv&gid={gid}"
    response = requests.get(url)
    data = response.text
    rows = data.split('\n')
    sheet_data = []
    for row in rows:
        sheet_data.append(row.split('\t'))
    return sheet_data


# Calculate the priority of the passengers based on the PNR Ranking
def priority_passengers():
    global PNRB
    global PNRP
    global PNR_RANKING
    passenger_info = get_sheet_data(PNRP)
    n = len(passenger_info) - 1
    passenger_priority = [0]*n
    get_pnrb = get_sheet_data(PNRB)
    pnrb = {}
    for i in range(1, len(get_pnrb)):                                    #Make a dictonary for PNRB sheet to get the PAX_CNT and cabin
        pnrb[get_pnrb[i][0]] = (get_pnrb[i][4], get_pnrb[i][7])
    
    
    
    
    for i in range(n):
        pass_info = passenger_info[i + 1]
        #First rule
        possible_SSR = [
            'INF'
            'WCHR',
            'WCHS',
            'WCHC',
            'LANG',
            'CHD',
            'MAAS',
            'UNMR',
            'BLND',
            'DEAF',
            'EXST',
            'MEAL',
            'NSST',
            'NRPS'
        ]
        if pass_info[11] in possible_SSR or pass_info[12] in possible_SSR or pass_info[13] in possible_SSR:
            passenger_priority[i] += PNR_RANKING['score'][0]*int(PNR_RANKING['enabled'][0])
        
        
        
        #Rule 2 - 7
        passenger_cabin = pnrb[pass_info[0]][0]
        if passenger_cabin == 'FirstClass':
            passenger_priority[i] += PNR_RANKING['score'][1]*int(PNR_RANKING['enabled'][1])
        elif passenger_cabin == 'BusinessClass':
            passenger_priority[i] += PNR_RANKING['score'][2]*int(PNR_RANKING['enabled'][2])
        elif passenger_cabin == 'PremiumEconomyClass':
            passenger_priority[i] += PNR_RANKING['score'][3]*int(PNR_RANKING['enabled'][3])
        elif passenger_cabin == 'DiscountedFirstClass':
            passenger_priority[i] += PNR_RANKING['score'][4]*int(PNR_RANKING['enabled'][4])
        elif passenger_cabin == 'DiscountedBusinessClass':
            passenger_priority[i] += PNR_RANKING['score'][5]*int(PNR_RANKING['enabled'][5])
        elif passenger_cabin == 'EconomyClass':
            passenger_priority[i] += PNR_RANKING['score'][6]*int(PNR_RANKING['enabled'][6])
            
            
            
            
        #Rule 8 - 9 ?? Don't know how to do it
        
        #Rule 10 - 11
        passenger_group = pnrb[pass_info[0]][1]
        passenger_priority[i] += passenger_group*PNR_RANKING['score'][10]*int(PNR_RANKING['enabled'][10])       #50*PAX_CNT
        if passenger_group > 1:
            passenger_priority[i] += PNR_RANKING['score'][9]*int(PNR_RANKING['enabled'][9])                     #500 for a group
            
        #Rule 12
        
        if PNR_RANKING['enabled'][11]:
            if pass_info[15] == 'Silver':
                passenger_priority[i] += PNR_RANKING['score'][11][3]
            elif pass_info[15] == 'Gold':
                passenger_priority[i] += PNR_RANKING['score'][11][2]
            elif pass_info[15] == 'Platinum':
                passenger_priority[i] += PNR_RANKING['score'][11][1]
            elif pass_info[15] == 'PresidentialPlatinum':
                passenger_priority[i] += PNR_RANKING['score'][11][0]
            
    
    
    
    return passenger_priority