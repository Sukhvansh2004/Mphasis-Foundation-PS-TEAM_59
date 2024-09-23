from flask import Flask, render_template, request
from config import UPLOAD_FOLDER
import os
import sys
import pandas as pd
current_dir = os.getcwd()
enginePath = os.path.join(current_dir, "engine")

if(os.path.isdir(enginePath)):
    sys.path.append(enginePath)

from CLASSICAL_FINAL import main as hybrid
from QUANTUM_FINAL import main as quantum

app = Flask(__name__)

@app.route('/')
def index():
    pnrRules = pd.read_csv(os.path.join(enginePath, "Rules", "Business_Rules_PNR.csv"))
    pnr_ranking_score = pnrRules['Score'].to_list()
    pnr_ranking_enabled = pnrRules['Enabled'].to_list()
    
    flightRules = pd.read_csv(os.path.join(enginePath, "Rules", "Flight_Scoring.csv"))
    flight_ranking_score = flightRules['Score'].to_list()
    flight_ranking_enabled = flightRules['Enabled'].to_list()
    
    context = {
        'pnr_ranking_score': pnr_ranking_score,
        'pnr_ranking_enabled': pnr_ranking_enabled,
        'flight_ranking_score': flight_ranking_score,
        'flight_ranking_enabled': flight_ranking_enabled
    }
    return render_template('index.html', **context)


@app.route('/submit-dataset', methods=['POST'])
def submit_dataset():
    if 'file-1' in request.files:
        file1 = request.files['file-1']
        print(file1.filename)
        file1.save(os.path.join(UPLOAD_FOLDER, 'inv.csv'))
    else:
        sheet1 = request.form['sheet-1']
        print(sheet1)
    
    if 'file-2' in request.files:
        file2 = request.files['file-2']
        print(file2.filename)
        file2.save(os.path.join(UPLOAD_FOLDER, 'sch.csv'))
    else:
        sheet2 = request.form['sheet-2']
        print(sheet2)
    
    if 'file-3' in request.files:
        file3 = request.files['file-3']
        print(file3.filename)
        file3.save(os.path.join(UPLOAD_FOLDER, 'pnrb.csv'))
    else:
        sheet3 = request.form['sheet-3']
        print(sheet3)
    
    if 'file-4' in request.files:
        file4 = request.files['file-4']
        print(file4.filename)
        file4.save(os.path.join(UPLOAD_FOLDER, 'pnrp.csv'))
    else:
        sheet4 = request.form['sheet-4']
        print(sheet4)

    response = {
        'status': 'success',
        'title': 'Success',
        'message': 'Dataset imported successfully'
    }
    return response


@app.route('/update-pnr-ranking-rules', methods=['POST'])
def update_pnr_ranking_rules():
    data = request.get_json()
    # print(data)
    pnrRules = pd.read_csv(os.path.join(enginePath, "Rules", "Business_Rules_PNR.csv"))
    try:
        pnr_ranking_score = data['pnr_ranking_score']
    except:
        response = {
            'status': 'error',
            'title': 'Error',
            'message': 'Invalid PNR ranking score'
        }
        return response
    
    pnr_ranking_enabled = data['pnr_ranking_enabled']
    
    pnrRules['Score'] = pnr_ranking_score
    pnrRules['Enabled'] = pnr_ranking_enabled
    print('Updated PNR ranking rules\nPNR_RANKING = ', end='')
    pnrRules = pnrRules[['Business Rules','Enabled','Score']]
    print(pnrRules)
    response = {
        'status': 'success',
        'title': 'Success',
        'message': 'PNR ranking rules updated successfully'
    }
    pnrRules.to_csv(os.path.join(enginePath, "Rules", "Business_Rules_PNR.csv"))
    return response

@app.route('/update-flight-ranking-rules', methods=['POST'])
def update_flight_ranking_rules():
    data = request.get_json()
    # print(data)
    flightRules = pd.read_csv(os.path.join(enginePath, "Rules", "Flight_Scoring.csv"))
    try:
        flight_ranking_score = data['flight_ranking_score']
    except:
        response = {
            'status': 'error',
            'title': 'Error',
            'message': 'Invalid PNR ranking score'
        }
        return response
    
    flight_ranking_enabled = data['flight_ranking_enabled']
    
    flightRules['Score'] = flight_ranking_score
    flightRules['Enabled'] = flight_ranking_enabled
    print('Updated PNR ranking rules\nPNR_RANKING = ', end='')
    flightRules = flightRules[['Flight Rules','Enabled','Score']]
    print(flightRules)
    response = {
        'status': 'success',
        'title': 'Success',
        'message': 'PNR ranking rules updated successfully'
    }
    flightRules.to_csv(os.path.join(enginePath, "Rules", "Flight_Scoring.csv"))
    return response


@app.route('/reschedule', methods=['POST'])
def reschedule():
    # Handle the request data here
    data = request.get_json()
    # Process the data or perform actions as needed
    mode = data["Mode"]
    try:
        if(mode == 'Quantum'):
            defaults, exceptions = quantum(*data["Flights"], TOKEN=data["Token"])
        else:
            defaults, exceptions = hybrid(*data["Flights"], TOKEN=data["Token"])
            
        response = {
            'status': 'success',
            'title': 'Success',
            'message': 'Flights rescheduled Successfully',
            'result': [defaults, exceptions]
        }
    
        return response
    
    except Exception as e:
        
        response = {
            'status' : 'error',
            'title' : 'Unable to Reschedule Flights',
            'message' : f'{e}'
        }

        return response

if __name__ == '__main__':
    app.run(debug=True)