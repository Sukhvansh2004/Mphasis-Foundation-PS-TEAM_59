import json
import os
import csv
from collections import Counter

# Open the ZIP and iterate JSON files (assuming current directory contains the JSON files)
import zipfile
zf = zipfile.ZipFile("Leap Hybrid JSONs.zip")
results = []

for name in zf.namelist():
    if not name.endswith('.json'):
        continue
    with zf.open(name) as f:
        data = json.load(f)
    # Extract QPU times and number of variables
    qpu_access_time = data['info'].get('qpu_access_time', None)
    charge_time = data['info'].get('charge_time', None)
    run_time = data['info'].get('run_time', None)
    total_vars = data['num_variables']
    # Get energies and feasible flags
    energies = data['vectors']['energy']['data']
    feasible = data['vectors']['is_feasible']['data']
    feas_inds = [i for i, flag in enumerate(feasible) if flag]
    if not feas_inds:
        continue  # skip if no feasible solutions
    # Find minimum energy among feasible
    min_energy = min(energies[i] for i in feas_inds)
    min_inds = [i for i in feas_inds if energies[i] == min_energy]
    # All PNRs in the problem
    var_labels = data['variable_labels']
    all_pnrs = {lbl[2] for lbl in var_labels}
    total_pnr = len(all_pnrs)
    # Analyze each min-energy solution
    for idx in min_inds:
        sample = data['sample_data']['data'][idx]
        # Find indices where variable=1
        ones_idx = [i for i,v in enumerate(sample) if v == 1.0]
        # Map PNR -> list of paths for that PNR
        pnr_to_paths = {}
        for i in ones_idx:
            path, flight, pnr, cls = var_labels[i]
            pnr_to_paths.setdefault(pnr, []).append(path)
        # Determine unique path for each PNR (if multiple, take any one)
        pnr_path_map = {}
        for pnr, paths in pnr_to_paths.items():
            unique = set(paths)
            pnr_path_map[pnr] = next(iter(unique))  # take first (all are identical if repeated)
        # Count how many PNRs use each path
        path_counts = Counter(pnr_path_map.values())
        if path_counts:
            default_path, default_count = path_counts.most_common(1)[0]
        else:
            default_path, default_count = (None, 0)
        exception_count = len(pnr_path_map) - default_count
        unacc_count = total_pnr - len(pnr_path_map)
        # Percentages of total PNRs
        default_perc = default_count/total_pnr*100 if total_pnr else 0
        exception_perc = exception_count/total_pnr*100 if total_pnr else 0
        unacc_perc = unacc_count/total_pnr*100 if total_pnr else 0
        results.append({
            "filename": os.path.basename(name),
            "energy": min_energy,
            "qpu_access_time": qpu_access_time,
            "charge_time": charge_time,
            "run_time": run_time,
            "total_variables": total_vars,
            "default_count": default_count,
            "default_percentage": default_perc,
            "exception_count": exception_count,
            "exception_percentage": exception_perc,
            "unaccommodated_count": unacc_count,
            "unaccommodated_percentage": unacc_perc
        })

# Write aggregated results to CSV
fieldnames = ["filename","energy","qpu_access_time","charge_time","run_time",
              "total_variables","default_count","default_percentage",
              "exception_count","exception_percentage",
              "unaccommodated_count","unaccommodated_percentage"]
with open('results.csv','w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        # Optionally format floats, e.g. round percentages
        row['default_percentage'] = round(row['default_percentage'],6)
        row['exception_percentage'] = round(row['exception_percentage'],6)
        row['unaccommodated_percentage'] = round(row['unaccommodated_percentage'],6)
        row['energy'] = round(row['energy'],6)
        writer.writerow(row)
