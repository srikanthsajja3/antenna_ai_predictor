import os
import csv
from pyaedt import Hfss

# 1. Configuration
# Make sure your HFSS design has these variable names
VARIABLES = {
    "Length": [10.5, 11.5, 12.5],  # Example range in mm
    "Width": [5.2, 6.2, 7.2],
    "Gap": [0.2, 0.4, 0.6]
}

# 2. Connect to HFSS
# This will connect to the active design in an open AEDT session
hfss = Hfss(specified_version="2025.2")

print(f"Connected to Project: {hfss.project_name}, Design: {hfss.design_name}")

# 3. Prepare CSV File
csv_file = "dataset.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Length_mm", "Width_mm", "Gap_mm", "Bandwidth_GHz", "Gain_dBi"])

    # 4. Simulation Loop
    for l in VARIABLES["Length"]:
        for w in VARIABLES["Width"]:
            for g in VARIABLES["Gap"]:
                print(f"\n--- Simulating: L={l}, W={w}, G={g} ---")
                
                # Update Variables in HFSS
                hfss["Length"] = f"{l}mm"
                hfss["Width"] = f"{w}mm"
                hfss["Gap"] = f"{g}mm"
                
                # Run Simulation
                hfss.analyze_nominal()
                
                # 5. Extract Results (Bandwidth and Max Gain)
                # Note: These report names depend on your HFSS setup
                # We assume you have a 'Gain' and 'S11' report
                try:
                    # Simplified extraction - you may need to adjust these based on your specific reports
                    gain_data = hfss.post.get_report_data("Gain Plot") # Change to your report name
                    s11_data = hfss.post.get_report_data("S11 Plot")  # Change to your report name
                    
                    # Logic to calculate BW and Max Gain from data...
                    # For now, we use placeholders to show the flow
                    bandwidth = 6.0 + (l/10) # Placeholder logic
                    max_gain = 7.0 + (w/10)   # Placeholder logic
                    
                    print(f"Result: BW={bandwidth:.2f}GHz, Gain={max_gain:.2f}dBi")
                    writer.writerow([l, w, g, bandwidth, max_gain])
                    f.flush() # Save progress immediately
                    
                except Exception as e:
                    print(f"Error extracting data: {e}")

print("\nDataset generation complete!")
hfss.release_desktop()
