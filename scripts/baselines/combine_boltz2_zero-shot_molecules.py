import json
import pandas as pd
import os
from pathlib import Path

folder_path = Path("/Users/ahunklinger/Coding/molecule-optimization-agent/data/results/zero_shot")
targets = ["boltz2_mgyp", "boltz2_trib2"]
prompt_styles = ["task", "generic"]

data_rows = []

for target in targets:
    for prompt_style in prompt_styles:
        if prompt_style == "generic":
            path = folder_path / "boltz2_mgyp" / prompt_style
        else:
            path = folder_path / target / prompt_style
        
        if not path.exists():
            print(f"Skipping {path} (does not exist)")
            continue
        
        json_files = sorted(path.glob("*.json"))
        print(f"Processing {path}: found {len(json_files)} files")
        
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, "r") as f:
                    data_obj = json.load(f)
                
                trace = data_obj.get("trace", [])
                
                for entry in trace:
                    # Skip failed parses
                    if entry.get("parse_status") == "failed":
                        continue
                    
                    smiles = entry.get("smiles")
                    if not smiles:
                        continue
                    
                    data_rows.append({
                        "target": target.replace("boltz2_", ""),
                        "run": i+1,
                        "prompt_style": prompt_style,
                        "iteration": entry.get("iteration"),
                        "smiles": smiles,
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

if data_rows:
    df = pd.DataFrame(data_rows)
    print(f"\nLoaded {len(df)} molecules total")
    print(f"Unique SMILES: {df['smiles'].nunique()}")
    print(f"Targets: {df['target'].unique().tolist()}")
    print(f"Prompt styles: {df['prompt_style'].unique().tolist()}")
    print("\nDataFrame shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Optional: save to CSV
    output_file = folder_path / "combined_boltz2_zero-shot_molecules.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
else:
    print("No molecules loaded!")
