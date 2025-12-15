#!/usr/bin/env python
"""
TDC Oracle subprocess wrapper.

This script runs in the isolated 'tdc' pixi environment and evaluates molecules
using TDC oracles. It communicates via command-line arguments and JSON output.

Usage:
    pixi run -e tdc python scripts/tdc_oracle_subprocess.py <oracle_name> <smiles> [--target_smiles <target>]

Output:
    JSON object with 'score' field, or 'error' field on failure.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Evaluate a molecule using a TDC oracle")
    parser.add_argument("oracle_name", help="Name of the TDC oracle (e.g., drd2, gsk3b, jnk3)")
    parser.add_argument("smiles", help="SMILES string of the molecule to evaluate")
    parser.add_argument("--target_smiles", default=None, help="Target SMILES for meta-oracles")
    
    args = parser.parse_args()
    
    try:
        # Import TDC only when running (not at module level)
        import tdc
        
        # Initialize oracle
        oracle = tdc.Oracle(
            name=args.oracle_name,
            target_smiles=args.target_smiles,
        )
        
        # Evaluate molecule
        score = oracle(args.smiles)
        score = float(score)  # Ensure JSON serializable
        
        result = {"score": score}
        
    except Exception as e:
        result = {"error": str(e)}
    
    # Output JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
