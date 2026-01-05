#!/usr/bin/env python3
"""
Script to check which statement generation methods were used for selected statements.

Usage:
    python check_statement_generation_methods.py <algorithm_name> [directory_name]

    If directory_name is not provided, processes all directories in data/results/statements/
    If directory_name is provided, it can be:
        - A single directory name (e.g., "0")
        - A comma-separated list of directories (e.g., "1,2,3,4")

Example:
    python check_statement_generation_methods.py greedy 0
    python check_statement_generation_methods.py greedy "1,2,3,4"
    python check_statement_generation_methods.py greedy  # processes all directories
"""

import json
import pandas as pd
import sys
from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict


def find_files(base_dir: Path, algorithm_name: str):
    """Find the required files in the directory structure (recursively)."""
    raw_output_file = None
    utility_matrix_file = None
    algorithm_file = None
    
    # Recursively search for files
    for path in base_dir.rglob("*"):
        if path.is_file():
            # Look for statement_generation_raw_output.csv
            if path.name == "statement_generation_raw_output.csv" and raw_output_file is None:
                raw_output_file = path
            
            # Look for utility_matrix_statements.csv
            elif path.name == "utility_matrix_statements.csv" and utility_matrix_file is None:
                utility_matrix_file = path
            
            # Look for algorithm JSON file in assignments directory
            elif path.name == f"{algorithm_name}.json" and "assignments" in path.parts and algorithm_file is None:
                algorithm_file = path
    
    return raw_output_file, utility_matrix_file, algorithm_file


def process_directory(directory_name: str, algorithm_name: str, verbose: bool = True) -> Optional[List[Dict]]:
    """Process a single directory and return results."""
    base_path = Path(__file__).parent.parent / "data" / "results" / "statements"
    base_dir = base_path / directory_name
    
    if not base_dir.exists():
        if verbose:
            print(f"Error: Directory not found: {base_dir}")
        return None
    
    # Find the required files
    raw_output_file, utility_matrix_file, algorithm_file = find_files(base_dir, algorithm_name)
    
    # Check if all files exist
    if raw_output_file is None or not raw_output_file.exists():
        if verbose:
            print(f"Error: statement_generation_raw_output.csv not found in {base_dir} or subdirectories")
        return None
    
    if utility_matrix_file is None or not utility_matrix_file.exists():
        if verbose:
            print(f"Error: utility_matrix_statements.csv not found in {base_dir} or subdirectories")
        return None
    
    if algorithm_file is None or not algorithm_file.exists():
        if verbose:
            print(f"Error: {algorithm_name}.json not found in assignments subdirectory within {base_dir}")
        return None
    
    if verbose:
        print("Found files:")
        print(f"  - {raw_output_file}")
        print(f"  - {utility_matrix_file}")
        print(f"  - {algorithm_file}")
        print()
    
    # Read the algorithm JSON to get selected statement IDs
    with open(algorithm_file, 'r') as f:
        algorithm_data = json.load(f)
    
    if 'slate' not in algorithm_data:
        if verbose:
            print(f"Error: 'slate' key not found in {algorithm_file}")
        return None
    
    selected_statement_ids = algorithm_data['slate']
    if verbose:
        print(f"Selected statement IDs: {selected_statement_ids}")
        print()
    
    # Read utility_matrix_statements.csv to map IDs to statements
    utility_df = pd.read_csv(utility_matrix_file)
    if verbose:
        print(f"Found {len(utility_df)} statements in utility_matrix_statements.csv")
    
    # Read statement_generation_raw_output.csv to get generation methods
    raw_output_df = pd.read_csv(raw_output_file)
    if verbose:
        print(f"Found {len(raw_output_df)} statements in statement_generation_raw_output.csv")
        print()
    
    # Create a mapping from statement text to generation method
    statement_to_method = {}
    for _, row in raw_output_df.iterrows():
        statement_text = row['statement']
        generation_method = row['generation_method']
        statement_to_method[statement_text] = generation_method
    
    # Now map selected statement IDs to their generation methods
    results = []
    for stmt_id in selected_statement_ids:
        # Find the statement text for this ID
        stmt_row = utility_df[utility_df['id'] == stmt_id]
        if stmt_row.empty:
            if verbose:
                print(f"Warning: Statement ID {stmt_id} not found in utility_matrix_statements.csv")
            continue
        
        statement_text = stmt_row.iloc[0]['statement']
        
        # Try to find matching generation method
        generation_method = statement_to_method.get(statement_text)
        
        if generation_method is None:
            # Try to find a close match (normalize whitespace)
            normalized_text = ' '.join(statement_text.split())
            for key, method in statement_to_method.items():
                normalized_key = ' '.join(key.split())
                if normalized_text == normalized_key:
                    generation_method = method
                    break
        
        if generation_method is None:
            # This statement might be from the survey (baseline statements)
            generation_method = "SURVEY/BASELINE (not generated)"
        
        results.append({
            'statement_id': stmt_id,
            'generation_method': generation_method,
            'directory': directory_name
        })
    
    return results


def check_statement_generation_methods(directory_name: Optional[str], algorithm_name: str):
    """Check which generation methods were used for selected statements."""
    base_path = Path(__file__).parent.parent / "data" / "results" / "statements"
    
    # Determine which directories to process
    if directory_name is None:
        # Process all directories
        directories = [d.name for d in base_path.iterdir() if d.is_dir()]
        if not directories:
            print(f"Error: No directories found in {base_path}")
            return
    else:
        # Parse comma-separated directory names
        directories = [d.strip() for d in directory_name.split(',') if d.strip()]
        if not directories:
            print(f"Error: No valid directories found in '{directory_name}'")
            return
    
    all_results = []
    
    # Process each directory
    for dir_name in directories:
        print("=" * 80)
        print(f"Processing directory: {dir_name}")
        print("=" * 80)
        print()
        
        results = process_directory(dir_name, algorithm_name, verbose=True)
        
        if results is None:
            print(f"Skipping directory {dir_name} due to errors.\n")
            continue
        
        # Print per-directory results
        print("=" * 80)
        print(f"Generation Methods for Selected Statements (Algorithm: {algorithm_name})")
        print(f"Directory: {dir_name}")
        print("=" * 80)
        print()
        
        for result in results:
            print(f"  {result['statement_id']}: {result['generation_method']}")
        
        print()
        print("=" * 80)
        print(f"Summary by Generation Method (Directory: {dir_name}):")
        print("=" * 80)
        
        method_counts = Counter([r['generation_method'] for r in results])
        for method, count in method_counts.most_common():
            print(f"  {method}: {count} statement(s)")
        
        print()
        print(f"Total selected statements: {len(results)}")
        print()
        print()
        
        all_results.extend(results)
    
    # Print aggregate summary if multiple directories were processed
    if len(directories) > 1:
        print("=" * 80)
        print("=" * 80)
        print("AGGREGATE SUMMARY (All Directories)")
        print("=" * 80)
        print("=" * 80)
        print()
        
        method_counts = Counter([r['generation_method'] for r in all_results])
        for method, count in method_counts.most_common():
            print(f"  {method}: {count} statement(s)")
        
        print()
        print(f"Total selected statements across all directories: {len(all_results)}")
        
        # Show which statements are from survey/baseline
        survey_statements = [r for r in all_results if 'SURVEY/BASELINE' in r['generation_method']]
        if survey_statements:
            print()
            print("Note: Statements marked as 'SURVEY/BASELINE' are from the original survey")
            print("      and were not generated, so they don't have a generation method.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(__doc__)
        sys.exit(1)
    
    algorithm_name = sys.argv[1]
    directory_name = sys.argv[2] if len(sys.argv) == 3 else None
    
    check_statement_generation_methods(directory_name, algorithm_name)

