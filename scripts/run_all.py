#!/usr/bin/env python3
import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

def run_script(script_name, description):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("="*70)
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        capture_output=False
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n[OK] {description} completed in {elapsed:.1f}s")
    else:
        print(f"\n[FAIL] {description} failed (exit code: {result.returncode})")
        return False
    
    return True

def main():
    print("="*70)
    print("SOUTHEAST ASIA AIR QUALITY ANALYSIS")
    print("Complete Pipeline Execution")
    print("="*70)
    
    start_total = time.time()
    
    scripts = [
        ("download_data.py", "Data Download / Generation"),
        ("preprocess.py", "Data Preprocessing"),
        ("eda.py", "Exploratory Data Analysis"),
        ("classification.py", "Classification Models"),
        ("clustering.py", "Clustering Analysis"),
        ("association.py", "Association Rule Mining"),
        ("anomaly_pca.py", "Anomaly Detection & PCA")
    ]
    
    results = []
    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\nPipeline stopped due to failure in: {description}")
            break
    
    total_time = time.time() - start_total
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for description, success in results:
        status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
        print(f"  {status}: {description}")
    
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    figures_dir = os.path.join(BASE_DIR, 'figures')
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"\nGenerated {len(figures)} visualization files in: {figures_dir}")
    
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    if os.path.exists(processed_dir):
        data_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        print(f"Generated {len(data_files)} data files in: {processed_dir}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
