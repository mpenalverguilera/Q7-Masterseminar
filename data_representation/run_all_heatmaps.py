import subprocess
import sys
from pathlib import Path

# RQ2 scripts (challenges-focused analysis)
rq2_scripts = [
    'RQ2/01_heatmap_challenges_neural.py',
    'RQ2/02_heatmap_challenges_symbolic.py',
    'RQ2/03_heatmap_challenges_agent_configuration.py',
    'RQ2/04_heatmap_challenges_correction.py',
    'RQ2/05_heatmap_challenges_compliance.py',
    'RQ2/06_heatmap_challenges_memory.py',
]

# RQ3 scripts (approaches/techniques analysis)
rq3_scripts = [
    'RQ3/00_heatmap_neural_symbolic.py',
    'RQ3/01_heatmap_neural_memory.py',
    'RQ3/02_heatmap_symbolic_memory.py',
    'RQ3/03_heatmap_neural_compliance.py',
    'RQ3/04_heatmap_symbolic_compliance.py',
    'RQ3/05_bubble_self_correction_compliance.py',
]

qa_scripts = [
    'QA/qa_venues_barchart.py',
]

def run_scripts(scripts, label):
    """Run a list of scripts and track failures."""
    script_dir = Path(__file__).parent
    failed_scripts = []
    
    print(f"\n{'=' * 60}")
    print(f"Running {label} scripts...")
    print(f"{'=' * 60}")
    
    for script in scripts:
        script_path = script_dir / script
        
        if not script_path.exists():
            print(f"\n❌ {script} - NOT FOUND")
            failed_scripts.append(script)
            continue
        
        print(f"\n▶ Running {script}...")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✓ {script} - COMPLETED")
            else:
                print(f"❌ {script} - FAILED")
                print("STDERR:", result.stderr)
                failed_scripts.append(script)
        
        except subprocess.TimeoutExpired:
            print(f"❌ {script} - TIMEOUT")
            failed_scripts.append(script)
        except Exception as e:
            print(f"❌ {script} - ERROR: {e}")
            failed_scripts.append(script)
    
    return failed_scripts

if __name__ == "__main__":
    # Parse command line argument
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    
    if target not in ["qa", "rq2", "rq3", "all"]:
        print("Usage: python run_all_heatmaps.py [rq2|rq3|qa|all]")
        print("  rq2  - Run only RQ2 (challenges) scripts")
        print("  rq3  - Run only RQ3 (approaches) scripts")
        print("  qa   - Run only QA scripts")
        print("  all  - Run all scripts (default)")
        sys.exit(1)
    
    all_failed = []
    
    if target in ["qa", "all"]:
        failed = run_scripts(qa_scripts, "QA")
        all_failed.extend(failed)

    if target in ["rq2", "all"]:
        failed = run_scripts(rq2_scripts, "RQ2")
        all_failed.extend(failed)
    
    if target in ["rq3", "all"]:
        failed = run_scripts(rq3_scripts, "RQ3")
        all_failed.extend(failed)
    
    print("\n" + "=" * 60)
    if not all_failed:
        print("✓ All graphics regenerated successfully!")
    else:
        print(f"⚠ {len(all_failed)} script(s) failed:")
        for script in all_failed:
            print(f"  - {script}")
    print("=" * 60)
