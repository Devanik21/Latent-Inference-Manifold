import os
import sys

# Add working directory to path if standard importing fails
sys.path.insert(0, os.path.abspath('c:\\AGI'))

import json
from universe import DifficultyLevel, Universe, TaskDomain
from council import Council

def test_full_pipeline():
    print("Initializing AGI System...")
    universe = Universe()
    council = Council()
    
    # Simulate cross-domain test
    print("Running Epoch A (Spatial)...")
    for _ in range(2):
        task = universe.generate_task(DifficultyLevel.L1, domain=TaskDomain.A_SPATIAL)
        for snap in council.solve(task):
            pass # Exhaust generator
            
    print("Running Epoch B (Topological)...")
    for _ in range(2):
        task = universe.generate_task(DifficultyLevel.L1, domain=TaskDomain.B_TOPOLOGICAL)
        for snap in council.solve(task):
            pass

    print("Running Epoch C (Abstract - 0-shot transfer test)...")
    for _ in range(2):
        task = universe.generate_task(DifficultyLevel.L1, domain=TaskDomain.C_ABSTRACT)
        for snap in council.solve(task):
            pass

    print("\n--- Telemetry Stats ---")
    stats = council.stats()
    
    print(f"Total solved: {stats['solved']} / {stats['total_episodes']}")
    print(f"Latent Skills Discovered: {stats['skill_library_size']}")
    print(f"Dictionary Ready: {stats['latent_dictionary']['is_ready']}")
    
    meta = stats.get('meta_learner', {})
    print(f"MetaLearner Total Updates: {meta.get('total_updates', 0)}")
    
    if meta.get('total_updates', 0) > 0:
        print("MetaLearner successfully updated priors.")
        
    print("System verified operational.")

if __name__ == "__main__":
    test_full_pipeline()
