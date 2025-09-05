#!/usr/bin/env python3

import argparse
import json
import os
import random
import requests
import time
import numpy as np
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

# Constants
API_BASE = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "a47fcacd-00d4-4b8f-8a9d-821e4b69feed"
N = 1000  # Venue size
MAX_REJECTS = 20000  # Game fail limit
EPS = 1e-9

# Logging with timestamp
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ========== Session Management (per-process) ==========
_session = None

def _get_session():
    """Get or create a session for this process"""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "berghain_empirical/3.0"})
    return _session

# ========== API Functions ==========
def new_game(scenario: int) -> Dict:
    """Start a new game"""
    url = f"{API_BASE}/new-game"
    params = {"scenario": scenario, "playerId": PLAYER_ID}
    sess = _get_session()
    
    for attempt in range(3):  # Retry logic
        try:
            resp = sess.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1)

def decide_and_next(game_id: str, person_index: int, accept: Optional[bool] = None) -> Dict:
    """Make decision and get next person"""
    url = f"{API_BASE}/decide-and-next"
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        params["accept"] = str(accept).lower()
    
    sess = _get_session()
    for attempt in range(3):  # Retry logic
        try:
            resp = sess.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1)

# ========== Decision Function ==========
def decide(
    constraints: List[Dict],
    admitted_count: int,
    next_person: Dict,
    accepted_count: Dict,
    params: Dict
) -> bool:
    """
    Core decision function with pressure rule and tuneable parameters
    """
    S = N - admitted_count
    K = len(constraints)
    
    # Calculate needs
    needs = []
    person_attrs = []
    for c in constraints:
        attr = c['attribute']
        needs.append(max(0, c['minCount'] - accepted_count.get(attr, 0)))
        person_attrs.append(next_person['attributes'].get(attr, False) if next_person else False)
    
    # Guard 1: Greedy finish if all mins met
    if all(n <= 0 for n in needs):
        return True
    
    # Guard 2: Must accept if critical for any attr
    if any(needs[k] == S and person_attrs[k] for k in range(K)):
        return True
    
    # Guard 3: Must reject if infeasible
    sum_missing = sum(needs[k] / max(S, EPS) for k in range(K) if needs[k] > 0 and not person_attrs[k])
    if sum_missing >= 1.0 - EPS:
        return False
    
    # Core pressure calculation with weights
    weights = params.get('attr_weights', [1.0] * K)
    if len(weights) != K:  # Safety check
        weights = [1.0] * K
    
    weighted_pressure = sum(
        (needs[k] / max(S, EPS)) * weights[k] 
        for k in range(K) 
        if needs[k] > 0 and not person_attrs[k]
    )
    
    # Adjustments
    threshold = params['threshold']
    
    # Early game bonus (FIXED: use >= instead of >)
    early_bonus = 0.0
    if S >= params.get('early_threshold', 800):
        early_bonus = params.get('early_bonus', 0.0)
    
    # All-attributes bonus (for K=2 mainly, but works for any K)
    ab_bonus = 0.0
    if K == 2 and all(person_attrs):
        ab_bonus = params.get('ab_bonus', 0.0)
    elif K > 2:
        # For K>2, bonus based on coverage ratio
        needed_attrs = sum(1 for k in range(K) if needs[k] > 0)
        if needed_attrs > 0:
            coverage = sum(person_attrs[k] for k in range(K) if needs[k] > 0) / needed_attrs
            if coverage >= 0.5:  # Has at least half of needed attributes
                ab_bonus = params.get('ab_bonus', 0.0) * coverage
    
    adjusted_pressure = weighted_pressure - early_bonus - ab_bonus
    
    return adjusted_pressure < threshold

# ========== Game Runner ==========
def run_game(args: Tuple[int, Dict]) -> Tuple[int, Dict]:
    """
    Run a single game with given parameters
    Returns (rejections, params_used)
    """
    scenario, params = args
    
    try:
        # Start game
        game_data = new_game(scenario)
        game_id = game_data['gameId']
        constraints = game_data['constraints']
        
        # Initialize state
        accepted_count = {c['attribute']: 0 for c in constraints}
        admitted = 0
        rejected = 0
        
        # Get first person
        resp = decide_and_next(game_id, 0)
        
        # Main game loop
        while resp['status'] == 'running':
            person = resp.get('nextPerson')
            if not person:
                break
            
            # Make decision
            accept = decide(constraints, admitted, person, accepted_count, params)
            
            # Update counts
            if accept:
                admitted += 1
                for c in constraints:
                    attr = c['attribute']
                    if person['attributes'].get(attr, False):
                        accepted_count[attr] += 1
            else:
                rejected += 1
            
            # Progress logging every 100 decisions (optional)
            if (admitted + rejected) % 500 == 0:
                acc_rate = admitted / (admitted + rejected)
                log(f"  Game progress: S={N-admitted}, acc={admitted}, rej={rejected}, rate={acc_rate:.3f}")
            
            # Get next person
            resp = decide_and_next(game_id, person['personIndex'], accept)
            
            # Check termination
            if admitted >= N or rejected >= MAX_REJECTS:
                break
        
        # Return results
        if resp['status'] == 'completed':
            final_rejected = resp.get('rejectedCount', rejected)
            return final_rejected, params
        else:
            # Failed - penalize
            return MAX_REJECTS, params
            
    except Exception as e:
        log(f"Game error: {e}")
        return MAX_REJECTS, params

# ========== Parameter Generation ==========
def generate_grid_params(K: int) -> List[Dict]:
    """Generate grid search parameters"""
    params_list = []
    
    # Extended threshold range to explore lower values
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    early_bonuses = [0.0, 0.05, 0.1]
    ab_bonuses = [0.0, 0.1, 0.2] if K == 2 else [0.0, 0.05, 0.1]
    early_thresholds = [600, 700, 800, 900]  # Keep <= 1000
    
    for t in thresholds:
        for eb in early_bonuses:
            for ab in ab_bonuses:
                for et in early_thresholds:
                    params = {
                        'threshold': t,
                        'early_bonus': eb,
                        'ab_bonus': ab,
                        'early_threshold': et,
                        'attr_weights': [1.0] * K
                    }
                    params_list.append(params)
    
    return params_list

def perturb_params(params: Dict, K: int, noise_scale: float) -> Dict:
    """Perturb parameters with gaussian noise"""
    new_params = params.copy()
    
    # Core parameters
    new_params['threshold'] += random.gauss(0, noise_scale * 0.1)
    new_params['threshold'] = max(0.5, min(1.0, new_params['threshold']))
    
    new_params['early_bonus'] += random.gauss(0, noise_scale * 0.05)
    new_params['early_bonus'] = max(0.0, min(0.2, new_params['early_bonus']))
    
    new_params['ab_bonus'] += random.gauss(0, noise_scale * 0.05)
    new_params['ab_bonus'] = max(0.0, min(0.3, new_params['ab_bonus']))
    
    new_params['early_threshold'] += int(random.gauss(0, noise_scale * 100))
    new_params['early_threshold'] = max(500, min(1000, new_params['early_threshold']))  # Allow up to 1000
    
    # Attribute weights - explore [0.8, 1.2] range typically
    weights = params.get('attr_weights', [1.0] * K).copy()
    for i in range(K):
        weights[i] += random.gauss(0, noise_scale * 0.1)
        weights[i] = max(0.5, min(1.5, weights[i]))
    new_params['attr_weights'] = weights
    
    return new_params

# ========== Main Tuner ==========
class Tuner:
    def __init__(self, scenario: int, workers: int = 4):
        self.scenario = scenario
        self.workers = workers
        self.state_file = f"berghain_s{scenario}_best.json"
        
        # Determine K from actual game constraints (FIXED: don't hardcode)
        self.K = None
        self.best_params = None
        self.best_rejects = float('inf')
        self.history = []
        
        # Load previous best if exists
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.best_params = state.get('best_params')
                self.best_rejects = state.get('best_rejects', float('inf'))
                self.history = state.get('history', [])
                self.K = state.get('K')  # Try to load K
                log(f"Loaded previous best: {self.best_rejects} rejects")
        
        # Get K from actual game if not loaded
        if self.K is None:
            log("Detecting number of attributes from game...")
            game_data = new_game(self.scenario)
            self.K = len(game_data['constraints'])
            log(f"Scenario {self.scenario} has K={self.K} attributes")
        
        # Initialize default params if needed
        if self.best_params is None:
            self.best_params = {
                'threshold': 0.9,
                'early_bonus': 0.05,
                'ab_bonus': 0.1 if self.K == 2 else 0.05,
                'early_threshold': 800,
                'attr_weights': [1.0] * self.K
            }
    
    def save_state(self):
        """Save current best to file (FIXED: save all params)"""
        state = {
            'scenario': self.scenario,
            'K': self.K,
            'best_params': self.best_params,
            'best_rejects': self.best_rejects,
            'history': self.history[-100:]  # Keep last 100 runs
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def tune(self, max_runs: int = 500, target_rejects: int = 750):
        """Main tuning loop with parallel execution (FIXED: process whole grid)"""
        log(f"Starting tuner for scenario {self.scenario} (K={self.K})")
        log(f"Target: {target_rejects} rejects, Max runs: {max_runs}")
        
        # Generate full grid upfront
        grid_params = generate_grid_params(self.K)
        grid_idx = 0
        log(f"Grid search phase: {len(grid_params)} parameter sets")
        
        runs_completed = 0
        consecutive_no_improvement = 0
        
        while runs_completed < max_runs and self.best_rejects > target_rejects:
            # Determine batch of parameters to test
            if grid_idx < len(grid_params):
                # Grid search phase (FIXED: process entire grid)
                batch_params = grid_params[grid_idx : min(grid_idx + self.workers, len(grid_params))]
                grid_idx += len(batch_params)
                
                if not batch_params:  # Grid exhausted
                    log(f"Grid search complete after {grid_idx} parameter sets")
                    continue
                    
                if grid_idx % 50 == 0:
                    log(f"Grid progress: {grid_idx}/{len(grid_params)} parameter sets tested")
            else:
                # Perturbation phase
                noise_scale = 0.5 * max(0.1, 1 - runs_completed / max_runs)  # Keep min noise
                batch_params = []
                for _ in range(min(self.workers, max_runs - runs_completed)):
                    perturbed = perturb_params(self.best_params, self.K, noise_scale)
                    batch_params.append(perturbed)
                
                if not batch_params:
                    break
            
            # Run games in parallel
            with Pool(self.workers) as pool:
                args = [(self.scenario, params) for params in batch_params]
                results = pool.map(run_game, args)
            
            # Process results
            improved = False
            for rejects, params in results:
                runs_completed += 1
                
                # Track history (FIXED: save all params)
                self.history.append({
                    'run': runs_completed,
                    'rejects': rejects,
                    'params': params,  # Save complete params
                    'timestamp': time.time()
                })
                
                # Update best
                if rejects < self.best_rejects:
                    self.best_rejects = rejects
                    self.best_params = params.copy()
                    improved = True
                    consecutive_no_improvement = 0
                    log(f"[Run {runs_completed}] NEW BEST: {rejects} rejects")
                    log(f"  Params: threshold={params['threshold']:.3f}, "
                        f"early_bonus={params['early_bonus']:.3f}, "
                        f"ab_bonus={params['ab_bonus']:.3f}, "
                        f"early_threshold={params['early_threshold']}")
                    if self.K > 2:
                        weights_str = ', '.join([f"{w:.2f}" for w in params['attr_weights']])
                        log(f"  Weights: [{weights_str}]")
                    self.save_state()
                else:
                    if runs_completed % 20 == 0:
                        log(f"[Run {runs_completed}] Current best: {self.best_rejects}")
            
            if not improved:
                consecutive_no_improvement += len(batch_params)
                
                # Early stopping if stuck
                if consecutive_no_improvement >= 100 and runs_completed > 200:
                    log("No improvement for 100 runs, stopping early")
                    break
        
        # Final report
        log(f"\n=== TUNING COMPLETE ===")
        log(f"Total runs: {runs_completed}")
        log(f"Best rejects: {self.best_rejects}")
        log(f"Best params: {json.dumps(self.best_params, indent=2)}")
        
        return self.best_params, self.best_rejects

# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="Empirical Berghain Optimizer v3")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3],
                       help="Scenario to optimize")
    parser.add_argument("--mode", choices=["single", "tune", "all"], default="tune",
                       help="Mode: single run, tune one scenario, or all scenarios")
    parser.add_argument("--max-runs", type=int, default=500,
                       help="Maximum tuning runs per scenario")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel workers for games")
    parser.add_argument("--target", type=int, default=None,
                       help="Target rejection count (default: 750/3100/3900)")
    
    args = parser.parse_args()
    
    # Default targets per scenario (adjusted based on leaderboard info)
    default_targets = {1: 720, 2: 3100, 3: 3900}
    
    if args.mode == "single":
        # Run single game with best known params
        tuner = Tuner(args.scenario, workers=1)
        if tuner.best_params is not None:
            params = tuner.best_params
            log(f"Running with best params: {params}")
        else:
            params = {
                'threshold': 0.9,
                'early_bonus': 0.05,
                'ab_bonus': 0.1 if tuner.K == 2 else 0.05,
                'early_threshold': 800,
                'attr_weights': [1.0] * tuner.K
            }
            log(f"Running with default params: {params}")
        
        rejects, _ = run_game((args.scenario, params))
        log(f"Result: {rejects} rejects")
    
    elif args.mode == "tune":
        # Tune single scenario
        target = args.target or default_targets[args.scenario]
        tuner = Tuner(args.scenario, workers=args.workers)
        tuner.tune(max_runs=args.max_runs, target_rejects=target)
    
    else:  # mode == "all"
        # Tune all scenarios sequentially
        for scenario in [1, 2, 3]:
            log(f"\n{'='*50}")
            log(f"SCENARIO {scenario}")
            log(f"{'='*50}")
            
            target = default_targets[scenario]
            max_runs = 1000 if scenario == 1 else 500  # More runs for S1
            
            tuner = Tuner(scenario, workers=args.workers)
            tuner.tune(max_runs=max_runs, target_rejects=target)
            
            # Brief pause between scenarios
            time.sleep(2)

if __name__ == "__main__":
    main()