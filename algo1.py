#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import requests
import time
import numpy as np
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple
from scipy import stats
from datetime import datetime, timedelta

# Constants
API_BASE = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "a47fcacd-00d4-4b8f-8a9d-821e4b69feed"
N = 1000  # Venue size
MAX_REJECTS = 20000  # Game fail limit
MIN_RUNS_PER_PARAM = 5  # Hardcoded minimum runs for grid search
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
        _session.headers.update({"User-Agent": "berghain_smart_optimizer/4.0"})
    return _session

# ========== API Functions with Rate Limiting ==========
def new_game(scenario: int, attempt: int = 0) -> Dict:
    """Start a new game with exponential backoff for rate limits"""
    url = f"{API_BASE}/new-game"
    params = {"scenario": scenario, "playerId": PLAYER_ID}
    sess = _get_session()
    
    for retry in range(3):
        try:
            resp = sess.get(url, params=params, timeout=10)
            if resp.status_code == 429:  # Rate limited
                wait = min(300, 10 * (2 ** attempt))
                log(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                return new_game(scenario, attempt + 1)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if retry == 2:
                raise
            time.sleep(1)

def decide_and_next(game_id: str, person_index: int, accept: Optional[bool] = None, attempt: int = 0) -> Dict:
    """Make decision and get next person with exponential backoff"""
    url = f"{API_BASE}/decide-and-next"
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        params["accept"] = str(accept).lower()
    
    sess = _get_session()
    for retry in range(3):
        try:
            resp = sess.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                wait = min(300, 10 * (2 ** attempt))
                log(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                return decide_and_next(game_id, person_index, accept, attempt + 1)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if retry == 2:
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
    """Core decision function with pressure rule and tuneable parameters"""
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
    if len(weights) != K:
        weights = [1.0] * K
    
    weighted_pressure = sum(
        (needs[k] / max(S, EPS)) * weights[k] 
        for k in range(K) 
        if needs[k] > 0 and not person_attrs[k]
    )
    
    # Adjustments
    threshold = params['threshold']
    
    # Early game bonus
    early_bonus = 0.0
    if S >= params.get('early_threshold', 800):
        early_bonus = params.get('early_bonus', 0.0)
    
    # All-attributes bonus
    ab_bonus = 0.0
    if K == 2 and all(person_attrs):
        ab_bonus = params.get('ab_bonus', 0.0)
    elif K > 2:
        needed_attrs = sum(1 for k in range(K) if needs[k] > 0)
        if needed_attrs > 0:
            coverage = sum(person_attrs[k] for k in range(K) if needs[k] > 0) / needed_attrs
            if coverage >= 0.5:
                ab_bonus = params.get('ab_bonus', 0.0) * coverage
    
    adjusted_pressure = weighted_pressure - early_bonus - ab_bonus
    
    return adjusted_pressure < threshold

# ========== Game Runner ==========
def run_game(args: Tuple[int, Dict]) -> Tuple[int, Dict, float]:
    """Run a single game and return (rejects, params, timestamp)"""
    scenario, params = args
    start_time = time.time()
    
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
            
            # Get next person
            resp = decide_and_next(game_id, person['personIndex'], accept)
            
            # Check termination
            if admitted >= N or rejected >= MAX_REJECTS:
                break
        
        # Return results
        if resp['status'] == 'completed':
            final_rejected = resp.get('rejectedCount', rejected)
            return final_rejected, params, start_time
        else:
            return MAX_REJECTS, params, start_time
            
    except Exception as e:
        log(f"Game error: {e}")
        return MAX_REJECTS, params, start_time

# ========== Parameter Management ==========
def param_hash(params: Dict) -> str:
    """Generate unique hash for parameter set"""
    # Create stable string representation
    key_parts = [
        f"t:{params['threshold']:.3f}",
        f"eb:{params.get('early_bonus', 0):.3f}",
        f"ab:{params.get('ab_bonus', 0):.3f}",
        f"et:{params.get('early_threshold', 800)}",
    ]
    weights = params.get('attr_weights', [])
    if weights:
        weights_str = ','.join(f"{w:.2f}" for w in weights)
        key_parts.append(f"w:[{weights_str}]")
    
    key = '|'.join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()[:12]

def generate_grid_params(K: int) -> List[Dict]:
    """Generate comprehensive grid search parameters"""
    params_list = []
    
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    early_bonuses = [0.0, 0.05, 0.1]
    ab_bonuses = [0.0, 0.1, 0.2] if K == 2 else [0.0, 0.05, 0.1]
    early_thresholds = [600, 700, 800, 900]
    
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

# ========== Statistical Analysis ==========
def calculate_statistics(runs: List[int]) -> Dict:
    """Calculate statistics for a set of runs"""
    if not runs:
        return {
            'mean': float('inf'),
            'median': float('inf'),
            'std': 0,
            'confidence_95': [float('inf'), float('inf')]
        }
    
    runs_array = np.array(runs)
    mean = np.mean(runs_array)
    median = np.median(runs_array)
    std = np.std(runs_array) if len(runs) > 1 else 0
    
    # 95% confidence interval
    if len(runs) >= 2:
        confidence = stats.t.interval(0.95, len(runs)-1, loc=mean, scale=std/np.sqrt(len(runs)))
        confidence_95 = list(confidence)
    else:
        confidence_95 = [mean - 2*std, mean + 2*std] if std > 0 else [mean, mean]
    
    return {
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'confidence_95': confidence_95
    }

# ========== State Manager ==========
class StateManager:
    def __init__(self, scenario: int):
        self.scenario = scenario
        self.state_file = f"berghain_s{scenario}_state.json"
        self.backup_file = f"berghain_s{scenario}_state.backup.json"
        self.last_backup = time.time()
        
    def load_state(self) -> Dict:
        """Load state with validation and recovery"""
        # Try main state file
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                if self._validate_state(state):
                    log(f"Loaded state: phase={state.get('phase', 'unknown')}")
                    return state
            except Exception as e:
                log(f"Main state corrupted: {e}")
        
        # Try backup
        if os.path.exists(self.backup_file):
            try:
                with open(self.backup_file, 'r') as f:
                    state = json.load(f)
                if self._validate_state(state):
                    log("Recovered from backup")
                    return state
            except Exception as e:
                log(f"Backup also corrupted: {e}")
        
        # Create fresh state
        log("Creating fresh state")
        return self._create_fresh_state()
    
    def save_state(self, state: Dict):
        """Save state with atomic write and periodic backup"""
        # Update last activity
        state['last_activity'] = time.time()
        
        # Atomic write using temp file
        temp_file = self.state_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, self.state_file)
        
        # Periodic backup (every hour)
        if time.time() - self.last_backup > 3600:
            with open(self.backup_file, 'w') as f:
                json.dump(state, f, indent=2)
            self.last_backup = time.time()
    
    def _validate_state(self, state: Dict) -> bool:
        """Validate state structure"""
        required_keys = ['scenario', 'grid_params', 'evaluations', 'phase']
        return all(key in state for key in required_keys)
    
    def _create_fresh_state(self) -> Dict:
        """Create fresh state structure"""
        # Get K from actual game
        game_data = new_game(self.scenario)
        K = len(game_data['constraints'])
        
        # Generate grid
        grid_params = generate_grid_params(K)
        
        return {
            'scenario': self.scenario,
            'K': K,
            'grid_params': grid_params,
            'evaluations': {},
            'champion': None,
            'phase': 'grid_search',
            'grid_completed': False,
            'last_activity': time.time(),
            'start_time': time.time(),
            'total_games': 0
        }

# ========== Smart Optimizer ==========
class SmartOptimizer:
    def __init__(self, scenario: int, workers: int = 4, target: Optional[int] = None):
        self.scenario = scenario
        self.workers = workers
        self.target = target
        self.state_manager = StateManager(scenario)
        self.state = self.state_manager.load_state()
        
        # Performance tracking
        self.start_time = self.state.get('start_time', time.time())
        self.games_per_hour = []
        
    def run_forever(self):
        """Main loop that runs indefinitely"""
        log(f"Smart Optimizer for Scenario {self.scenario}")
        log(f"Target: {self.target if self.target else 'Not set (will run forever)'}")
        log(f"Workers: {self.workers}")
        log("="*60)
        
        while True:
            try:
                # Update phase based on state
                self._update_phase()
                
                # Display progress
                self._display_progress()
                
                # Execute current phase
                if self.state['phase'] == 'grid_search':
                    self._run_grid_search()
                elif self.state['phase'] == 'validation':
                    self._run_validation()
                elif self.state['phase'] == 'improvement':
                    self._run_improvement()
                
                # Check target achievement
                if self.target and self.state.get('champion'):
                    best_run = min(self.state['champion'].get('all_runs', [float('inf')]))
                    if best_run <= self.target:
                        log(f"🎯 TARGET {self.target} ACHIEVED WITH {best_run}!")
                
                # Brief pause between batches
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                log("\nGracefully shutting down...")
                self.state_manager.save_state(self.state)
                break
            except Exception as e:
                log(f"Error in main loop: {e}")
                time.sleep(5)
    
    def _update_phase(self):
        """Determine current phase from state"""
        if not self.state['grid_completed']:
            # Check if grid is actually complete
            incomplete = self._find_incomplete_grid_params()
            if not incomplete:
                self.state['grid_completed'] = True
                self.state['phase'] = 'validation'
                log("Grid search complete, moving to validation phase")
            else:
                self.state['phase'] = 'grid_search'
        elif not self.state['champion']:
            self.state['phase'] = 'validation'
        else:
            self.state['phase'] = 'improvement'
    
    def _find_incomplete_grid_params(self) -> List[Tuple[Dict, str]]:
        """Find parameter sets that need more runs"""
        incomplete = []
        
        for params in self.state['grid_params']:
            p_hash = param_hash(params)
            
            # Initialize evaluation if not exists
            if p_hash not in self.state['evaluations']:
                self.state['evaluations'][p_hash] = {
                    'params': params,
                    'runs': [],
                    'timestamps': []
                }
            
            runs_completed = len(self.state['evaluations'][p_hash]['runs'])
            if runs_completed < MIN_RUNS_PER_PARAM:
                for _ in range(MIN_RUNS_PER_PARAM - runs_completed):
                    incomplete.append((params, p_hash))
        
        return incomplete
    
    def _run_grid_search(self):
        """Phase 1: Complete grid search with minimum runs"""
        incomplete = self._find_incomplete_grid_params()
        
        if not incomplete:
            self.state['grid_completed'] = True
            self.state_manager.save_state(self.state)
            return
        
        # Run batch
        batch_size = min(self.workers, len(incomplete))
        batch = incomplete[:batch_size]
        
        log(f"Grid search: Running {batch_size} games...")
        
        with Pool(self.workers) as pool:
            args = [(self.scenario, params) for params, _ in batch]
            results = pool.map(run_game, args)
        
        # Update state with results
        for (params, p_hash), (rejects, _, timestamp) in zip(batch, results):
            self.state['evaluations'][p_hash]['runs'].append(rejects)
            self.state['evaluations'][p_hash]['timestamps'].append(timestamp)
            
            # Update statistics
            stats = calculate_statistics(self.state['evaluations'][p_hash]['runs'])
            self.state['evaluations'][p_hash].update(stats)
            
            self.state['total_games'] = self.state.get('total_games', 0) + 1
            
            # Check if this is a new best
            if self._check_new_best(p_hash, rejects):
                log(f"  NEW GRID BEST: {rejects} (median: {stats['median']:.1f})")
        
        self.state_manager.save_state(self.state)
    
    def _run_validation(self):
        """Phase 2: Select champion through statistical validation"""
        # Get candidates with enough runs
        candidates = [
            (p_hash, eval_data)
            for p_hash, eval_data in self.state['evaluations'].items()
            if len(eval_data['runs']) >= MIN_RUNS_PER_PARAM
        ]
        
        if len(candidates) < 10:
            log("Not enough candidates for validation, continuing grid search...")
            self.state['grid_completed'] = False
            return
        
        # Sort by median performance
        candidates.sort(key=lambda x: x[1].get('median', float('inf')))
        
        best_hash, best_data = candidates[0]
        second_hash, second_data = candidates[1] if len(candidates) > 1 else (None, None)
        
        log(f"Best candidate: median={best_data['median']:.1f}, runs={len(best_data['runs'])}")
        
        # Check statistical significance
        if second_data is None or best_data['confidence_95'][1] < second_data['confidence_95'][0]:
            # Clear winner
            self.state['champion'] = {
                'param_hash': best_hash,
                'params': best_data['params'],
                'selection_date': time.time(),
                'all_runs': best_data['runs'].copy(),
                'best_single_run': min(best_data['runs']),
                'recent_performance': best_data['runs'][-20:]
            }
            log(f"CHAMPION SELECTED: median={best_data['median']:.1f}, best={min(best_data['runs'])}")
            log(f"  Params: {json.dumps(best_data['params'], indent=2)}")
            self.state['phase'] = 'improvement'
        else:
            # Too close, run more evaluations on top candidates
            log("Top candidates too close, running more evaluations...")
            
            # Run additional evaluations on top 5
            with Pool(self.workers) as pool:
                top_params = [candidates[i][1]['params'] for i in range(min(5, len(candidates)))]
                args = [(self.scenario, params) for params in top_params]
                results = pool.map(run_game, args)
            
            # Update evaluations
            for params, (rejects, _, timestamp) in zip(top_params, results):
                p_hash = param_hash(params)
                self.state['evaluations'][p_hash]['runs'].append(rejects)
                self.state['evaluations'][p_hash]['timestamps'].append(timestamp)
                stats = calculate_statistics(self.state['evaluations'][p_hash]['runs'])
                self.state['evaluations'][p_hash].update(stats)
                self.state['total_games'] = self.state.get('total_games', 0) + 1
        
        self.state_manager.save_state(self.state)
    
    def _run_improvement(self):
        """Phase 3: Run champion indefinitely seeking improvements"""
        if not self.state['champion']:
            log("No champion selected, returning to validation")
            self.state['phase'] = 'validation'
            return
        
        champion_params = self.state['champion']['params']
        
        # Run multiple games in parallel
        with Pool(self.workers) as pool:
            args = [(self.scenario, champion_params) for _ in range(self.workers)]
            results = pool.map(run_game, args)
        
        # Update champion stats
        for rejects, _, timestamp in results:
            self.state['champion']['all_runs'].append(rejects)
            self.state['champion']['recent_performance'].append(rejects)
            if len(self.state['champion']['recent_performance']) > 20:
                self.state['champion']['recent_performance'].pop(0)
            
            self.state['total_games'] = self.state.get('total_games', 0) + 1
            
            if rejects < self.state['champion']['best_single_run']:
                self.state['champion']['best_single_run'] = rejects
                total_runs = len(self.state['champion']['all_runs'])
                log(f"🚀 NEW BEST SCORE: {rejects} (run #{total_runs})")
                
                if self.target and rejects <= self.target:
                    log(f"🎯 TARGET {self.target} ACHIEVED WITH {rejects}!")
        
        # Periodic re-validation (every 100 champion runs)
        if len(self.state['champion']['all_runs']) % 100 == 0:
            log("Periodic champion re-validation...")
            # Could trigger mini-tournament here
        
        self.state_manager.save_state(self.state)
    
    def _check_new_best(self, p_hash: str, rejects: int) -> bool:
        """Check if this is a new best result"""
        current_best = float('inf')
        for eval_data in self.state['evaluations'].values():
            if eval_data['runs']:
                current_best = min(current_best, min(eval_data['runs']))
        return rejects < current_best
    
    def _display_progress(self):
        """Display current progress and statistics"""
        total_games = self.state.get('total_games', 0)
        elapsed = time.time() - self.start_time
        
        if total_games > 0:
            games_per_hour = total_games / (elapsed / 3600)
            
            if self.state['phase'] == 'grid_search':
                incomplete = len(self._find_incomplete_grid_params())
                total_needed = len(self.state['grid_params']) * MIN_RUNS_PER_PARAM
                completed = total_needed - incomplete
                progress_pct = (completed / total_needed) * 100
                eta_hours = incomplete / games_per_hour if games_per_hour > 0 else 0
                
                log(f"Grid Search: {completed}/{total_needed} runs ({progress_pct:.1f}%) | "
                    f"ETA: {eta_hours:.1f}h | {games_per_hour:.1f} games/h")
                
            elif self.state['phase'] == 'validation':
                log(f"Validation Phase | Total games: {total_games} | {games_per_hour:.1f} games/h")
                
            elif self.state['phase'] == 'improvement':
                champion = self.state['champion']
                if champion:
                    recent = champion['recent_performance']
                    if recent:
                        recent_median = np.median(recent)
                        log(f"Improvement: Best={champion['best_single_run']} | "
                            f"Recent median={recent_median:.1f} | "
                            f"Champion runs={len(champion['all_runs'])} | "
                            f"{games_per_hour:.1f} games/h")

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Smart Berghain Optimizer v4.0")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3],
                       help="Scenario to optimize (required for --target)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel workers for games")
    parser.add_argument("--target", type=int, default=None,
                       help="Target rejection count (requires --scenario)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target and not args.scenario:
        parser.error("--target requires --scenario")
    
    # Default to scenario 1 if not specified
    if not args.scenario:
        args.scenario = 1
        log("No scenario specified, defaulting to Scenario 1")
    
    # Run optimizer
    optimizer = SmartOptimizer(args.scenario, args.workers, args.target)
    optimizer.run_forever()

if __name__ == "__main__":
    main()