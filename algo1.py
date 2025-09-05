#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import os
import random
import requests
import signal
import sys
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
    """Get or create a session for this process with connection pooling"""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "berghain_smart_optimizer/4.0"})
        
        # Add connection pooling for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            pool_block=False,
            max_retries=0  # We handle retries manually
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
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
            time.sleep(0.2)  # Reduced from 1s for faster retries

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
            time.sleep(0.2)  # Reduced from 1s for faster retries

# ========== Decision Function ==========
def decide(
    constraints: List[Dict],
    admitted_count: int,
    next_person: Dict,
    accepted_count: Dict,
    params: Dict
) -> bool:
    """Core decision function (optimized with numpy and early returns)"""
    S = N - admitted_count
    
    # Early return for common cases
    if S <= 0:
        return False
    
    K = len(constraints)
    
    # Pre-calculate needs array using numpy for speed
    needs = np.array([max(0, c['minCount'] - accepted_count.get(c['attribute'], 0)) 
                      for c in constraints])
    
    # Fast path: all mins met
    if not needs.any():
        return True
    
    # Extract person attributes
    person_attrs = np.array([next_person['attributes'].get(c['attribute'], False) 
                              for c in constraints]) if next_person else np.zeros(K, dtype=bool)
    
    # Guard 2: Must accept if critical for any attr
    if np.any((needs == S) & person_attrs):
        return True
    
    # Guard 3: Must reject if infeasible
    missing_mask = (needs > 0) & (~person_attrs)
    sum_missing = np.sum(needs[missing_mask] / max(S, EPS))
    if sum_missing >= 1.0 - EPS:
        return False
    
    # Core pressure calculation with weights
    weights = np.array(params.get('attr_weights', [1.0] * K))
    if len(weights) != K:
        weights = np.ones(K)
    
    weighted_pressure = np.sum((needs[missing_mask] / max(S, EPS)) * weights[missing_mask])
    
    # Adjustments
    threshold = params['threshold']
    
    # Early game bonus
    early_bonus = params.get('early_bonus', 0.0) if S >= params.get('early_threshold', 800) else 0.0
    
    # All-attributes bonus
    ab_bonus = 0.0
    if K == 2 and person_attrs.all():
        ab_bonus = params.get('ab_bonus', 0.0)
    elif K > 2:
        needed_mask = needs > 0
        needed_count = np.sum(needed_mask)
        if needed_count > 0:
            coverage = np.sum(person_attrs[needed_mask]) / needed_count
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
    """Generate unique hash for parameter set (optimized with built-in hash)"""
    # Use tuple hash instead of MD5 for speed
    key = (
        round(params['threshold'], 3),
        round(params.get('early_bonus', 0), 3),
        round(params.get('ab_bonus', 0), 3),
        params.get('early_threshold', 800),
        tuple(round(w, 2) for w in params.get('attr_weights', []))
    )
    # Use Python's built-in hash and make it positive
    return str(abs(hash(key)))[:12]

def generate_grid_params(K: int) -> List[Dict]:
    """Generate comprehensive grid search parameters (with caching)"""
    # Check cache first
    cache_key = f"_grid_params_K{K}"
    if hasattr(generate_grid_params, cache_key):
        return getattr(generate_grid_params, cache_key).copy()
    
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
    
    # Cache result
    setattr(generate_grid_params, cache_key, params_list)
    return params_list

# ========== Statistical Analysis ==========
def calculate_statistics(runs: List[int], detailed: bool = True) -> Dict:
    """Calculate statistics for a set of runs (optimized with detailed flag)"""
    if not runs:
        return {
            'mean': float('inf'),
            'median': float('inf'),
            'std': 0,
            'confidence_95': [float('inf'), float('inf')]
        }
    
    runs_array = np.array(runs)
    median = np.median(runs_array)
    
    # Fast path: only calculate median when not detailed
    if not detailed:
        return {'median': float(median)}
    
    # Full statistics for detailed mode
    mean = np.mean(runs_array)
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
    def __init__(self, scenario: int, workers: int = 4, target: Optional[int] = None, debug: bool = False, quiet: bool = False):
        self.scenario = scenario
        self.workers = workers
        self.target = target
        self.debug = debug
        self.quiet = quiet
        self.state_manager = StateManager(scenario)
        self.state = self.state_manager.load_state()
        self.pool = None  # Track pool for cleanup
        self.shutdown_requested = False
        
        # Performance tracking
        self.start_time = self.state.get('start_time', time.time())
        self.games_per_hour = []
        
        # Batch saving optimization
        self.games_since_save = 0
        self.save_interval = 10  # Save every 10 games
        
        # Create persistent pool (optimization)
        if self.workers > 1:
            self.pool = Pool(self.workers)
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Migrate old state format if needed
        self._migrate_state_if_needed()
    
    def __del__(self):
        """Cleanup pools on destruction"""
        self._cleanup_pool()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.shutdown_requested = True
        if self.pool:
            self.pool.terminate()  # Immediately terminate pool
        if not self.quiet:
            print(f"\n[{time.strftime('%H:%M:%S')}] 🛑 Shutdown requested, cleaning up...", flush=True)
    
    def _cleanup_pool(self):
        """Clean up any active pool"""
        if self.pool:
            try:
                self.pool.terminate()
                self.pool.join(timeout=2)
            except:
                pass
            finally:
                self.pool = None
    
    def _save_state_batch(self, force: bool = False, is_best: bool = False):
        """Save state with batching (reduces I/O overhead)"""
        self.games_since_save += 1
        
        # Save immediately if: forced, new best, or interval reached
        if force or is_best or self.games_since_save >= self.save_interval:
            self.state_manager.save_state(self.state)
            self.games_since_save = 0
    
    def _migrate_state_if_needed(self):
        """Migrate old state format to new bounded format"""
        migrated = False
        
        # Migrate evaluations
        for p_hash, eval_data in self.state.get('evaluations', {}).items():
            if 'runs' in eval_data and 'run_summary' not in eval_data:
                runs = eval_data['runs']
                eval_data['run_summary'] = {
                    'last_100': runs[-100:] if len(runs) > 100 else runs.copy(),
                    'total_count': len(runs),
                    'best_score': min(runs) if runs else float('inf'),
                    'percentiles': self._calculate_percentiles(runs) if runs else {}
                }
                del eval_data['runs']
                migrated = True
        
        # Migrate champion
        if self.state.get('champion') and 'all_runs' in self.state['champion']:
            all_runs = self.state['champion']['all_runs']
            self.state['champion']['run_summary'] = {
                'last_500': all_runs[-500:] if len(all_runs) > 500 else all_runs.copy(),
                'total_count': len(all_runs),
                'best_score': min(all_runs) if all_runs else float('inf'),
                'percentiles': self._calculate_percentiles(all_runs) if all_runs else {}
            }
            del self.state['champion']['all_runs']
            migrated = True
        
        # Track global best
        if 'global_best' not in self.state:
            self.state['global_best'] = float('inf')
            for eval_data in self.state.get('evaluations', {}).values():
                if 'run_summary' in eval_data:
                    self.state['global_best'] = min(self.state['global_best'], eval_data['run_summary']['best_score'])
            migrated = True
        
        if migrated:
            if not self.quiet:
                log("Migrated state to new bounded format")
            self._save_state_batch(force=True)
    
    def _calculate_percentiles(self, runs: List[int]) -> Dict:
        """Calculate percentiles for a list of runs"""
        if not runs:
            return {}
        arr = np.array(runs)
        return {
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95))
        }
        
    def run_forever(self):
        """Main loop that runs indefinitely"""
        if not self.quiet:
            log(f"Smart Optimizer for Scenario {self.scenario}")
            log(f"Target: {self.target if self.target else 'Not set (will run forever)'}")
            log(f"Workers: {self.workers}")
            if self.debug:
                log(f"Debug mode: ON")
            log("="*60)
        
        try:
            while not self.shutdown_requested:
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
                        champion_summary = self.state['champion'].get('run_summary')
                        if champion_summary:
                            best_run = champion_summary.get('best_score', float('inf'))
                            if best_run <= self.target:
                                if not self.quiet:
                                    log(f"🎯 TARGET {self.target} ACHIEVED WITH {best_run}!")
                    
                    # Check for shutdown
                    if self.shutdown_requested:
                        break
                    
                    # Minimal pause between batches (removed to maximize throughput)
                    # time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    self.shutdown_requested = True
                    break
                except Exception as e:
                    log(f"Error in main loop: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    time.sleep(5)
        finally:
            # Cleanup
            if self.pool:
                try:
                    self.pool.terminate()  # Force terminate
                    self.pool.join(timeout=1)
                except:
                    pass
            
            try:
                self._save_state_batch(force=True)  # Force save on shutdown
                if not self.quiet:
                    print(f"\n[{time.strftime('%H:%M:%S')}] ✅ State saved. Goodbye!", flush=True)
            except:
                pass
            
            os._exit(0)  # Force exit
    
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
                    'run_summary': {
                        'last_100': [],
                        'total_count': 0,
                        'best_score': float('inf'),
                        'percentiles': {}
                    },
                    'timestamps': []
                }
            
            # Handle both old and new format
            eval_data = self.state['evaluations'][p_hash]
            if 'run_summary' in eval_data:
                runs_completed = eval_data['run_summary']['total_count']
            else:
                # Old format compatibility
                runs_completed = len(eval_data.get('runs', []))
            
            if runs_completed < MIN_RUNS_PER_PARAM:
                for _ in range(MIN_RUNS_PER_PARAM - runs_completed):
                    incomplete.append((params, p_hash))
        
        return incomplete
    
    def _run_grid_search(self):
        """Phase 1: Complete grid search with minimum runs"""
        incomplete = self._find_incomplete_grid_params()
        
        if not incomplete:
            self.state['grid_completed'] = True
            self._save_state_batch(force=True)
            return
        
        # Run batch
        batch_size = min(self.workers, len(incomplete))
        batch = incomplete[:batch_size]
        
        # Check if we're testing new parameters
        if batch and self.state.get('last_params_hash') != batch[0][1]:
            self.state['last_params_hash'] = batch[0][1]
            if not self.quiet:
                log(f"Grid search: Testing parameters (hash: {batch[0][1][:6]}...)")
        
        if not self.quiet:
            log(f"Grid search: Running {batch_size} games...")
        
        # Use pool properly with async
        try:
            results = []
            
            # Use existing pool or run sequentially if workers=1
            if self.workers == 1:
                # Sequential execution for single worker
                for params, _ in batch:
                    if self.shutdown_requested:
                        break
                    results.append(run_game((self.scenario, params)))
                if self.shutdown_requested:
                    return
            else:
                # Parallel execution with persistent pool
                if not self.pool:
                    self.pool = Pool(self.workers)
                
                args = [(self.scenario, params) for params, _ in batch]
                
                # Use map_async instead of map
                async_result = self.pool.map_async(run_game, args)
                
                # Wait with timeout, checking for shutdown
                while not self.shutdown_requested:
                    try:
                        results = async_result.get(timeout=0.5)
                        break
                    except multiprocessing.TimeoutError:
                        if async_result.ready():
                            results = async_result.get()
                            break
                        continue
                
                if self.shutdown_requested:
                    if self.pool:
                        self.pool.terminate()
                        self.pool = None
                    return
                
        except (KeyboardInterrupt, SystemExit):
            self.shutdown_requested = True
            if self.pool:
                self.pool.terminate()
            return
        finally:
            # Keep pool alive for reuse (optimization)
            pass
        
        # Update state with results
        for (params, p_hash), (rejects, _, timestamp) in zip(batch, results):
            eval_data = self.state['evaluations'][p_hash]
            run_summary = eval_data['run_summary']
            
            # Add to bounded history
            run_summary['last_100'].append(rejects)
            if len(run_summary['last_100']) > 100:
                run_summary['last_100'].pop(0)
            
            run_summary['total_count'] += 1
            run_summary['best_score'] = min(run_summary['best_score'], rejects)
            
            # Update percentiles
            if len(run_summary['last_100']) >= 5:
                run_summary['percentiles'] = self._calculate_percentiles(run_summary['last_100'])
            
            # Update timestamps
            eval_data['timestamps'].append(timestamp)
            if len(eval_data['timestamps']) > 100:
                eval_data['timestamps'].pop(0)
            
            # Update statistics (full stats for validation)
            stats = calculate_statistics(run_summary['last_100'], detailed=True)
            eval_data.update(stats)
            
            self.state['total_games'] = self.state.get('total_games', 0) + 1
            
            # Check for new global best
            is_new_global_best = rejects < self.state.get('global_best', float('inf'))
            if is_new_global_best:
                self.state['global_best'] = rejects
                log(f"🔥 NEW GLOBAL BEST: {rejects} rejections!")
                if self.debug:
                    log(f"   Params: {json.dumps(params, indent=2)}")
            
            # Check for new phase best
            phase_best_key = f'{self.state["phase"]}_best'
            if rejects < self.state.get(phase_best_key, float('inf')):
                self.state[phase_best_key] = rejects
                if not self.quiet:
                    log(f"  NEW GRID BEST: {rejects} (median: {stats['median']:.1f})")
            
            # Use batch saving (save on global best or every N games)
            self._save_state_batch(is_best=is_new_global_best)
    
    def _run_validation(self):
        """Phase 2: Select champion through statistical validation"""
        # Get candidates with enough runs
        candidates = []
        for p_hash, eval_data in self.state['evaluations'].items():
            if 'run_summary' in eval_data:
                if eval_data['run_summary']['total_count'] >= MIN_RUNS_PER_PARAM:
                    candidates.append((p_hash, eval_data))
            elif 'runs' in eval_data:
                # Old format compatibility
                if len(eval_data['runs']) >= MIN_RUNS_PER_PARAM:
                    candidates.append((p_hash, eval_data))
        
        if len(candidates) < 10:
            if not self.quiet:
                log("Not enough candidates for validation, continuing grid search...")
            self.state['grid_completed'] = False
            return
        
        # Sort by median performance
        candidates.sort(key=lambda x: x[1].get('median', float('inf')))
        
        best_hash, best_data = candidates[0]
        second_hash, second_data = candidates[1] if len(candidates) > 1 else (None, None)
        
        # Get run count properly
        if 'run_summary' in best_data:
            run_count = best_data['run_summary']['total_count']
            best_score = best_data['run_summary']['best_score']
            runs_for_champion = best_data['run_summary']['last_100'].copy()
        else:
            run_count = len(best_data['runs'])
            best_score = min(best_data['runs'])
            runs_for_champion = best_data['runs'].copy()
        
        if not self.quiet:
            log(f"Best candidate: median={best_data['median']:.1f}, runs={run_count}")
        
        # Check statistical significance
        if second_data is None or best_data['confidence_95'][1] < second_data['confidence_95'][0]:
            # Clear winner
            self.state['champion'] = {
                'param_hash': best_hash,
                'params': best_data['params'],
                'selection_date': time.time(),
                'run_summary': {
                    'last_500': runs_for_champion[-500:],
                    'total_count': len(runs_for_champion),
                    'best_score': best_score,
                    'percentiles': self._calculate_percentiles(runs_for_champion)
                },
                'recent_performance': runs_for_champion[-20:]
            }
            log(f"🏆 CHAMPION SELECTED: median={best_data['median']:.1f}, best={best_score}")
            if not self.quiet or self.debug:
                log(f"  Params: {json.dumps(best_data['params'], indent=2)}")
            self.state['phase'] = 'improvement'
        else:
            # Too close, run more evaluations on top candidates
            if not self.quiet:
                log("Top candidates too close, running more evaluations...")
            
            # Run additional evaluations on top 5
            try:
                # Use existing pool or create if needed
                if not self.pool and self.workers > 1:
                    self.pool = Pool(self.workers)
                
                top_params = [candidates[i][1]['params'] for i in range(min(5, len(candidates)))]
                
                # Handle single worker case
                if self.workers == 1:
                    results = []
                    for params in top_params:
                        if self.shutdown_requested:
                            break
                        results.append(run_game((self.scenario, params)))
                    if self.shutdown_requested:
                        return
                else:
                    args = [(self.scenario, params) for params in top_params]
                    
                    # Use map_async instead of map
                    async_result = self.pool.map_async(run_game, args)
                    results = []
                    
                    # Wait with timeout, checking for shutdown
                    while not self.shutdown_requested:
                        try:
                            results = async_result.get(timeout=0.5)
                            break
                        except multiprocessing.TimeoutError:
                            if async_result.ready():
                                results = async_result.get()
                                break
                            continue
                    
                    if self.shutdown_requested:
                        if self.pool:
                            self.pool.terminate()
                            self.pool = None
                        return
                    
            except (KeyboardInterrupt, SystemExit):
                self.shutdown_requested = True
                if self.pool:
                    self.pool.terminate()
                return
            finally:
                if self.pool:
                    try:
                        self.pool.close()
                        self.pool.join(timeout=2)
                    except:
                        if self.pool:
                            self.pool.terminate()
                    self.pool = None
            
            # Update evaluations
            for params, (rejects, _, timestamp) in zip(top_params, results):
                p_hash = param_hash(params)
                eval_data = self.state['evaluations'][p_hash]
                
                if 'run_summary' in eval_data:
                    run_summary = eval_data['run_summary']
                    run_summary['last_100'].append(rejects)
                    if len(run_summary['last_100']) > 100:
                        run_summary['last_100'].pop(0)
                    run_summary['total_count'] += 1
                    run_summary['best_score'] = min(run_summary['best_score'], rejects)
                    run_summary['percentiles'] = self._calculate_percentiles(run_summary['last_100'])
                    stats = calculate_statistics(run_summary['last_100'], detailed=True)
                else:
                    # Old format
                    eval_data['runs'].append(rejects)
                    stats = calculate_statistics(eval_data['runs'], detailed=True)
                
                eval_data['timestamps'].append(timestamp)
                if len(eval_data['timestamps']) > 100:
                    eval_data['timestamps'].pop(0)
                eval_data.update(stats)
                
                self.state['total_games'] = self.state.get('total_games', 0) + 1
        
        # Force save after validation phase
        self._save_state_batch(force=True)
    
    def _run_improvement(self):
        """Phase 3: Run champion indefinitely seeking improvements"""
        if not self.state['champion']:
            if not self.quiet:
                log("No champion selected, returning to validation")
            self.state['phase'] = 'validation'
            return
        
        champion_params = self.state['champion']['params']
        
        # Run multiple games in parallel
        try:
            results = []
            
            # Handle single worker case
            if self.workers == 1:
                results = [run_game((self.scenario, champion_params))]
            else:
                # Use existing pool or create if needed
                if not self.pool:
                    self.pool = Pool(self.workers)
                
                args = [(self.scenario, champion_params) for _ in range(self.workers)]
                
                # Use map_async instead of map
                async_result = self.pool.map_async(run_game, args)
                
                # Wait with timeout, checking for shutdown
                while not self.shutdown_requested:
                    try:
                        results = async_result.get(timeout=0.5)
                        break
                    except multiprocessing.TimeoutError:
                        if async_result.ready():
                            results = async_result.get()
                            break
                        continue
                
                if self.shutdown_requested:
                    if self.pool:
                        self.pool.terminate()
                        self.pool = None
                    return
                
        except (KeyboardInterrupt, SystemExit):
            self.shutdown_requested = True
            if self.pool:
                self.pool.terminate()
            return
        finally:
            # Keep pool alive for reuse (optimization)
            pass
        
        # Update champion stats
        champion_summary = self.state['champion'].get('run_summary')
        if not champion_summary:
            # Initialize if missing (migration case)
            champion_summary = {
                'last_500': [],
                'total_count': 0,
                'best_score': float('inf'),
                'percentiles': {}
            }
            self.state['champion']['run_summary'] = champion_summary
        
        for rejects, _, timestamp in results:
            # Update bounded history
            champion_summary['last_500'].append(rejects)
            if len(champion_summary['last_500']) > 500:
                champion_summary['last_500'].pop(0)
            
            champion_summary['total_count'] += 1
            
            # Update recent performance
            self.state['champion']['recent_performance'].append(rejects)
            if len(self.state['champion']['recent_performance']) > 20:
                self.state['champion']['recent_performance'].pop(0)
            
            self.state['total_games'] = self.state.get('total_games', 0) + 1
            
            # Check for improvement
            if rejects < champion_summary['best_score']:
                champion_summary['best_score'] = rejects
                total_runs = champion_summary['total_count']
                log(f"🚀 NEW CHAMPION BEST: {rejects} (run #{total_runs})")
                
                # Check global best
                if rejects < self.state.get('global_best', float('inf')):
                    self.state['global_best'] = rejects
                    log(f"🔥 NEW GLOBAL BEST: {rejects} rejections!")
                
                if self.target and rejects <= self.target:
                    log(f"🎯 TARGET {self.target} ACHIEVED WITH {rejects}!")
                    # Force save on target achievement
                    self._save_state_batch(force=True)
        
        # Update percentiles
        if len(champion_summary['last_500']) >= 5:
            champion_summary['percentiles'] = self._calculate_percentiles(champion_summary['last_500'])
        
        # Periodic re-validation (every 100 champion runs)
        if champion_summary['total_count'] % 100 == 0:
            if not self.quiet:
                log(f"Periodic check: {champion_summary['total_count']} champion runs completed")
                if champion_summary['percentiles']:
                    log(f"  Performance: p50={champion_summary['percentiles']['p50']:.1f}, "
                        f"p95={champion_summary['percentiles']['p95']:.1f}")
        
        # Periodic progress summary (every 50 games)
        if self.state['total_games'] % 50 == 0:
            recent_median = np.median(self.state['champion']['recent_performance']) if self.state['champion']['recent_performance'] else float('inf')
            if not self.quiet:
                log(f"📊 Progress: Total games={self.state['total_games']}, "
                    f"Global best={self.state.get('global_best', 'N/A')}, "
                    f"Recent median={recent_median:.1f}")
        
        # Use batch saving for improvement phase
        self._save_state_batch()
    
    def _check_new_best(self, p_hash: str, rejects: int) -> bool:
        """Check if this is a new best result"""
        current_best = float('inf')
        for eval_data in self.state['evaluations'].values():
            if 'run_summary' in eval_data:
                current_best = min(current_best, eval_data['run_summary']['best_score'])
            elif 'runs' in eval_data and eval_data['runs']:
                current_best = min(current_best, min(eval_data['runs']))
        return rejects < current_best
    
    def _display_progress(self):
        """Display current progress and statistics"""
        if self.quiet:
            return
            
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
                    
                if self.debug:
                    log(f"  Global best: {self.state.get('global_best', 'N/A')}")
                
            elif self.state['phase'] == 'validation':
                log(f"Validation Phase | Total games: {total_games} | {games_per_hour:.1f} games/h")
                
            elif self.state['phase'] == 'improvement':
                champion = self.state['champion']
                if champion:
                    recent = champion['recent_performance']
                    champion_summary = champion.get('run_summary', {})
                    
                    if recent:
                        recent_median = np.median(recent)
                        best_score = champion_summary.get('best_score', 'N/A')
                        run_count = champion_summary.get('total_count', 0)
                        
                        log(f"Improvement: Best={best_score} | "
                            f"Recent median={recent_median:.1f} | "
                            f"Champion runs={run_count} | "
                            f"{games_per_hour:.1f} games/h")

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Smart Berghain Optimizer v4.0")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3],
                       help="Scenario to optimize (required for --target)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel workers for games")
    parser.add_argument("--target", type=int, default=None,
                       help="Target rejection count (requires --scenario)")
    parser.add_argument("--debug", action="store_true",
                       help="Show detailed parameter information and debug output")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress all but critical messages")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target and not args.scenario:
        parser.error("--target requires --scenario")
    
    if args.debug and args.quiet:
        parser.error("--debug and --quiet are mutually exclusive")
    
    # Default to scenario 1 if not specified
    if not args.scenario:
        args.scenario = 1
        if not args.quiet:
            log("No scenario specified, defaulting to Scenario 1")
    
    # Run optimizer
    optimizer = SmartOptimizer(args.scenario, args.workers, args.target, args.debug, args.quiet)
    optimizer.run_forever()

if __name__ == "__main__":
    main()