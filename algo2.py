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

# Bayesian Optimization imports
try:
    from skopt import gp_minimize, Optimizer, dump, load
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import Matern
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not available, will fallback to random search")

# Constants
API_BASE = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "a47fcacd-00d4-4b8f-8a9d-821e4b69feed"
N = 1000  # Venue size
MAX_REJECTS = 20000  # Game fail limit
MIN_RUNS_PER_PARAM = 5  # Base runs for objective evaluation
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
        _session.headers.update({"User-Agent": "berghain_bo_optimizer/2.0"})
        
        # Connection pool optimized for single worker
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,  # Small pool for single worker
            pool_maxsize=2,      # Saves memory on Raspberry Pi
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
    weights = np.array([params.get(f'weight_{i}', 1.0) for i in range(K)])
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
def run_game(args: Tuple[int, Dict]) -> Tuple[int, Dict]:
    """Run a single game and return (rejects, params)"""
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
            return MAX_REJECTS, params
            
    except Exception as e:
        log(f"Game error: {e}")
        return MAX_REJECTS, params

# ========== Bayesian Optimization Functions ==========
def define_search_space(K: int) -> List:
    """Define the search space for Bayesian Optimization"""
    space = [
        Real(0.60, 1.00, name='threshold'),
        Real(0.0, 0.1, name='early_bonus'),
        Real(0.0, 0.2, name='ab_bonus'),  # Fixed upper bound; no conditional
        Integer(600, 900, name='early_threshold')
    ]
    # Add per-attribute weights for K>2 to handle varying mins
    for i in range(K):
        space.append(Real(0.5, 2.0, name=f'weight_{i}'))
    return space

def create_objective_function(scenario: int, workers: int, space: List):
    """Create the objective function for BO with variance handling"""
    @use_named_args(space)
    def objective(**params: Dict) -> float:
        num_runs = 3  # Base; increase to 5-10 if needed
        args = [(scenario, params) for _ in range(num_runs)]
        
        if workers > 1:
            with Pool(workers) as p:
                results = p.map(run_game, args)
        else:
            results = [run_game(arg) for arg in args]
        
        rejects = np.array([r[0] for r in results])
        
        # Variance-based resampling: If std too high, add runs
        if np.std(rejects) > 1000:  # Threshold based on observed variance
            extra_args = [(scenario, params) for _ in range(2)]  # Add 3 more
            if workers > 1:
                with Pool(workers) as p:
                    extra_results = p.map(run_game, extra_args)
            else:
                extra_results = [run_game(arg) for arg in extra_args]
            rejects = np.append(rejects, [r[0] for r in extra_results])
        
        median_score = np.median(rejects)
        return median_score  # Minimize median for robustness
    
    return objective

def fallback_random_search(scenario: int, K: int, workers: int, n_calls: int = 100):
    """Fallback to random search if scikit-optimize is not available"""
    log("Running random search fallback (scikit-optimize not available)")
    
    best_params = None
    best_score = float('inf')
    
    for i in range(n_calls):
        # Sample random parameters
        params = {
            'threshold': np.random.uniform(0.60, 1.00),
            'early_bonus': np.random.uniform(0.0, 0.1),
            'ab_bonus': np.random.uniform(0.0, 0.2),
            'early_threshold': np.random.randint(600, 901)
        }
        
        # Add weights for K>2
        for j in range(K):
            params[f'weight_{j}'] = np.random.uniform(0.5, 2.0)
        
        # Run games
        num_runs = 5
        args = [(scenario, params) for _ in range(num_runs)]
        
        if workers > 1:
            with Pool(workers) as p:
                results = p.map(run_game, args)
        else:
            results = [run_game(arg) for arg in args]
        
        rejects = [r[0] for r in results]
        median_score = np.median(rejects)
        
        if median_score < best_score:
            best_score = median_score
            best_params = params.copy()
            log(f"Random search: New best found - median={median_score:.1f}")
        
        if i % 10 == 0:
            log(f"Random search progress: {i+1}/{n_calls} evaluations")
    
    return best_params, best_score

# ========== Statistical Analysis ==========
def calculate_statistics(runs: List[int], detailed: bool = True) -> Dict:
    """Calculate statistics for a set of runs"""
    if not runs:
        return {
            'mean': float('inf'),
            'median': float('inf'),
            'std': 0,
            'confidence_95': [float('inf'), float('inf')]
        }
    
    runs_array = np.array(runs)
    median = np.median(runs_array)
    
    if not detailed:
        return {'median': float(median)}
    
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
        self.state_file = f"berghain_s{scenario}_bo_state.json"
        self.backup_file = f"berghain_s{scenario}_bo_state.backup.json"
        self.bo_state_file = f"berghain_s{scenario}_bo.pkl"
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
    
    def save_state(self, state: Dict, optimizer: Optional[Optimizer] = None):
        """Save state with atomic write and periodic backup"""
        # Update last activity
        state['last_activity'] = time.time()
        
        # Save optimizer if provided and skopt available
        if optimizer and SKOPT_AVAILABLE:
            try:
                dump(optimizer, self.bo_state_file)
                state['bo_state_file'] = self.bo_state_file
            except Exception as e:
                log(f"Warning: Could not save optimizer state: {e}")
        
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
    
    def load_optimizer(self, state: Dict) -> Optional[Optimizer]:
        """Load saved optimizer if available"""
        if not SKOPT_AVAILABLE:
            return None
        
        bo_file = state.get('bo_state_file', self.bo_state_file)
        if os.path.exists(bo_file):
            try:
                return load(bo_file)
            except Exception as e:
                log(f"Warning: Could not load optimizer state: {e}")
        return None
    
    def _validate_state(self, state: Dict) -> bool:
        """Validate state structure"""
        required_keys = ['scenario', 'phase', 'K']
        return all(key in state for key in required_keys)
    
    def _create_fresh_state(self) -> Dict:
        """Create fresh state structure"""
        # Get K from actual game
        game_data = new_game(self.scenario)
        K = len(game_data['constraints'])
        
        # Create serializable search space representation
        search_space = []
        search_space.append({'type': 'Real', 'low': 0.60, 'high': 1.00, 'name': 'threshold'})
        search_space.append({'type': 'Real', 'low': 0.0, 'high': 0.1, 'name': 'early_bonus'})
        search_space.append({'type': 'Real', 'low': 0.0, 'high': 0.2, 'name': 'ab_bonus'})
        search_space.append({'type': 'Integer', 'low': 600, 'high': 900, 'name': 'early_threshold'})
        
        for i in range(K):
            search_space.append({'type': 'Real', 'low': 0.5, 'high': 2.0, 'name': f'weight_{i}'})
        
        return {
            'scenario': self.scenario,
            'K': K,
            'search_space': search_space,
            'bo_evaluations': [],
            'champion': None,
            'phase': 'bo_search',
            'bo_state_file': self.bo_state_file,
            'last_activity': time.time(),
            'start_time': time.time(),
            'total_games': 0,
            'global_best': float('inf')
        }

# ========== Smart Optimizer with BO ==========
class SmartBOOptimizer:
    def __init__(self, scenario: int, workers: int = 4, target: Optional[int] = None, debug: bool = False, quiet: bool = False):
        self.scenario = scenario
        self.workers = workers
        self.target = target
        self.debug = debug
        self.quiet = quiet
        self.state_manager = StateManager(scenario)
        self.state = self.state_manager.load_state()
        self.pool = None
        self.shutdown_requested = False
        self.optimizer = None
        self.space = None
        self.objective = None
        
        # Performance tracking
        self.start_time = self.state.get('start_time', time.time())
        
        # Batch saving optimization
        self.games_since_save = 0
        self.save_interval = 1  # Save after each evaluation
        self.pending_evaluations = []  # Track unsaved evaluations
        
        # Create persistent pool (optimization)
        if self.workers > 1 and not self.shutdown_requested:
            self.pool = Pool(self.workers)
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize Bayesian Optimization
        self._initialize_bo()
    
    def __del__(self):
        """Cleanup pools on destruction"""
        self._cleanup_pool()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.shutdown_requested = True
        if self.pool:
            self.pool.terminate()
            self.pool = None  # Prevent reuse
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
    
    def _initialize_bo(self):
        """Initialize Bayesian Optimization components"""
        K = self.state['K']
        
        if SKOPT_AVAILABLE:
            # Define search space
            self.space = define_search_space(K)
            
            # Create objective function
            self.objective = create_objective_function(self.scenario, self.workers, self.space)
            
            # Initialize optimizer with knife-edge mitigations
            kernel = Matern(nu=2.5)  # Better for non-smooth landscapes
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)
            
            # Try to load existing optimizer
            self.optimizer = self.state_manager.load_optimizer(self.state)
            
            if not self.optimizer:
                self.optimizer = Optimizer(
                    dimensions=self.space,
                    base_estimator=gpr,
                    n_initial_points=15,  # Higher initial for exploration
                    acq_func='gp_hedge',  # Hedge between different acquisition functions
                    acq_optimizer='auto',
                    random_state=42
                )
                
                # Migrate from grid if available (warm start)
                self._migrate_from_grid()
        else:
            log("WARNING: Bayesian Optimization not available, using random search")
    
    def _migrate_from_grid(self):
        """Migrate evaluations from grid search if available"""
        # Check if there's a grid search state file
        grid_state_file = f"berghain_s{self.scenario}_state.json"
        if os.path.exists(grid_state_file) and SKOPT_AVAILABLE:
            try:
                with open(grid_state_file, 'r') as f:
                    grid_state = json.load(f)
                
                if 'evaluations' in grid_state and 'grid_params' in grid_state:
                    log("Found grid search results, migrating to BO...")
                    
                    # Convert top grid evaluations to BO initial points
                    evaluations = []
                    for idx_str, eval_data in grid_state['evaluations'].items():
                        if 'run_summary' in eval_data:
                            idx = int(idx_str)
                            params = grid_state['grid_params'][idx]
                            median = eval_data.get('median', float('inf'))
                            if median < float('inf'):
                                evaluations.append((params, median))
                    
                    # Sort by performance and take top results
                    evaluations.sort(key=lambda x: x[1])
                    top_evals = evaluations[:min(20, len(evaluations))]
                    
                    # Tell optimizer about these points
                    for params, score in top_evals:
                        # Convert params to list matching space order
                        point = [
                            params['threshold'],
                            params['early_bonus'],
                            params['ab_bonus'],
                            params['early_threshold']
                        ]
                        # Add weights
                        K = self.state['K']
                        for i in range(K):
                            point.append(params.get('attr_weights', [1.0]*K)[i])
                        
                        # Tell optimizer
                        self.optimizer.tell([point], [score])
                    
                    log(f"Migrated {len(top_evals)} evaluations from grid search")
            except Exception as e:
                log(f"Could not migrate grid search results: {e}")
    
    def _save_state_batch(self, force: bool = False, is_best: bool = False):
        """Save state with batching"""
        self.games_since_save += 1
        
        # Save any pending evaluations
        if self.pending_evaluations:
            self.state['bo_evaluations'].extend(self.pending_evaluations)
            self.pending_evaluations = []
        
        if force or is_best or self.games_since_save >= self.save_interval:
            self.state_manager.save_state(self.state, self.optimizer)
            self.games_since_save = 0
            if not self.quiet:
                log(f"💾 State saved (total evaluations: {len(self.state.get('bo_evaluations', []))})")
    
    def _update_phase(self):
        """Determine current phase from state"""
        bo_evals = len(self.state.get('bo_evaluations', []))
        
        if bo_evals < 50:  # Initial BO search phase
            self.state['phase'] = 'bo_search'
        elif not self.state.get('champion'):
            self.state['phase'] = 'validation'
        else:
            self.state['phase'] = 'improvement'
    
    def run_forever(self):
        """Main loop that runs indefinitely"""
        if not self.quiet:
            log(f"Bayesian Optimization for Scenario {self.scenario}")
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
                    if self.state['phase'] == 'bo_search':
                        self._run_bo_search()
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
                    self.pool.terminate()
                    self.pool.join(timeout=1)
                except:
                    pass
            
            try:
                # Save any pending evaluations before exit
                if self.pending_evaluations:
                    self.state['bo_evaluations'].extend(self.pending_evaluations)
                    if not self.quiet:
                        print(f"\n[{time.strftime('%H:%M:%S')}] 💾 Saving {len(self.pending_evaluations)} pending evaluations...", flush=True)
                    self.pending_evaluations = []
                
                self._save_state_batch(force=True)
                if not self.quiet:
                    total_evals = len(self.state.get('bo_evaluations', []))
                    best = self.state.get('global_best', 'N/A')
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ State saved. Total evaluations: {total_evals}, Best: {best}", flush=True)
                    print(f"[{time.strftime('%H:%M:%S')}] Goodbye!", flush=True)
            except Exception as e:
                print(f"Error saving state: {e}")
            
            os._exit(0)
    
    def _run_bo_search(self):
        """Phase 1: Bayesian Optimization search"""
        if not SKOPT_AVAILABLE:
            # Fallback to random search
            if len(self.state.get('bo_evaluations', [])) == 0:
                best_params, best_score = fallback_random_search(
                    self.scenario, self.state['K'], self.workers, n_calls=100
                )
                self.state['bo_evaluations'].append({
                    'params': best_params,
                    'median': best_score,
                    'runs': [],
                    'variance': 0
                })
                self.state['phase'] = 'validation'
                self._save_state_batch(force=True)
            return
        
        # Check budget
        bo_evals = len(self.state.get('bo_evaluations', []))
        if bo_evals >= 100:  # Total budget
            log("BO search budget exhausted, moving to validation")
            self.state['phase'] = 'validation'
            self._save_state_batch(force=True)
            return
        
        # Check for no improvement
        if bo_evals > 20:
            recent_best = min([e['median'] for e in self.state['bo_evaluations'][-10:]])
            overall_best = self.state.get('global_best', float('inf'))
            if recent_best > overall_best * 0.95:  # No 5% improvement
                log("No significant improvement in last 10 iterations, moving to validation")
                self.state['phase'] = 'validation'
                self._save_state_batch(force=True)
                return
        
        # Batch acquisition for parallel evaluation
        n_points = min(self.workers, 100 - bo_evals)  # Don't exceed budget
        
        if not self.quiet:
            current_best = self.state.get('global_best', 'N/A')
            log(f"\n📊 BO Search Progress: {bo_evals}/100 evaluations completed")
            log(f"   Current best: {current_best} rejections")
            log(f"   Requesting {n_points} new points to evaluate...")
        
        # Ask optimizer for next points
        next_points = self.optimizer.ask(n_points=n_points)
        
        # Convert points to param dicts
        param_dicts = []
        for point in next_points:
            params = {
                'threshold': point[0],
                'early_bonus': point[1],
                'ab_bonus': point[2],
                'early_threshold': int(point[3])
            }
            # Add weights
            for i in range(self.state['K']):
                params[f'weight_{i}'] = point[4 + i]
            param_dicts.append(params)
        
        # Evaluate points in parallel
        results = []
        for param_idx, params in enumerate(param_dicts, 1):
            if self.shutdown_requested:
                break
            
            # Run multiple games per parameter set
            num_runs = 5
            args = [(self.scenario, params) for _ in range(num_runs)]
            
            if not self.quiet:
                log(f"\n🎮 Evaluation {bo_evals + param_idx}/{min(100, bo_evals + n_points)}: Running {num_runs} games...")
            
            if self.workers > 1:
                # Check if pool was terminated by shutdown
                if self.shutdown_requested:
                    if not self.quiet:
                        log("Shutdown requested, skipping parallel games...")
                    break
                with Pool(self.workers) as p:
                    game_results = p.map(run_game, args)
            else:
                game_results = []
                for game_idx, arg in enumerate(args, 1):
                    if self.shutdown_requested:
                        if not self.quiet:
                            log(f"   Stopping games due to shutdown request...")
                        break
                    if not self.quiet:
                        log(f"   Game {game_idx}/{num_runs} running...")
                    result = run_game(arg)
                    game_results.append(result)
                    if not self.quiet:
                        log(f"   Game {game_idx}/{num_runs} complete: {result[0]} rejections")
            
            # Skip if shutdown was requested and we have incomplete results
            if self.shutdown_requested:
                if not self.quiet:
                    log("Shutdown requested, skipping remaining evaluations...")
                break
                
            rejects = np.array([r[0] for r in game_results])
            
            # Variance-based resampling
            if np.std(rejects) > 1000 and not self.shutdown_requested:
                if not self.quiet:
                    log(f"   High variance (σ={np.std(rejects):.1f}), running 3 additional games...")
                extra_args = [(self.scenario, params) for _ in range(3)]
                if self.workers > 1:
                    # Check if pool was terminated
                    if self.shutdown_requested:
                        if not self.quiet:
                            log("Shutdown requested, skipping extra parallel games...")
                        extra_results = []
                    else:
                        with Pool(self.workers) as p:
                            extra_results = p.map(run_game, extra_args)
                else:
                    extra_results = []
                    for extra_idx, arg in enumerate(extra_args, 1):
                        if self.shutdown_requested:
                            if not self.quiet:
                                log(f"   Stopping extra games due to shutdown request...")
                            break
                        if not self.quiet:
                            log(f"   Extra game {extra_idx}/3 running...")
                        result = run_game(arg)
                        extra_results.append(result)
                        if not self.quiet:
                            log(f"   Extra game {extra_idx}/3 complete: {result[0]} rejections")
                if extra_results:  # Only append if we got results
                    rejects = np.append(rejects, [r[0] for r in extra_results])
                else:
                    rejects = np.append(rejects, [])  # Empty append for consistency
            
            median_score = float(np.median(rejects))
            variance = float(np.std(rejects))
            
            results.append(median_score)
            
            # Store evaluation (as pending until saved)
            new_eval = {
                'params': params,
                'median': median_score,
                'runs': rejects.tolist(),
                'variance': variance
            }
            self.pending_evaluations.append(new_eval)
            
            self.state['total_games'] += len(rejects)
            
            if not self.quiet:
                log(f"   ✅ Evaluation complete: median={median_score:.1f}, variance={variance:.1f}")
                log(f"   Games: {rejects.tolist()}")
            
            # Check for new global best
            if median_score < self.state.get('global_best', float('inf')):
                self.state['global_best'] = median_score
                log(f"\n🔥 NEW GLOBAL BEST: median={median_score:.1f}, variance={variance:.1f}")
                if self.debug:
                    log(f"   Params: {json.dumps(params, indent=2)}")
                # Save immediately on new best
                self._save_state_batch(is_best=True)
            else:
                # Save after each evaluation
                self._save_state_batch()
        
        if not self.shutdown_requested:
            # Tell optimizer about results
            self.optimizer.tell(next_points[:len(results)], results)
            self._save_state_batch(force=True)
            if not self.quiet:
                log(f"\n📈 BO iteration complete. Total evaluations: {len(self.state.get('bo_evaluations', []))}/100")
    
    def _run_validation(self):
        """Phase 2: Select champion through validation"""
        bo_evals = self.state.get('bo_evaluations', [])
        
        if not bo_evals:
            log("No BO evaluations to validate, returning to search")
            self.state['phase'] = 'bo_search'
            return
        
        # Sort by median performance
        sorted_evals = sorted(bo_evals, key=lambda x: x['median'])
        
        # Select top 5 candidates for validation
        top_candidates = sorted_evals[:min(5, len(sorted_evals))]
        
        if not self.quiet:
            log(f"Validating top {len(top_candidates)} candidates with extended runs...")
        
        # Run 20 additional evaluations for each top candidate
        validation_results = []
        
        for candidate in top_candidates:
            if self.shutdown_requested:
                break
            
            params = candidate['params']
            args = [(self.scenario, params) for _ in range(20)]
            
            if self.workers > 1:
                with Pool(self.workers) as p:
                    results = p.map(run_game, args)
            else:
                results = [run_game(arg) for arg in args]
            
            rejects = [r[0] for r in results]
            stats = calculate_statistics(rejects, detailed=True)
            
            validation_results.append({
                'params': params,
                'stats': stats,
                'runs': rejects
            })
            
            self.state['total_games'] += 20
            
            if not self.quiet:
                log(f"Validation: median={stats['median']:.1f}, CI=[{stats['confidence_95'][0]:.1f}, {stats['confidence_95'][1]:.1f}]")
        
        if validation_results and not self.shutdown_requested:
            # Select best based on median
            best_result = min(validation_results, key=lambda x: x['stats']['median'])
            
            # Set as champion
            self.state['champion'] = {
                'params': best_result['params'],
                'selection_date': time.time(),
                'run_summary': {
                    'last_500': best_result['runs'][-500:],
                    'total_count': len(best_result['runs']),
                    'best_score': min(best_result['runs']),
                    'percentiles': {
                        'p25': float(np.percentile(best_result['runs'], 25)),
                        'p50': float(np.percentile(best_result['runs'], 50)),
                        'p75': float(np.percentile(best_result['runs'], 75)),
                        'p90': float(np.percentile(best_result['runs'], 90)),
                        'p95': float(np.percentile(best_result['runs'], 95))
                    }
                },
                'recent_performance': best_result['runs'][-20:]
            }
            
            log(f"🏆 CHAMPION SELECTED: median={best_result['stats']['median']:.1f}, best={min(best_result['runs'])}")
            if not self.quiet or self.debug:
                log(f"  Params: {json.dumps(best_result['params'], indent=2)}")
            
            self.state['phase'] = 'improvement'
            self._save_state_batch(force=True)
    
    def _run_improvement(self):
        """Phase 3: Run champion with periodic re-optimization"""
        if not self.state.get('champion'):
            log("No champion selected, returning to validation")
            self.state['phase'] = 'validation'
            return
        
        champion_params = self.state['champion']['params']
        champion_summary = self.state['champion'].get('run_summary')
        
        # Check if we should re-optimize (every 200 runs)
        if champion_summary['total_count'] > 0 and champion_summary['total_count'] % 200 == 0:
            if SKOPT_AVAILABLE:
                log("Running periodic re-optimization...")
                
                # Create narrowed search space around champion
                narrowed_space = []
                for i, dim in enumerate(self.space):
                    if i < 4:  # Main parameters
                        center = champion_params[dim.name]
                        if dim.name == 'early_threshold':
                            # Integer parameter
                            lower = max(600, int(center * 0.8))
                            upper = min(900, int(center * 1.2))
                            narrowed_space.append(Integer(lower, upper, name=dim.name))
                        else:
                            # Real parameters
                            lower = max(dim.low, center * 0.8)
                            upper = min(dim.high, center * 1.2)
                            narrowed_space.append(Real(lower, upper, name=dim.name))
                    else:
                        # Weight parameters
                        narrowed_space.append(dim)  # Keep original bounds
                
                # Create new optimizer for local search
                kernel = Matern(nu=2.5)
                gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-3)  # Add noise
                local_optimizer = Optimizer(
                    dimensions=narrowed_space,
                    base_estimator=gpr,
                    n_initial_points=5,
                    acq_func='EI',  # Use EI for exploitation
                    random_state=42
                )
                
                # Run 20 evaluations
                for _ in range(20):
                    if self.shutdown_requested:
                        break
                    
                    point = local_optimizer.ask()
                    params = {}
                    for i, dim in enumerate(narrowed_space):
                        if dim.name == 'early_threshold':
                            params[dim.name] = int(point[i])
                        else:
                            params[dim.name] = point[i]
                    
                    # Evaluate
                    args = [(self.scenario, params) for _ in range(5)]
                    if self.workers > 1:
                        with Pool(self.workers) as p:
                            results = p.map(run_game, args)
                    else:
                        results = [run_game(arg) for arg in args]
                    
                    rejects = [r[0] for r in results]
                    median_score = np.median(rejects)
                    
                    local_optimizer.tell([point], [median_score])
                    
                    # Update champion if better
                    if median_score < champion_summary['best_score'] * 0.95:  # 5% improvement
                        self.state['champion']['params'] = params
                        log(f"🚀 Champion improved through re-optimization: median={median_score:.1f}")
        
        # Run regular champion games
        args = [(self.scenario, champion_params) for _ in range(self.workers)]
        
        if self.workers > 1:
            with Pool(self.workers) as p:
                results = p.map(run_game, args)
        else:
            results = [run_game((self.scenario, champion_params))]
        
        # Update champion stats
        for rejects, _ in results:
            # Update bounded history
            champion_summary['last_500'].append(rejects)
            if len(champion_summary['last_500']) > 500:
                champion_summary['last_500'].pop(0)
            
            champion_summary['total_count'] += 1
            
            # Update recent performance
            self.state['champion']['recent_performance'].append(rejects)
            if len(self.state['champion']['recent_performance']) > 20:
                self.state['champion']['recent_performance'].pop(0)
            
            self.state['total_games'] += 1
            
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
                    self._save_state_batch(force=True)
        
        # Update percentiles
        if len(champion_summary['last_500']) >= 5:
            champion_summary['percentiles'] = {
                'p25': float(np.percentile(champion_summary['last_500'], 25)),
                'p50': float(np.percentile(champion_summary['last_500'], 50)),
                'p75': float(np.percentile(champion_summary['last_500'], 75)),
                'p90': float(np.percentile(champion_summary['last_500'], 90)),
                'p95': float(np.percentile(champion_summary['last_500'], 95))
            }
        
        # Periodic progress summary
        if self.state['total_games'] % 50 == 0:
            recent_median = np.median(self.state['champion']['recent_performance']) if self.state['champion']['recent_performance'] else float('inf')
            if not self.quiet:
                log(f"📊 Progress: Total games={self.state['total_games']}, "
                    f"Global best={self.state.get('global_best', 'N/A')}, "
                    f"Recent median={recent_median:.1f}")
        
        self._save_state_batch()
    
    def _display_progress(self):
        """Display current progress and statistics"""
        if self.quiet:
            return
        
        total_games = self.state.get('total_games', 0)
        elapsed = time.time() - self.start_time
        
        if total_games > 0:
            games_per_hour = total_games / (elapsed / 3600)
            
            if self.state['phase'] == 'bo_search':
                bo_evals = len(self.state.get('bo_evaluations', []))
                pending = len(self.pending_evaluations)
                best_median = min([e['median'] for e in self.state['bo_evaluations']]) if self.state.get('bo_evaluations') else 'N/A'
                eta_hours = (100 - bo_evals) / (games_per_hour / 5) if games_per_hour > 0 else 0  # 5 games per eval
                
                log(f"BO Search: {bo_evals}/100 evaluations (+{pending} pending) | "
                    f"Best: {best_median} | "
                    f"ETA: {eta_hours:.1f}h | "
                    f"{games_per_hour:.1f} games/h")
                
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
    parser = argparse.ArgumentParser(description="Bayesian Optimization Berghain Optimizer v2.0")
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
    optimizer = SmartBOOptimizer(args.scenario, args.workers, args.target, args.debug, args.quiet)
    optimizer.run_forever()

if __name__ == "__main__":
    main()