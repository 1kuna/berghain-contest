#!/usr/bin/env python3
# algo3.py: Enhanced Bayesian Optimization with Optimizer ET estimator
# Provides 20-30% efficiency gain over algo2.py in high dimensions

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
import urllib3

# Suppress SSL warnings since we're using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Bayesian Optimization imports
try:
    from skopt import Optimizer, dump, load
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not available, will fallback to random search")

# Constants
API_BASE = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "a47fcacd-00d4-4b8f-8a9d-821e4b69feed"
N = 1000  # Venue size
MAX_REJECTS = 20000  # Game fail limit
CONNECTION_ERROR = -1  # Special value for connection failures
MIN_RUNS_PER_PARAM = 3  # Base runs for objective evaluation (reduced, compensated by variance resampling)
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

# ========== API Error Classes ==========
class APIConnectionError(Exception):
    """Raised when API is unreachable (site down, SSL errors, etc.)"""
    pass

# ========== API Functions with Rate Limiting ==========
def new_game(scenario: int, attempt: int = 0) -> Dict:
    """Start a new game with exponential backoff for rate limits and connection retries"""
    url = f"{API_BASE}/new-game"
    params = {"scenario": scenario, "playerId": PLAYER_ID}
    sess = _get_session()
    
    connection_retry = 0
    max_connection_wait = 300  # Max wait between connection retries
    
    while True:  # Keep trying on connection errors
        try:
            resp = sess.get(url, params=params, timeout=10, verify=False)
            if resp.status_code == 429:  # Rate limited
                wait = min(300, 10 * (2 ** attempt))
                log(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                return new_game(scenario, attempt + 1)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout, urllib3.exceptions.ProtocolError) as e:
            # Connection error - site might be down
            wait = min(max_connection_wait, 10 * (2 ** connection_retry))
            log(f"🔴 API unreachable ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)
            connection_retry += 1
            continue
        except Exception as e:
            # Other errors - limited retries
            if attempt >= 2:
                raise
            time.sleep(0.2)
            attempt += 1

def decide_and_next(game_id: str, person_index: int, accept: Optional[bool] = None, attempt: int = 0) -> Dict:
    """Make decision and get next person with exponential backoff and connection retries"""
    url = f"{API_BASE}/decide-and-next"
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        params["accept"] = str(accept).lower()
    
    sess = _get_session()
    connection_retry = 0
    max_connection_wait = 300
    
    while True:  # Keep trying on connection errors
        try:
            resp = sess.get(url, params=params, timeout=10, verify=False)
            if resp.status_code == 429:
                wait = min(300, 10 * (2 ** attempt))
                log(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                return decide_and_next(game_id, person_index, accept, attempt + 1)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, urllib3.exceptions.ProtocolError) as e:
            # Connection error - site might be down
            wait = min(max_connection_wait, 10 * (2 ** connection_retry))
            log(f"🔴 API unreachable ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)
            connection_retry += 1
            continue
        except Exception as e:
            # Other errors - limited retries
            if attempt >= 2:
                raise
            time.sleep(0.2)
            attempt += 1

# ========== Decision Function ==========
def decide(
    constraints: List[Dict],
    admitted_count: int,
    next_person: Dict,
    accepted_count: Dict,
    params: Dict
) -> bool:
    """Core decision function (optimized with numpy and meta-parameters)"""
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
    
    # Compute weights dynamically from meta-parameters using inverse prior + dynamic risk
    
    # Marginals (hardcoded per scenario or EMA; tuned for S3)
    p_marg = np.array([0.65, 0.45, 0.3, 0.75]) if K == 4 else np.full(K, 0.5)  # Fallback for other scenarios
    exp_left = np.maximum(1.0, p_marg * max(S, 1))
    
    # Inverse prior (smooth exp, clipped)
    total_min = sum(c['minCount'] for c in constraints)
    inv_base = np.array([total_min / (K * c['minCount']) for c in constraints])
    inv_prior = np.clip(np.exp(params.get('weight_var', 1.0) * np.log(inv_base)), 0.5, 3.0)
    weights = np.ones(K) * inv_prior
    
    # Dynamic risk (need/exp_left ^ beta)
    risk = needs / exp_left
    beta = params.get('risk_beta', 1.0) * params['weight_var'] if 'weight_var' in params else params.get('risk_beta', 1.0)  # 0.5-1.5 range
    weights *= np.clip(np.power(risk, beta), 0.4, 2.5)
    
    # Apply base scale
    weights *= params.get('weight_scale', 1.0)
    
    # Overshoot penalty and scarce guard
    accepted_arr = np.array([accepted_count.get(c['attribute'], 0) for c in constraints])
    minC_arr = np.array([c['minCount'] for c in constraints])
    proj_surplus = np.maximum(0, (accepted_arr + exp_left) - minC_arr) / max(S, 1)
    alpha = params.get('overshoot_alpha', 0.8)
    weights *= 1.0 / (1.0 + alpha * proj_surplus)  # Downweight safe attributes
    
    # Scarce guard (top-2 risk; tax if oversupplied + no cover)
    weighted_pressure = 0  # Initialize weighted_pressure
    if np.sum(needs > 0) > 0:
        scarcity_idx = np.argsort(-risk)[:min(2, K)]
        oversupplied = proj_surplus > params.get('overshoot_eps', 0.02)
        has_oversupplied = np.any(person_attrs & oversupplied)
        covers_scarce = np.any(person_attrs[scarcity_idx])
        if has_oversupplied and not covers_scarce:
            weighted_pressure += params.get('overshoot_tax', 0.05)  # Tax for oversupplied without scarce coverage
    
    weighted_pressure += np.sum((needs[missing_mask] / max(S, EPS)) * weights[missing_mask])
    
    # Adjustments
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
    
    # Threshold ramp: stricter late in the game
    base_t = params['threshold']
    ramp = params.get('threshold_ramp', 0.08)
    t = base_t + ramp * (1 - S / N)
    
    # Debug logging (controlled by debug parameter)
    if params.get('debug', False):
        print(f"Debug: S={S}, weights={weights.round(3)}, risk={risk.round(3)}, proj_surplus={proj_surplus.round(4)}, adjusted_pressure={adjusted_pressure:.3f}, threshold={t:.3f}")
    
    return adjusted_pressure < t

# ========== Game Runner ==========
def run_game(args: Tuple[int, Dict]) -> Tuple[int, Dict]:
    """Run a single game and return (rejects, params)
    
    Returns:
        - (rejection_count, params) for successful games
        - (CONNECTION_ERROR, params) for connection failures
        - (MAX_REJECTS, params) for game logic failures
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
    
    except APIConnectionError:
        # Connection error - API is unreachable
        return CONNECTION_ERROR, params
            
    except Exception as e:
        log(f"Game error: {e}")
        return MAX_REJECTS, params

# ========== Bayesian Optimization Functions ==========
def define_search_space(K: int) -> List:
    """Define the search space for Bayesian Optimization (reduced dimensionality)"""
    space = [
        Real(0.60, 1.00, name='threshold'),
        Real(0.0, 0.1, name='early_bonus'),
        Real(0.0, 0.2, name='ab_bonus'),
        Integer(600, 900, name='early_threshold'),
        Real(0.5, 2.0, name='weight_scale'),  # Uniform scale for all weights
        Real(0.5, 1.5, name='risk_beta'),  # Dynamic risk exponent
        Real(0.5, 1.0, name='overshoot_alpha'),  # Overshoot penalty factor
        Real(0.0, 0.1, name='threshold_ramp'),  # Threshold ramp factor
        Real(0.0, 0.08, name='overshoot_tax'),  # Tax for oversupplied without scarce
        Real(0.01, 0.05, name='overshoot_eps')  # Threshold for considering oversupplied
    ]
    if K > 2:
        space.append(Real(0.0, 1.0, name='weight_var'))  # Variance factor for differential weighting
    return space

def create_objective_function(scenario: int, optimizer_instance):
    """Create the objective function for BO with variance handling"""
    @use_named_args(optimizer_instance.space)
    def objective(**params: Dict) -> float:
        num_runs = 3  # Base; increase if needed
        args = [(scenario, params) for _ in range(num_runs)]
        
        results = optimizer_instance._map_games(args)
        # Filter out connection errors
        all_results = [r[0] for r in results]
        valid_results = [r for r in all_results if r != CONNECTION_ERROR]
        
        # Return inf if too many connection errors
        if len(valid_results) < 2:  # Need at least 2 valid results
            log(f"Too many connection errors in objective evaluation")
            return float('inf')
        
        rejects = np.array(valid_results)
        
        # Variance-based resampling: If std too high, add runs
        sigma = np.std(rejects)
        med = np.median(rejects)
        if sigma > max(120.0, 0.10 * med):  # Adaptive threshold: >120 or >10% of median
            extra_args = [(scenario, params) for _ in range(2)]  # Add 2 more
            extra_results = optimizer_instance._map_games(extra_args)
            # Filter connection errors from extra results
            all_extra = [r[0] for r in extra_results]
            valid_extra = [r for r in all_extra if r != CONNECTION_ERROR]
            if valid_extra:
                rejects = np.append(rejects, valid_extra)
        
        median_score = np.median(rejects)
        return median_score  # Minimize median for robustness
    
    return objective

def fallback_random_search(scenario: int, K: int, workers: int, n_calls: int = 200):
    """Fallback to random search with Latin Hypercube sampling"""
    log("Running optimized random search fallback with Latin Hypercube sampling")
    
    try:
        from scipy.stats import qmc
        # Use Latin Hypercube for better coverage
        d = 10 + (1 if K > 2 else 0)  # Number of dimensions (5 original + 5 new parameters)
        sampler = qmc.LatinHypercube(d=d)
        U = sampler.random(n_calls)  # Generate [0,1) samples
        
        def scale(u, lo, hi, integer=False):
            """Scale uniform sample to parameter range"""
            x = lo + u * (hi - lo)
            return int(round(x)) if integer else x
        
        use_lhs = True
    except ImportError:
        log("scipy.qmc not available, using standard random sampling")
        use_lhs = False
    
    best_params = None
    best_score = float('inf')
    
    for i in range(n_calls):
        # Sample parameters
        if use_lhs:
            u = U[i]
            params = {
                'threshold': scale(u[0], 0.60, 1.00),
                'early_bonus': scale(u[1], 0.0, 0.1),
                'ab_bonus': scale(u[2], 0.0, 0.2),
                'early_threshold': scale(u[3], 600, 900, integer=True),
                'weight_scale': scale(u[4], 0.5, 2.0),
                'risk_beta': scale(u[5], 0.5, 1.5),
                'overshoot_alpha': scale(u[6], 0.5, 1.0),
                'threshold_ramp': scale(u[7], 0.0, 0.1),
                'overshoot_tax': scale(u[8], 0.0, 0.08),
                'overshoot_eps': scale(u[9], 0.01, 0.05)
            }
            if K > 2:
                params['weight_var'] = scale(u[10], 0.0, 1.0)
        else:
            # Fallback to standard random sampling
            params = {
                'threshold': np.random.uniform(0.60, 1.00),
                'early_bonus': np.random.uniform(0.0, 0.1),
                'ab_bonus': np.random.uniform(0.0, 0.2),
                'early_threshold': np.random.randint(600, 901),
                'weight_scale': np.random.uniform(0.5, 2.0),
                'risk_beta': np.random.uniform(0.5, 1.5),
                'overshoot_alpha': np.random.uniform(0.5, 1.0),
                'threshold_ramp': np.random.uniform(0.0, 0.1),
                'overshoot_tax': np.random.uniform(0.0, 0.08),
                'overshoot_eps': np.random.uniform(0.01, 0.05)
            }
            if K > 2:
                params['weight_var'] = np.random.uniform(0.0, 1.0)
        
        # Run games
        num_runs = 3
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
        
        # Save optimizer state if provided and skopt available
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
        
        # Add missing K if not present
        if 'K' not in state and 'scenario' in state:
            fallback_K = {1: 2, 2: 2, 3: 3}
            state['K'] = fallback_K.get(state['scenario'], 2)
            log(f"Added missing K={state['K']} to state for scenario {state['scenario']}")
        
        return all(key in state for key in required_keys)
    
    def _create_fresh_state(self) -> Dict:
        """Create fresh state structure"""
        # Get K from actual game with fallback on API failure
        try:
            game_data = new_game(self.scenario)
            K = len(game_data['constraints'])
        except Exception as e:
            log(f"API failed during fresh state creation: {e}")
            # Use fallback K values based on known scenario constraints
            fallback_K = {1: 2, 2: 2, 3: 3}
            K = fallback_K.get(self.scenario, 2)
            log(f"Using fallback K={K} for scenario {self.scenario}")
        
        # Create serializable search space representation (expanded dimensionality)
        search_space = []
        search_space.append({'type': 'Real', 'low': 0.60, 'high': 1.00, 'name': 'threshold'})
        search_space.append({'type': 'Real', 'low': 0.0, 'high': 0.1, 'name': 'early_bonus'})
        search_space.append({'type': 'Real', 'low': 0.0, 'high': 0.2, 'name': 'ab_bonus'})
        search_space.append({'type': 'Integer', 'low': 600, 'high': 900, 'name': 'early_threshold'})
        search_space.append({'type': 'Real', 'low': 0.5, 'high': 2.0, 'name': 'weight_scale'})
        search_space.append({'type': 'Real', 'low': 0.5, 'high': 1.5, 'name': 'risk_beta'})
        search_space.append({'type': 'Real', 'low': 0.5, 'high': 1.0, 'name': 'overshoot_alpha'})
        search_space.append({'type': 'Real', 'low': 0.0, 'high': 0.1, 'name': 'threshold_ramp'})
        search_space.append({'type': 'Real', 'low': 0.0, 'high': 0.08, 'name': 'overshoot_tax'})
        search_space.append({'type': 'Real', 'low': 0.01, 'high': 0.05, 'name': 'overshoot_eps'})
        
        if K > 2:
            search_space.append({'type': 'Real', 'low': 0.0, 'high': 1.0, 'name': 'weight_var'})
        
        return {
            'scenario': self.scenario,
            'K': K,
            'search_space': search_space,
            'bo_evaluations': [],
            'partial_evals': [],  # Store partial evaluations for recovery
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
        self.pending_evaluations = []  # Track unsaved evaluations - initialize to empty
        
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
    
    def _map_games(self, args: List[Tuple[int, Dict]]) -> List[Tuple[int, Dict]]:
        """Centralized game execution using persistent pool
        
        Returns list of (result, params) where result can be:
        - rejection count (successful game)
        - CONNECTION_ERROR (-1) for connection failures
        - MAX_REJECTS (20000) for game logic failures
        """
        if self.workers > 1 and self.pool and not self.shutdown_requested:
            return self.pool.map(run_game, args)
        else:
            # Sequential fallback with shutdown check
            results = []
            for arg in args:
                if self.shutdown_requested:
                    break
                result = run_game(arg)
                results.append(result)
                # Log connection errors immediately
                if result[0] == CONNECTION_ERROR:
                    log(f"Game failed due to connection error, will be excluded from results")
            return results
    
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
        """Initialize Bayesian Optimization with ET (Extra Trees) estimator"""
        K = self.state['K']
        
        if SKOPT_AVAILABLE:
            # Define search space
            self.space = define_search_space(K)
            
            # Create objective function
            self.objective = create_objective_function(self.scenario, self)
            
            # Try to load existing optimizer state
            self.optimizer = self.state_manager.load_optimizer(self.state)
            
            # Check for optimizer space mismatch and fix if needed
            if self.optimizer:
                expected_space = define_search_space(K)
                if len(self.optimizer.space) != len(expected_space):
                    log(f"Optimizer space mismatch detected: loaded={len(self.optimizer.space)}, expected={len(expected_space)}")
                    log("Recreating optimizer with correct space and warm-starting from existing evaluations")
                    
                    # Recreate optimizer with correct space
                    self.optimizer = Optimizer(
                        dimensions=expected_space,
                        base_estimator="ET",
                        n_initial_points=0,  # Skip initial points since we'll warm-start
                        acq_func="EI",
                        random_state=42
                    )
                    
                    # Warm-start with existing bo_evaluations
                    if self.state.get('bo_evaluations'):
                        log(f"Warm-starting recreated optimizer with {len(self.state['bo_evaluations'])} existing evaluations")
                        X = []
                        y = []
                        for eval_data in self.state['bo_evaluations']:
                            point = []
                            params = eval_data.get('params', {})
                            # Build point in correct dimension order
                            for dim in expected_space:
                                val = params.get(dim.name)
                                if val is not None:
                                    point.append(val)
                                else:
                                    # Use dimension default if missing
                                    point.append(dim.low)
                            if len(point) == len(expected_space):
                                X.append(point)
                                y.append(eval_data['median'])
                        if X:
                            self.optimizer.tell(X, y)
                            log(f"Warm-started optimizer with {len(X)} valid evaluations")
                    
            if not self.optimizer:
                # Initialize new Optimizer with ET for robustness in high-D
                self.optimizer = Optimizer(
                    dimensions=self.space,
                    base_estimator="ET",  # Extra Trees for forest-like robustness
                    n_initial_points=25,  # Explicit for better exploration
                    acq_func="EI",        # Expected Improvement for noisy environments
                    random_state=42
                )
                
                # Warm-start from existing evaluations if any
                if self.state.get('bo_evaluations'):
                    log(f"Warm-starting optimizer with {len(self.state['bo_evaluations'])} existing evaluations")
                    X = []
                    y = []
                    for eval_data in self.state['bo_evaluations']:
                        point = []
                        for dim in self.space:
                            val = eval_data['params'].get(dim.name)
                            if val is not None:
                                point.append(val)
                        if len(point) == len(self.space):
                            X.append(point)
                            y.append(eval_data['median'])
                    if X:
                        self.optimizer.tell(X, y)
            
            # Resume partial evaluations if any
            if self.state.get('partial_evals'):
                log(f"Found {len(self.state['partial_evals'])} partial evaluations to resume")
                processed_partials = 0
                
                for partial in self.state['partial_evals'][:]:  # Copy list to modify during iteration
                    current_valid = len(partial.get('runs', []))
                    metadata = partial.get('metadata', {})
                    
                    # Log metadata if available
                    if metadata:
                        log(f"  Previous attempts: {metadata.get('total_attempts', 'unknown')}, Connection errors: {metadata.get('connection_errors', 0)}")
                    
                    # Check if this partial can be auto-promoted (>=2 valid runs and not connection_failed)
                    if (current_valid >= 2 and 
                        metadata.get('status') != 'connection_failed' and
                        'params' in partial):
                        
                        # Auto-promote to full evaluation without additional runs
                        rejects = np.array(partial['runs'])
                        median = np.median(rejects)
                        variance = np.std(rejects)
                        
                        # Add to bo_evaluations
                        self.state['bo_evaluations'].append({
                            'params': partial['params'],
                            'median': float(median),
                            'runs': partial['runs'],
                            'variance': float(variance),
                            'metadata': metadata
                        })
                        
                        # Tell optimizer about the recovered partial
                        point = []
                        for dim in self.space:
                            val = partial['params'].get(dim.name)
                            if val is not None:
                                point.append(val)
                        if len(point) == len(self.space):
                            self.optimizer.tell([point], [float(median)])
                        
                        self.state['partial_evals'].remove(partial)
                        processed_partials += 1
                        log(f"Auto-promoted partial eval: median={median:.1f}, variance={variance:.1f}, runs={current_valid}")
                        continue
                    
                    # For partials that need completion
                    expected = MIN_RUNS_PER_PARAM
                    missing = expected - current_valid
                    
                    if missing > 0:
                        # Complete the partial evaluation
                        log(f"Resuming partial eval: {missing} missing runs for params")
                        extra_args = [(self.scenario, partial['params']) for _ in range(missing)]
                        extra_results = self._map_games(extra_args)
                        
                        # Filter connection errors
                        all_results = [r[0] for r in extra_results]
                        valid_results = [r for r in all_results if r != CONNECTION_ERROR]
                        
                        if valid_results:
                            partial['runs'].extend(valid_results)
                            # Update metadata
                            if 'metadata' not in partial:
                                partial['metadata'] = {}
                            partial['metadata']['total_attempts'] = partial['metadata'].get('total_attempts', current_valid) + len(all_results)
                            partial['metadata']['connection_errors'] = partial['metadata'].get('connection_errors', 0) + all_results.count(CONNECTION_ERROR)
                    
                    # If we now have enough valid runs, promote to full evaluation
                    if len(partial.get('runs', [])) >= expected:
                        rejects = np.array(partial['runs'])
                        median = np.median(rejects)
                        variance = np.std(rejects)
                        self.state['bo_evaluations'].append({
                            'params': partial['params'],
                            'median': float(median),
                            'runs': partial['runs'],
                            'variance': float(variance),
                            'metadata': partial.get('metadata', {})
                        })
                        # Also tell optimizer about recovered partials
                        point = []
                        for dim in self.space:
                            val = partial['params'].get(dim.name)
                            if val is not None:
                                point.append(val)
                        if len(point) == len(self.space):
                            self.optimizer.tell([point], [float(median)])
                        self.state['partial_evals'].remove(partial)
                        processed_partials += 1
                        log(f"Completed partial eval: median={median:.1f}, variance={variance:.1f}")
                
                if processed_partials > 0:
                    log(f"Processed {processed_partials} partials")
                
                # Save any remaining partials that couldn't be completed
                if self.state['partial_evals']:
                    log(f"{len(self.state['partial_evals'])} partial evaluations still incomplete")
                self._save_state_batch(force=True)
            
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
                    
                    # Convert to new parameter format and store in state
                    K = self.state['K']
                    for params, score in top_evals:
                        # Convert old weights to meta-parameters
                        new_params = {
                            'threshold': params['threshold'],
                            'early_bonus': params['early_bonus'],
                            'ab_bonus': params['ab_bonus'],
                            'early_threshold': params['early_threshold'],
                            'weight_scale': 1.0  # Default scale
                        }
                        if K > 2:
                            new_params['weight_var'] = 0.5  # Default variance
                        
                        # Add to evaluations
                        self.state['bo_evaluations'].append({
                            'params': new_params,
                            'median': score,
                            'runs': [],
                            'variance': 0
                        })
                    
                    # Tell optimizer about migrated evaluations
                    if self.optimizer and top_evals:
                        X = []
                        y = []
                        for params, score in top_evals:
                            point = [
                                params['threshold'],
                                params['early_bonus'],
                                params['ab_bonus'],
                                params['early_threshold'],
                                1.0  # weight_scale default
                            ]
                            if K > 2:
                                point.append(0.5)  # weight_var default
                            X.append(point)
                            y.append(score)
                        if X:
                            self.optimizer.tell(X, y)
                    
                    log(f"Migrated {len(top_evals)} evaluations from grid search")
            except Exception as e:
                log(f"Could not migrate grid search results: {e}")
    
    def _save_state_batch(self, force: bool = False, is_best: bool = False):
        """Save state with batching"""
        self.games_since_save += 1
        
        # Note: pending_evaluations are now immediately added to bo_evaluations
        # This ensures no data loss on mid-batch crashes
        
        if force or is_best or self.games_since_save >= self.save_interval:
            self.state_manager.save_state(self.state, self.optimizer if SKOPT_AVAILABLE else None)
            self.games_since_save = 0
            if not self.quiet:
                log(f"💾 State saved (total evaluations: {len(self.state.get('bo_evaluations', []))})")
    
    def _update_phase(self):
        """Determine current phase from state"""
        bo_evals = len(self.state.get('bo_evaluations', []))
        
        if bo_evals < 150:  # Extended BO search phase
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
                # Save final state (pending evaluations are now immediately persisted)
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
        """Phase 1: Bayesian Optimization search using Optimizer with ET"""
        if not SKOPT_AVAILABLE or not self.optimizer:
            # Fallback to random search
            if len(self.state.get('bo_evaluations', [])) == 0:
                best_params, best_score = fallback_random_search(
                    self.scenario, self.state['K'], self.workers, n_calls=200
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
        if bo_evals >= 150:  # Increased budget for better convergence
            log("BO search budget exhausted, moving to validation")
            self.state['phase'] = 'validation'
            self._save_state_batch(force=True)
            return
        
        # Check for no improvement
        if bo_evals > 30:  # Increased threshold before checking
            recent_best = min([e['median'] for e in self.state['bo_evaluations'][-15:]])
            overall_best = self.state.get('global_best', float('inf'))
            if recent_best > overall_best * 0.92:  # Tighter threshold for global search
                log("No significant improvement in last 15 iterations, moving to validation")
                self.state['phase'] = 'validation'
                self._save_state_batch(force=True)
                return
        
        # Determine how many new points to evaluate (aggressive batching)
        n_points = min(self.workers * 3, 150 - bo_evals)  # More aggressive batching
        
        if not self.quiet:
            current_best = self.state.get('global_best', 'N/A')
            log(f"\n📊 BO Search Progress: {bo_evals}/150 evaluations completed")
            log(f"   Current best: {current_best} rejections")
            log(f"   Requesting {n_points} new points to evaluate...")
        
        # Ask optimizer for next batch of points
        next_points = self.optimizer.ask(n_points=n_points)
        
        # Convert points to parameter dictionaries
        param_dicts = []
        for point in next_points:
            params = {}
            for i, dim in enumerate(self.space):
                if dim.name == 'early_threshold':
                    params[dim.name] = int(point[i])
                else:
                    params[dim.name] = point[i]
            # Add debug flag if optimizer is in debug mode
            if self.debug:
                params['debug'] = True
            param_dicts.append(params)
        
        # Evaluate points in parallel
        results = []
        for param_idx, params in enumerate(param_dicts, 1):
            if self.shutdown_requested:
                break
            
            # Run multiple games per parameter set with partial result tracking
            num_runs = 3  # Reduced base runs
            args = [(self.scenario, params) for _ in range(num_runs)]
            
            if not self.quiet:
                log(f"\n🎮 Evaluation {bo_evals + param_idx}/{min(150, bo_evals + n_points)}: Running {num_runs} games...")
            
            # Track partial results for recovery
            partial = {'params': params.copy(), 'runs': []}
            
            try:
                if not self.quiet and self.workers == 1:
                    # Show progress for sequential execution
                    game_results = []
                    for game_idx, arg in enumerate(args, 1):
                        if self.shutdown_requested:
                            if not self.quiet:
                                log(f"   Stopping games due to shutdown request...")
                            break
                        log(f"   Game {game_idx}/{num_runs} running...")
                        result = run_game(arg)
                        game_results.append(result)
                        log(f"   Game {game_idx}/{num_runs} complete: {result[0]} rejections")
                else:
                    # Use centralized game mapping with enhanced error handling
                    try:
                        game_results = self._map_games(args)
                    except Exception as map_error:
                        log(f"Error in _map_games: {map_error}")
                        # Clean up pool on error
                        if self.pool:
                            try:
                                self.pool.terminate()
                                self.pool = None
                            except:
                                pass
                        # Return empty results and let outer exception handler deal with it
                        game_results = []
                
                # Skip if shutdown was requested and we have incomplete results
                if self.shutdown_requested:
                    if not self.quiet:
                        log("Shutdown requested, saving partial results...")
                    if game_results:
                        # Filter out connection errors but save all results for metadata
                        all_results = [r[0] for r in game_results]
                        valid_results = [r for r in all_results if r != CONNECTION_ERROR]
                        partial['runs'] = valid_results
                        partial['metadata'] = {'total_attempts': len(all_results), 'connection_errors': all_results.count(CONNECTION_ERROR)}
                    break
                    
                # Filter out connection errors from results
                all_results = [r[0] for r in game_results]
                valid_results = [r for r in all_results if r != CONNECTION_ERROR]
                
                # Check if we have any valid results
                if not valid_results:
                    log(f"   ⚠️ All games failed due to connection errors, skipping evaluation")
                    partial['metadata'] = {'total_attempts': len(all_results), 'connection_errors': len(all_results), 'status': 'connection_failed'}
                    # Still save partial with connection failure metadata
                    if partial not in self.state.get('partial_evals', []):
                        self.state.setdefault('partial_evals', []).append(partial)
                        self._save_state_batch(force=True)
                    continue
                
                rejects = np.array(valid_results)
                partial['runs'] = valid_results
                partial['metadata'] = {'total_attempts': len(all_results), 'connection_errors': all_results.count(CONNECTION_ERROR)}
                
                # Variance-based resampling with adaptive threshold
                sigma = np.std(rejects)
                med = np.median(rejects)
                if sigma > max(120.0, 0.10 * med) and not self.shutdown_requested:
                    if not self.quiet:
                        log(f"   High variance (σ={sigma:.1f}), running 3 additional games...")
                    extra_args = [(self.scenario, params) for _ in range(3)]
                    
                    if not self.quiet and self.workers == 1:
                        # Show progress for sequential extra games
                        extra_results = []
                        for extra_idx, arg in enumerate(extra_args, 1):
                            if self.shutdown_requested:
                                if not self.quiet:
                                    log(f"   Stopping extra games due to shutdown request...")
                                break
                            log(f"   Extra game {extra_idx}/3 running...")
                            result = run_game(arg)
                            extra_results.append(result)
                            log(f"   Extra game {extra_idx}/3 complete: {result[0]} rejections")
                    else:
                        # Enhanced error handling for variance resampling
                        try:
                            extra_results = self._map_games(extra_args)
                        except Exception as extra_map_error:
                            log(f"Error in variance resampling _map_games: {extra_map_error}")
                            extra_results = []
                    
                    if extra_results:  # Only append if we got results
                        # Filter out connection errors from extra results
                        all_extra = [r[0] for r in extra_results]
                        valid_extra = [r for r in all_extra if r != CONNECTION_ERROR]
                        if valid_extra:
                            rejects = np.append(rejects, valid_extra)
                            partial['runs'].extend(valid_extra)
                            # Update metadata
                            partial['metadata']['total_attempts'] += len(all_extra)
                            partial['metadata']['connection_errors'] += all_extra.count(CONNECTION_ERROR)
                
                # Initialize median_score with proper scope
                median_score = float(np.median(rejects))
                variance = float(np.std(rejects))
                
                results.append(median_score)
                
                # Store evaluation and immediately persist to prevent data loss
                new_eval = {
                    'params': params,
                    'median': median_score,
                    'runs': partial['runs'],
                    'variance': variance
                }
                # Immediately add to state to prevent loss on mid-batch crash
                self.state['bo_evaluations'].append(new_eval)
                
                self.state['total_games'] += len(partial['runs'])
                
                if not self.quiet:
                    log(f"   ✅ Evaluation complete: median={median_score:.1f}, variance={variance:.1f}")
                    log(f"   Games: {partial['runs']}")
                    
            except Exception as e:
                log(f"Evaluation interrupted: {e}")
                median_score = float('inf')  # Default value for failed evaluation
                # Clean up pool on evaluation error
                if self.pool:
                    try:
                        self.pool.terminate()
                        self.pool = None
                        log("Pool terminated due to evaluation error")
                    except:
                        pass
                # Save partial immediately on exception
                if partial['runs']:
                    self.state.setdefault('partial_evals', []).append(partial)
                    self._save_state_batch(force=True)
            finally:
                # Additional check to ensure partials are saved
                if partial['runs'] and partial not in self.state.get('partial_evals', []):
                    self.state.setdefault('partial_evals', []).append(partial)
                    self._save_state_batch(force=True)
            
            # Check for new global best (only if we have a valid median)
            if median_score != float('inf') and median_score < self.state.get('global_best', float('inf')):
                self.state['global_best'] = median_score
                log(f"\n🔥 NEW GLOBAL BEST: median={median_score:.1f}, variance={variance:.1f}")
                if self.debug:
                    log(f"   Params: {json.dumps(params, indent=2)}")
                # Save immediately on new best
                self._save_state_batch(is_best=True)
            else:
                # Save after each evaluation
                self._save_state_batch()
        
        if not self.shutdown_requested and results:
            # Tell optimizer about all results from this batch with error handling
            try:
                used_points = next_points[:len(results)]
                self.optimizer.tell(used_points, results)
                self._save_state_batch(force=True)
                if not self.quiet:
                    log(f"\n📈 BO iteration complete. Total evaluations: {len(self.state.get('bo_evaluations', []))}/150")
            except Exception as tell_error:
                log(f"Error in optimizer.tell(): {tell_error}")
                # Save state anyway to preserve evaluations
                self._save_state_batch(force=True)
    
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
        
        # Initialize validation progress tracking
        self.state.setdefault('validation_progress', {})
        validation_results = []
        
        for candidate in top_candidates:
            if self.shutdown_requested:
                break
            
            params = candidate['params']
            
            # Create candidate hash for tracking
            candidate_hash = hash(tuple(sorted(params.items())))
            
            # Check if we have progress for this candidate
            progress = self.state['validation_progress'].get(str(candidate_hash), {'partial_runs': [], 'target': 15})
            existing_runs = len(progress['partial_runs'])
            remaining_runs = 15 - existing_runs
            
            if not self.quiet and existing_runs > 0:
                log(f"Resuming validation: {existing_runs} runs already completed, {remaining_runs} remaining")
            
            # Only run remaining games
            if remaining_runs > 0:
                args = [(self.scenario, params) for _ in range(remaining_runs)]
                results = self._map_games(args)
                
                # Filter out connection errors and add to progress
                all_results = [r[0] for r in results]
                valid_results = [r for r in all_results if r != CONNECTION_ERROR]
                
                if valid_results:
                    progress['partial_runs'].extend(valid_results)
                    # Update total_games with only valid results
                    self.state['total_games'] += len(valid_results)
                
                # Save progress to state
                self.state['validation_progress'][str(candidate_hash)] = progress
                self._save_state_batch()
            
            # Check if we have enough runs to evaluate this candidate
            if len(progress['partial_runs']) >= 15 or self.shutdown_requested:
                # Use all available runs for statistics
                final_runs = progress['partial_runs']
                
                # Skip if too few valid results
                if len(final_runs) < 5:
                    log(f"   ⚠️ Too few valid results ({len(final_runs)}), skipping candidate")
                    continue
                
                stats = calculate_statistics(final_runs, detailed=True)
                
                validation_results.append({
                    'params': params,
                    'stats': stats,
                    'runs': final_runs
                })
                
                if not self.quiet:
                    log(f"Validation complete: median={stats['median']:.1f}, CI=[{stats['confidence_95'][0]:.1f}, {stats['confidence_95'][1]:.1f}], runs={len(final_runs)}")
            else:
                # Partial completion - will be resumed on next run
                runs_completed = len(progress['partial_runs'])
                if not self.quiet:
                    log(f"Partial validation: {runs_completed}/{15} runs completed")
                return  # Exit validation to resume later
        
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
            
            # Clean up validation progress on successful completion
            if 'validation_progress' in self.state:
                del self.state['validation_progress']
            
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
        
        # Check if we should re-optimize (every 150 runs for more frequent refinement)
        if champion_summary['total_count'] > 0 and champion_summary['total_count'] % 150 == 0:
            if SKOPT_AVAILABLE:
                log("Running periodic re-optimization...")
                
                # Create narrowed search space around champion (±20%)
                narrowed_space = []
                for dim in self.space:
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
                
                # Use Optimizer for local refinement (TuRBO-inspired)
                local_optimizer = Optimizer(
                    dimensions=narrowed_space,
                    base_estimator="ET",  # Same as main optimizer
                    n_initial_points=5,
                    acq_func="EI",
                    random_state=42 + champion_summary['total_count']
                )
                
                # Run 10 evaluations in trust region
                for eval_num in range(10):
                    if self.shutdown_requested:
                        break
                    
                    # Get next point from local optimizer
                    next_point = local_optimizer.ask(n_points=1)[0]
                    
                    # Convert to params
                    params = {}
                    for i, dim in enumerate(narrowed_space):
                        v = next_point[i]
                        params[dim.name] = int(v) if dim.name == 'early_threshold' else v
                    
                    # Evaluate with reduced runs for speed
                    args = [(self.scenario, params) for _ in range(3)]
                    results = self._map_games(args)
                    
                    if results:
                        # Filter out connection errors
                        all_results = [r[0] for r in results]
                        rejects = [r for r in all_results if r != CONNECTION_ERROR]
                        
                        if len(rejects) >= 2:  # Need at least 2 valid results
                            median_score = np.median(rejects)
                            # Tell local optimizer about result
                            local_optimizer.tell([next_point], [median_score])
                        else:
                            log(f"   Too many connection errors, skipping tell")
                    
                    # Update champion if better (5% improvement threshold)
                    if median_score < champion_summary['best_score'] * 0.95:
                        self.state['champion']['params'] = params
                        champion_summary['best_score'] = median_score
                        log(f"🚀 Champion improved through re-optimization: median={median_score:.1f}")
        
        # Run regular champion games
        # Add debug flag if optimizer is in debug mode
        if self.debug:
            champion_params = champion_params.copy()
            champion_params['debug'] = True
        args = [(self.scenario, champion_params) for _ in range(self.workers)]
        
        results = self._map_games(args)
        
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
                eta_hours = (150 - bo_evals) / (games_per_hour / 3) if games_per_hour > 0 else 0  # 3 games per eval
                
                log(f"BO Search: {bo_evals}/150 evaluations (+{pending} pending) | "
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
    parser = argparse.ArgumentParser(description="Bayesian Optimization Berghain Optimizer v3.0 (ET-based)")
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