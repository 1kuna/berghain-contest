# Berghain Contest Optimizer

Smart phase-based optimizer for the Berghain admissions challenge with automatic parameter tuning.

## Quick Start (Raspberry Pi / Linux)

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd berghain-contest
```

### 2. Make the script executable (ONE TIME ONLY)
```bash
chmod +x start.sh
```

### 3. Run the optimizer
```bash
./start.sh
```

That's it! The script will:
- Create a virtual environment if needed
- Install all dependencies automatically
- Ask you which scenario to optimize (1, 2, or 3)
- Start optimizing with safe rate limits

## Manual Setup (if you prefer)

### Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run directly
```bash
python algo1.py --scenario 1 --workers 1
```

## Command Options

```bash
python algo1.py [options]
```

Options:
- `--scenario N`: Choose scenario 1, 2, or 3 (required)
- `--workers N`: Number of parallel workers (default: 1, keep at 1 for rate limits!)
- `--target N`: Stop when reaching this rejection count (optional)
- `--debug`: Show detailed parameter information
- `--quiet`: Suppress all but critical messages

## How It Works

The optimizer uses three phases:
1. **Grid Search**: Tests comprehensive parameter combinations
2. **Statistical Validation**: Selects the best performer with confidence
3. **Infinite Improvement**: Runs the champion seeking new records

## Files Created

- `berghain_s1_state.json` - Scenario 1 progress (auto-saved)
- `berghain_s2_state.json` - Scenario 2 progress (auto-saved)
- `berghain_s3_state.json` - Scenario 3 progress (auto-saved)

You can safely interrupt with Ctrl+C anytime - progress is saved automatically.

## Performance

- Optimized for Raspberry Pi (single core, low memory)
- ~35-50% faster than v3 with connection pooling
- Graceful shutdown within 0.5 seconds
- Auto-saves every 10 games

## Troubleshooting

If you get permission denied:
```bash
chmod +x start.sh
```

If you get "python3: command not found":
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

If numpy/scipy installation is slow on Raspberry Pi:
- Be patient, it can take 10-20 minutes on older Pi models
- Consider using pre-compiled wheels: `sudo apt-get install python3-numpy python3-scipy`

## Tips

- Keep `--workers 1` to avoid rate limiting
- Let it run overnight for best results
- Check the JSON files to see all discovered parameters
- Use `--debug` to see which parameters work best