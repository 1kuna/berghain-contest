#!/bin/bash

# Berghain Contest Optimizer Launcher for Raspberry Pi
# Automatically manages virtual environment and dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Berghain Contest Optimizer v4.0     ${NC}"
echo -e "${GREEN}       Raspberry Pi Edition             ${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Detect existing environment
ENV_TYPE=""
ENV_PATH=""

# Check for conda environment
if [ ! -z "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    ENV_TYPE="conda"
    ENV_PATH="$CONDA_PREFIX"
    echo -e "${GREEN}✓ Detected active conda environment: $CONDA_DEFAULT_ENV${NC}"
    
# Check for existing venv in current directory
elif [ -d "venv" ]; then
    ENV_TYPE="venv"
    ENV_PATH="venv"
    echo -e "${GREEN}✓ Found existing virtual environment: venv/${NC}"
    
# Check for existing .venv in current directory
elif [ -d ".venv" ]; then
    ENV_TYPE="venv"
    ENV_PATH=".venv"
    echo -e "${GREEN}✓ Found existing virtual environment: .venv/${NC}"
    
# Check if already in a venv
elif [ ! -z "$VIRTUAL_ENV" ]; then
    ENV_TYPE="venv"
    ENV_PATH="$VIRTUAL_ENV"
    echo -e "${GREEN}✓ Already in virtual environment: $VIRTUAL_ENV${NC}"
    
else
    # No environment found, create one
    echo -e "${YELLOW}No virtual environment detected. Creating one...${NC}"
    
    # Check if conda is available
    if command_exists conda; then
        read -p "Use conda (c) or venv (v)? [v]: " env_choice
        env_choice=${env_choice:-v}
        
        if [ "$env_choice" = "c" ]; then
            ENV_TYPE="conda"
            echo "Creating conda environment 'berghain'..."
            conda create -n berghain python=3.9 -y
            eval "$(conda shell.bash hook)"
            conda activate berghain
            ENV_PATH="$CONDA_PREFIX"
        else
            ENV_TYPE="venv"
            ENV_PATH="venv"
        fi
    else
        ENV_TYPE="venv"
        ENV_PATH="venv"
    fi
    
    if [ "$ENV_TYPE" = "venv" ] && [ ! -d "$ENV_PATH" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$ENV_PATH"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    fi
fi

# Activate environment if needed
if [ "$ENV_TYPE" = "venv" ]; then
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Activating virtual environment..."
        source "$ENV_PATH/bin/activate"
    fi
elif [ "$ENV_TYPE" = "conda" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate berghain
fi

# Check and install requirements
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Check if all required packages are installed
MISSING_DEPS=false
if ! check_package numpy; then
    MISSING_DEPS=true
    echo "  ❌ numpy not installed"
else
    echo "  ✓ numpy installed"
fi

if ! check_package scipy; then
    MISSING_DEPS=true
    echo "  ❌ scipy not installed"
else
    echo "  ✓ scipy installed"
fi

if ! check_package requests; then
    MISSING_DEPS=true
    echo "  ❌ requests not installed"
else
    echo "  ✓ requests installed"
fi

# Install missing dependencies
if [ "$MISSING_DEPS" = true ]; then
    echo -e "\n${YELLOW}Installing missing dependencies...${NC}"
    echo "This may take a while on Raspberry Pi..."
    
    # Upgrade pip first
    pip3 install --upgrade pip
    
    # Install requirements
    pip3 install -r requirements.txt
    
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
fi

# Check which algorithms are available
ALGOS_AVAILABLE=""
if [ -f "algo1.py" ]; then
    ALGOS_AVAILABLE="${ALGOS_AVAILABLE}1"
fi
if [ -f "algo2.py" ]; then
    ALGOS_AVAILABLE="${ALGOS_AVAILABLE}2"
fi
if [ -f "algo3.py" ]; then
    ALGOS_AVAILABLE="${ALGOS_AVAILABLE}3"
fi

if [ -z "$ALGOS_AVAILABLE" ]; then
    echo -e "${RED}Error: No algorithm files (algo1.py, algo2.py, algo3.py) found${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}         Configuration Setup            ${NC}"
echo -e "${GREEN}========================================${NC}"

# Ask for scenario
echo -e "\n${YELLOW}Select scenario to optimize:${NC}"
echo "  1) Scenario 1 (K=2, 6D problem)"
echo "  2) Scenario 2 (K=4, 8D problem)"
echo "  3) Scenario 3 (K=6, 10D problem)"
read -p "Enter choice [1-3]: " scenario

# Validate scenario input
while [[ ! "$scenario" =~ ^[1-3]$ ]]; do
    echo -e "${RED}Invalid choice. Please enter 1, 2, or 3.${NC}"
    read -p "Enter choice [1-3]: " scenario
done

# Ask for algorithm
echo -e "\n${YELLOW}Select optimization algorithm:${NC}"
if [[ "$ALGOS_AVAILABLE" == *"1"* ]]; then
    echo "  1) Grid Search (algo1.py)"
    echo "     - Exhaustive search, guaranteed to find optimal"
    echo "     - Best for: Small search spaces, high reliability needs"
    echo "     - Speed: Slowest but most thorough"
fi
if [[ "$ALGOS_AVAILABLE" == *"2"* ]]; then
    echo "  2) Bayesian Optimization with Forest (algo2.py)"
    echo "     - Smart search using Random Forest surrogate"
    echo "     - Best for: Balanced speed/quality, medium dimensions"
    echo "     - Speed: 2-3x faster than grid search"
fi
if [[ "$ALGOS_AVAILABLE" == *"3"* ]]; then
    echo "  3) Bayesian Optimization with ET Optimizer (algo3.py)"
    echo "     - Advanced BO with Extra Trees estimator"
    echo "     - Best for: High dimensions (K=6), noisy environments"
    echo "     - Speed: 20-30% faster than algo2 in high-D"
fi

# Build valid choices string
VALID_ALGOS=""
[[ "$ALGOS_AVAILABLE" == *"1"* ]] && VALID_ALGOS="${VALID_ALGOS}1"
[[ "$ALGOS_AVAILABLE" == *"2"* ]] && VALID_ALGOS="${VALID_ALGOS}2"
[[ "$ALGOS_AVAILABLE" == *"3"* ]] && VALID_ALGOS="${VALID_ALGOS}3"

# Read algorithm choice
read -p "Enter choice [${VALID_ALGOS// /,}]: " algorithm

# Validate algorithm input
while [[ ! "$algorithm" =~ ^[1-3]$ ]] || [[ ! "$ALGOS_AVAILABLE" == *"$algorithm"* ]]; do
    echo -e "${RED}Invalid choice or algorithm not available.${NC}"
    read -p "Enter choice [${VALID_ALGOS// /,}]: " algorithm
done

# Set algorithm file based on choice
ALGO_FILE="algo${algorithm}.py"

# Ask for target (optional)
echo -e "\n${YELLOW}Target rejection count (optional):${NC}"
echo "Leave empty to run indefinitely until best possible score"
read -p "Target [empty for none]: " target

# Validate target input
if [ ! -z "$target" ]; then
    if ! [[ "$target" =~ ^[0-9]+$ ]]; then
        echo -e "${YELLOW}Invalid target. Running without target.${NC}"
        target=""
    fi
fi

# Ask for mode
echo -e "\n${YELLOW}Select output mode:${NC}"
echo "  1) Normal (show progress)"
echo "  2) Debug (detailed output)"
echo "  3) Quiet (minimal output)"
read -p "Enter choice [1-3, default=1]: " mode
mode=${mode:-1}

# Build command
CMD="python3 $ALGO_FILE --scenario $scenario --workers 1"

if [ ! -z "$target" ]; then
    CMD="$CMD --target $target"
fi

case $mode in
    2)
        CMD="$CMD --debug"
        ;;
    3)
        CMD="$CMD --quiet"
        ;;
esac

# Ask about auto-restart
echo -e "\n${YELLOW}Enable auto-restart on crash?${NC}"
read -p "Auto-restart? [y/N]: " restart
restart=${restart:-n}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}         Starting Optimizer             ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "  Algorithm: $ALGO_FILE"
echo "  Scenario: $scenario"
if [ ! -z "$target" ]; then
    echo "  Target: $target rejections"
else
    echo "  Target: None (run indefinitely)"
fi
echo "  Workers: 1 (rate limit safe)"
case $mode in
    2) echo "  Mode: Debug" ;;
    3) echo "  Mode: Quiet" ;;
    *) echo "  Mode: Normal" ;;
esac
echo "  Auto-restart: $([ "$restart" = "y" ] && echo "Yes" || echo "No")"
echo
echo -e "${YELLOW}Press Ctrl+C to stop gracefully${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Function to run the optimizer
run_optimizer() {
    echo -e "${GREEN}Starting optimizer...${NC}"
    echo "Command: $CMD"
    echo
    
    if [ "$restart" = "y" ] || [ "$restart" = "Y" ]; then
        # Run with auto-restart
        while true; do
            $CMD
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo -e "\n${GREEN}Optimizer exited normally${NC}"
                break
            elif [ $EXIT_CODE -eq 130 ]; then
                echo -e "\n${YELLOW}Interrupted by user (Ctrl+C)${NC}"
                break
            else
                echo -e "\n${YELLOW}Optimizer crashed (exit code: $EXIT_CODE)${NC}"
                echo "Restarting in 5 seconds..."
                sleep 5
                echo
            fi
        done
    else
        # Run without auto-restart
        $CMD
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "\n${GREEN}Optimizer exited normally${NC}"
        elif [ $EXIT_CODE -eq 130 ]; then
            echo -e "\n${YELLOW}Interrupted by user (Ctrl+C)${NC}"
        else
            echo -e "\n${RED}Optimizer exited with error code: $EXIT_CODE${NC}"
        fi
    fi
}

# Trap Ctrl+C to ensure clean shutdown
trap 'echo -e "\n${YELLOW}Shutting down...${NC}"; exit 130' INT

# Run the optimizer
run_optimizer

# Deactivate environment if we activated it
if [ "$ENV_TYPE" = "venv" ] && [ ! -z "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
elif [ "$ENV_TYPE" = "conda" ] && [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate 2>/dev/null || true
fi

echo -e "${GREEN}Goodbye!${NC}"