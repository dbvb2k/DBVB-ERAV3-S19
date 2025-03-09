# Value Iteration in GridWorld

This project implements Value Iteration algorithm for solving a 4x4 GridWorld problem using dynamic programming. The implementation demonstrates how an agent can learn the optimal values for each state in a simple grid environment.

## Problem Description

The environment is a 4x4 grid where:
- The agent starts at the top-left corner (state 0)
- The goal is to reach the bottom-right corner (state 15)
- The agent can move in four directions: up, down, left, or right
- Each action has an equal probability of 0.25
- Each move has a reward of -1
- The terminal state (goal) has a reward of 0
- There are no obstacles in the grid

## Mathematical Background

The implementation uses the Bellman equation for value iteration:

V(s) = max_a ∑[P(s'|s,a) * (R(s,a,s') + γV(s'))]

Where:
- V(s) is the value of state s
- P(s'|s,a) is the transition probability (0.25 for all actions)
- R(s,a,s') is the reward (-1 for all moves)
- γ (gamma) is the discount factor (set to 1.0 in this implementation)
- s' is the next state

## Requirements

- Python 3.8+
- NumPy
- Colorama

Install the required packages using:
```bash
pip install numpy colorama
```

## File Structure

```
.
├── README.md
├── actual_output.md
└── value_iteration.py
```

## Implementation Details

The code consists of a `GridWorld` class with the following key methods:
- `__init__`: Initializes the environment parameters
- `get_state_coords`: Converts state number to grid coordinates
- `get_state_number`: Converts grid coordinates to state number
- `get_next_state`: Determines the next state given current state and action
- `print_values`: Displays the value function as a grid
- `value_iteration`: Implements the main value iteration algorithm

## Usage

The program can be run with optional parameters for grid size and output logging:

```bash
# Run with default 4x4 grid
python value_iteration.py

# Run with custom grid size (e.g., 5x5)
python value_iteration.py -s 5

# Run and save output to a file
python value_iteration.py -o output.txt

# Run with both custom size and output file
python value_iteration.py -s 5 -o output.txt

# Show help message
python value_iteration.py -h
```

Command-line options:
- `-s, --size`: Size of the grid (default: 4, minimum: 2)
- `-o, --output`: Output file to log results (optional)

The program will:
1. Initialize the value function to zeros
2. Perform value iteration until convergence
3. Display progress every 100 iterations in yellow
4. Show the final converged values in green
5. If an output file is specified, log all console output to the file (without color codes)

## Output Format

The output shows:
- Execution start time
- Initial state values (all zeros)
- Progress updates every 100 iterations (in yellow on console, plain text in file)
- Maximum change in values (delta) at each printed iteration
- Final converged values (in green on console, plain text in file)
- Execution end time

Example output:
```
Execution started at: 2024-03-15 10:30:00

Starting Value Iteration for 4x4 grid...
Initial state values:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

Iteration 100:
Maximum change (delta): 0.165648
Current value function:
[[-51.23249022 -49.55722337 -46.91675657 -44.74768574]
 [-49.55722337 -47.14615743 -43.02928474 -39.13065581]
 [-46.91675657 -43.02928474 -35.45329583 -26.09141026]
 [-44.74768574 -39.13065581 -26.09141026   0.        ]]

...

Final Value Function:
[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]

Execution completed at: 2024-03-15 10:30:05
```

## Actual Output Samples

For actual program outputs with screenshots for different grid sizes, please see [Actual Outputs](actual_output.md).

## Convergence

The algorithm continues until the maximum change in values (delta) between iterations is less than the convergence threshold (θ = 1e-4). This ensures that the values have stabilized and represents the optimal solution.

## Interpretation of Results

The final values show:
- The terminal state (bottom-right) has value 0
- States further from the goal have more negative values
- Values represent the expected cumulative reward to reach the goal
- The top-left state has the most negative value as it's furthest from the goal

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open-source and available under the MIT License. 