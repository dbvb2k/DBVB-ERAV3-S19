import numpy as np
from colorama import init, Fore, Style
import argparse
import sys
from datetime import datetime

# Initialize colorama
init()

class Logger:
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log_file = open(filename, 'w') if filename else None
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            # Remove color codes for file output
            clean_message = message.replace(Fore.YELLOW, '').replace(Fore.GREEN, '').replace(Fore.RED, '').replace(Style.RESET_ALL, '')
            self.log_file.write(clean_message)
            self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = 1.0  # discount factor
        self.theta = 1e-4  # convergence threshold
        
        # Initialize value function
        self.V = np.zeros(self.n_states)
        
        # Initialize rewards (-1 for all states except terminal state)
        self.rewards = -np.ones(self.n_states)
        self.rewards[self.n_states - 1] = 0  # terminal state has 0 reward
        
    def get_state_coords(self, state):
        """Convert state number to grid coordinates"""
        row = state // self.size
        col = state % self.size
        return row, col
    
    def get_state_number(self, row, col):
        """Convert grid coordinates to state number"""
        return row * self.size + col
    
    def get_next_state(self, state, action):
        """Get next state given current state and action"""
        row, col = self.get_state_coords(state)
        
        if action == 'up':
            next_row = max(0, row - 1)
            next_col = col
        elif action == 'down':
            next_row = min(self.size - 1, row + 1)
            next_col = col
        elif action == 'left':
            next_row = row
            next_col = max(0, col - 1)
        else:  # right
            next_row = row
            next_col = min(self.size - 1, col + 1)
            
        return self.get_state_number(next_row, next_col)
    
    def print_values(self, color=None):
        """Print the value function as a grid with optional color"""
        values_grid = self.V.reshape((self.size, self.size))
        if color:
            print(f"{color}{np.array2string(values_grid, precision=8, suppress_small=True)}{Style.RESET_ALL}")
        else:
            print(np.array2string(values_grid, precision=8, suppress_small=True))
    
    def value_iteration(self, print_interval=50):
        """Perform value iteration until convergence"""
        print(f"\nStarting Value Iteration for {self.size}x{self.size} grid...")
        print("Initial state values:")
        self.print_values()
        
        iteration = 0
        while True:
            delta = 0
            V_new = self.V.copy()
            
            # For each state
            for s in range(self.n_states - 1):  # excluding terminal state
                if s == self.n_states - 1:  # skip terminal state
                    continue
                    
                # Calculate value for each action
                action_values = []
                for action in self.actions:
                    s_next = self.get_next_state(s, action)
                    # Equal probability (0.25) for each action
                    value = 0.25 * (self.rewards[s] + self.gamma * self.V[s_next])
                    action_values.append(value)
                
                # Update value function with sum of action values
                V_new[s] = sum(action_values)
                
                # Track maximum change
                delta = max(delta, abs(V_new[s] - self.V[s]))
            
            # Update value function
            self.V = V_new
            iteration += 1
            
            # Print progress at intervals in yellow
            if iteration % print_interval == 0:
                print(f"\n{Fore.YELLOW}Iteration {iteration}:")
                print(f"Maximum change (delta): {delta:.6f}")
                print("Current value function:")
                self.print_values(Fore.YELLOW)
                print(Style.RESET_ALL)
            
            # Check for convergence
            if delta < self.theta:
                print(f"\nConverged! Final delta: {delta:.6f}")
                break
                
        return iteration

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Value Iteration for GridWorld')
    parser.add_argument('-s', '--size', 
                      type=int, 
                      default=4,
                      help='Size of the grid (default: 4)')
    parser.add_argument('-o', '--output',
                      type=str,
                      help='Output file to log results')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    if args.output:
        sys.stdout = Logger(args.output)
    
    # Print start time
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate grid size
    if args.size < 2:
        print(f"{Fore.RED}Error: Grid size must be at least 2x2{Style.RESET_ALL}")
        return
    
    # Create and solve the GridWorld
    env = GridWorld(size=args.size)
    iterations = env.value_iteration(print_interval=100)  # Print every 100 iterations
    
    print(f"\nValue Iteration completed after {iterations} iterations")
    print(f"\n{Fore.GREEN}Final Value Function:{Style.RESET_ALL}")
    env.print_values(Fore.GREEN)
    
    # Print end time
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Close the logger
    if args.output:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main() 