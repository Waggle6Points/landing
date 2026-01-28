import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# ===============================
# 1. THE CORE SIMULATION CLASSES
# (Your original code, slightly cleaned)
# ===============================

class PacingModel:
    def control(self, state, context):
        raise NotImplementedError

class EnergyUF(PacingModel):
    def __init__(self, energyfactor, ufactor, a0, fatigue_factor, O2_intake, f):
        self.energyfactor = energyfactor
        self.ufactor = ufactor
        self.a0 = a0
        self.fatigue_factor = fatigue_factor
        self.O2_intake = O2_intake
        self.f = f

    def control(self, state, context):
        # 1. Update Energy State (Internal logic)
        # Note: In a pure simulation for fitting, we might update state outside, 
        # but we keep it here to match your logic.
        
        # Current progress (0.0 to 1.0)
        if context['total_distance'] > 0:
            progress = state['x'] / context['total_distance']
        else:
            progress = 0

        # The "Unknown" Pacing Logic
        # We assume the shape is abs(sin), but we optimize 'ufactor' (magnitude)
        pacing_component = self.ufactor * abs(np.sin(progress - 0.5))
        
        # The Energy Logic
        energy_component = self.energyfactor * state['E']
        
        a_desired = energy_component + pacing_component

        # Physical Constraints (Fatigue & Max Accel)
        a_max = self.a0 / (1.0 + self.fatigue_factor * state['F'])
        
        return np.clip(a_desired, -a_max, a_max)

class Runner:
    def __init__(self, pacing_model):
        self.state = {'x': 0.0, 'v': 0.0, 'E': 100.0, 'F': 0.0}
        self.model = pacing_model
        self.finished_time = None

def O2_from_VO2max(VO2max, kg):
    return VO2max * kg * 5.04 * 4.184 /(60 * 22400)

def simulate_single_runner(runner, race_distance=1600, dt=1.0, sigma=0.0):
    """
    Runs a simulation for ONE runner and returns their velocity profile.
    Returns: numpy array of velocities
    """
    context = {'total_distance': race_distance}
    velocities = []
    
    # Safety break to prevent infinite loops if params are bad
    max_steps = 10000 
    
    for t in range(max_steps):
        if runner.finished_time is not None:
            break
            
        state = runner.state
        
        # 1. Update Internal Physics vars (Energy/Fatigue) 
        # (Moved logic from control to here for clarity, or kept in control? 
        #  Your original code updated E inside loop, let's stick to that)
        
        dE_dt = runner.model.O2_intake - runner.model.f * state['v']
        dF_dt = 5.0 * state['v'] - 3.5
        
        # Update potential energy/fatigue pools
        state['E'] = max(state['E'] + dE_dt, 0) # Simple Euler
        state['F'] = max(state['F'] + dF_dt, 0)

        # 2. Get Control
        a_control = runner.model.control(state, context)
        
        # 3. Apply Noise (Sigma)
        a = a_control + sigma * np.random.randn()

        # 4. Update Kinematics
        state['v'] = max(state['v'] + a * dt, 0.0)
        state['x'] += state['v'] * dt
        
        # Update Energy consumed by acceleration (from your code)
        state['E'] = max(state['E'] - abs(a_control) * dt, 0.0)

        velocities.append(state['v'])

        if state['x'] >= race_distance:
            runner.finished_time = t * dt
            
    return np.array(velocities)

# ===============================
# 2. THE OPTIMIZATION ENGINE
# ===============================

def objective_function(unknowns, fixed_params, real_velocity_data, dt):
    """
    This function measures how 'bad' a set of parameters is.
    
    unknowns: [energyfactor, ufactor, fatigue_factor]
    fixed_params: {VO2max, kg, f, a0, distance}
    real_velocity_data: The actual observed data we want to match
    """
    
    # Unpack the guesses
    e_fac, u_fac, fat_fac = unknowns
    
    # Build the model with these guesses
    model = EnergyUF(
        energyfactor=e_fac, 
        ufactor=u_fac, 
        fatigue_factor=fat_fac,
        a0=fixed_params['a0'],
        O2_intake=O2_from_VO2max(fixed_params['VO2max'], fixed_params['kg']),
        f=fixed_params['f']
    )
    
    runner = Runner(model)
    
    # Run Simulation (sigma=0 for deterministic fitting)
    sim_vel = simulate_single_runner(runner, race_distance=fixed_params['dist'], dt=dt, sigma=0.0)
    
    # --- ERROR CALCULATION ---
    # We need to compare sim_vel vs real_velocity_data.
    # They might be different lengths (simulation finished earlier/later than real life).
    # We truncate to the shorter length for RMSE calculation.
    
    min_len = min(len(sim_vel), len(real_velocity_data))
    if min_len == 0: return 1e9 # Huge error if simulation failed immediately
    
    diff = sim_vel[:min_len] - real_velocity_data[:min_len]
    mse = np.mean(diff**2) # Mean Squared Error
    
    # Penalize length mismatch (if simulation time is wildly different from real time)
    len_penalty = abs(len(sim_vel) - len(real_velocity_data)) * 0.1
    
    return mse + len_penalty

# ===============================
# 3. MAIN EXECUTION
# ===============================

import pandas as pd # Make sure to import pandas at the top

if __name__ == "__main__":
    # ===============================
    # A. LOAD REAL DATA
    # ===============================
    csv_path = "Vel_Fixed.csv"  # REPLACE with your actual file path
    
    try:
        df = pd.read_csv(csv_path)
        
        # 1. Extract Velocity
        # Ensure units are meters/second. If in km/h, divide by 3.6
        # Example: real_velocity_data = df['velocity_kmh'].values / 3.6
        real_velocity_data = df['velocity'].values 
        
        # 2. Define known Physical Parameters for THIS specific runner
        # (You likely have these in a separate file or inputs)
        fixed_setup = {
            'VO2max': 55,      # Measured value
            'kg': 68,          # Measured value
            'f': 3.9,          # Measured (Power vs Velocity slope)
            'a0': 4.0,         # Measured (Max acceleration)
            'dist': 1600       # The race distance in meters
        }
        
        # 3. Handle Time Steps (dt)
        # Check your CSV time intervals. If it's 1Hz (1 row per second), dt=1.0
        # If it's variable, you might need to resample the data to a fixed grid.
        dt = 1.0 

        print(f"Loaded {len(real_velocity_data)} data points from {csv_path}")

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the path.")
        exit()

    # ===============================
    # B. DEFINE BOUNDS (Ranges to search)
    # ===============================
    # [EnergyFactor, UFactor, FatigueFactor]
    bounds = [
        (0.001, 0.2),  # Energy Factor: usually small
        (0.0, 1.0),    # U Factor: 0 to 1 implies how strong the pacing strategy is
        (0.0, 0.1)     # Fatigue Factor: usually very small decay
    ]

    print("\nStarting Parameter Estimation for Real Runner...")

    # ===============================
    # C. RUN OPTIMIZER
    # ===============================
    # We pass the loaded 'real_velocity_data' into the args tuple
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(fixed_setup, real_velocity_data, dt), 
        strategy='best1bin',
        maxiter=100,      # Increased iterations for real noisy data
        popsize=20,       # Increased population for better search
        disp=True 
    )

    # ===============================
    # D. VISUALIZE FIT
    # ===============================
    print("\n" + "="*30)
    print("OPTIMIZATION RESULTS")
    print(f"Found Values: {np.round(result.x, 5)}")
    print(f"RMSE Error:   {result.fun:.5f}")
    
    found_e, found_u, found_fat = result.x
    
    # Re-run simulation with found values to plot
    best_model = EnergyUF(
        found_e, found_u, fixed_setup['a0'], found_fat, 
        O2_from_VO2max(fixed_setup['VO2max'], fixed_setup['kg']), 
        fixed_setup['f']
    )
    best_runner = Runner(best_model)
    
    # Simulate
    sim_vel = simulate_single_runner(best_runner, race_distance=fixed_setup['dist'], dt=dt)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot real data
    plt.plot(real_velocity_data, 'k', alpha=0.5, label='Actual Runner Data')
    
    # Plot simulation
    # Ensure x-axis alignment if lengths differ
    t_sim = np.arange(len(sim_vel)) * dt
    plt.plot(t_sim, sim_vel, 'r--', linewidth=2, label='Fitted Model')
    
    plt.title(f"Model Fit: E-Fac={found_e:.3f}, U-Fac={found_u:.3f}, Fat={found_fat:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.show()