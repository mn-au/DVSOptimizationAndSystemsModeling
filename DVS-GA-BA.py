# Overview:
# This code implements a simulation and optimization model for a Dynamic Voltage Scaling (DVS) server.
# It uses SimPy for the simulation environment and applies a Genetic Algorithm (GA) combined with Bayesian optimization.
# The simulation handles job arrivals, state transitions (OFF, SETUP, BUSY, SCALED, IDLE), and tracks performance metrics.

import simpy  
import numpy as np 
import scipy.stats as stats  
import random 
import copy  
import statistics 
from sklearn.gaussian_process import GaussianProcessRegressor  # Surrogate modeling
from sklearn.gaussian_process.kernels import Matern  # Kernel for the GP
from scipy.stats import norm  # Normal distribution functions
from math import erf, sqrt, exp, pi  # Math functions

# Helper Function to calculate statistics: mean and confidence interval for the given dataset.
def calculate_statistics(data, confidence_level=0.95):

    # Compute the mean, standard error, and degrees of freedom
    mean_val = np.mean(data)
    degrees_freedom = len(data) - 1
    sample_standard_error = stats.sem(data)

    # Compute the confidence interval using the t-distribution.
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean_val, sample_standard_error)

    return mean_val, confidence_interval

# Class representing a Dynamic Voltage Scaling (DVS) server simulation.
class DVSServer:
    # Initialize the server with various simulation parameters and initial state.
    def __init__(self, env, arrival_rate, setup_time, alpha, scale_speed_factor,
                 turn_on_threshold, scale_threshold, service_rate, idle_power):
        
        """
        Initialize the DVSServer with the simulation environment and parameters.

        Parameters:
            env (simpy.Environment): The simulation environment.
            arrival_rate (float): The rate at which jobs arrive.
            alpha (float): The timeout duration in the IDLE state.
            scale_speed_factor (float): The factor by which the service rate is scaled in the SCALED state.
            service_rate (float): The rate at which jobs are processed.
            idle_power (float): The power consumption when the server is in the IDLE state.
            turn_on_threshold (int): The queue length threshold required to turn the server ON.
            scale_threshold (int): The queue length threshold required to switch to the SCALED state.
            setup_time (float): The time required for the server to transition from OFF to SETUP.
        """
        # Store the simulation environment and parameters.
        self.env = env 
        self.arrival_rate = arrival_rate 
        self.alpha = alpha 
        self.scale_speed_factor = scale_speed_factor  
        self.service_rate = service_rate 
        self.idle_power = idle_power  
        self.turn_on_threshold = turn_on_threshold  
        self.scale_threshold = scale_threshold 
        self.setup_time = setup_time  

        # Initialize logging and metrics.
        self.log = []  
        self.response_times = []  
        self.server_off_counter = 0 
        self.jobs_slow_speed_counter = 0 
        self.jobs_fast_speed_counter = 0  

        # Set the initial server state and energy metrics.
        self.state = 'OFF' 
        self.job_queue = simpy.PriorityStore(env)  # Priority queue to manage incoming jobs.

        # Start the main server process and job arrival process.
        self.server_process_instance = self.env.process(self.server_process()) 
        self.job_arrival_process_instance = self.env.process(self.job_arrival_process())  
        self.idle_process_instance = None  #  idle process placeholder.

        # Dictionary to track the time spent in each server state.
        self.state_times = {'OFF': 0, 'SETUP': 0, 'BUSY': 0, 'SCALED': 0, 'IDLE': 0}
        self.last_change = 0  # timestamp of the last state change.
        self.total_energy = 0  # accumulator for total energy consumed.
        self.powers = []  # records the power consumption values.

    # Log a message along with the current simulation time.
    def log_message(self, message):
        timestamp = self.env.now 
        self.log.append((timestamp, message))  

    # Calculate the instantaneous power consumption based on the current server state.
    def calculate_power(self):
        if self.state == 'OFF':
            return 0  
        elif self.state in ['SETUP', 'BUSY']:
            return 1 
        elif self.state == 'SCALED':
            # experimenting with this value: power consumption is reduced in this scenario.
            return 1 * 0.30000000000000004**2
        elif self.state == 'IDLE':
            return self.idle_power 

    # Main server process: manages state transitions and job processing.
    def server_process(self):

        # Update energy consumption and time spent in the current state as we loop.
        while True:

            # Determine the current power consumption
            power = self.calculate_power()
            self.powers.append(power) 

            # Updates the total energy consumed by the server as well as the time spent in the current state.
            self.total_energy += power * (self.env.now - self.last_change)  
            self.state_times[self.state] += self.env.now - self.last_change

            self.last_change = self.env.now  # RESETS the time marker for state change.

            # Execute behavior based on the current state.
            if self.state == 'OFF':
                # If there are enough jobs waiting, transition to SETUP.
                if len(self.job_queue.items) >= self.turn_on_threshold:
                    yield from self.transition_to_setup()
                else:
                    try:
                        yield self.env.timeout(1)  # Wait for 1 time unit if no jobs trigger a state change (look over this and check if there is a logic error here)
                    except simpy.Interrupt:
                        pass 

            elif self.state == 'SETUP':
                # After setup, decide whether to operate in BUSY or SCALED state.
                self.transition_to_busy_or_scaled()

            elif self.state in ['BUSY', 'SCALED']:
                if len(self.job_queue.items) > 0:
                    yield from self.process_job()  # Process a job if one is available.
                else:
                    self.transition_to_idle()  # Transition to IDLE if there are no jobs.
            elif self.state == 'IDLE':
                yield from self.idle_to_off_or_busy()  # In IDLE state, wait for a job or timeout.

    # Transition from OFF state to SETUP state by waiting for the setup time.
    def transition_to_setup(self):
        try:
            yield self.env.timeout(self.setup_time)  # Simulate the setup period.
            self.log_message("Finished setup time.")
        except simpy.Interrupt:
            self.log_message("Interrupted during setup time.")

        self.state = 'SETUP'
        self.log_message("Transitioned to SETUP state.")

    # Decide whether to transition to BUSY or SCALED state based on the job queue length.
    def transition_to_busy_or_scaled(self):
        if len(self.job_queue.items) >= self.scale_threshold:
            self.state = 'SCALED' 
            self.log_message("Transitioned to SCALED state.")
        else:
            self.state = 'BUSY' 
            self.log_message("Transitioned to BUSY state.")

    # Transition to IDLE state when there are no jobs in the queue.
    def transition_to_idle(self):
        self.state = 'IDLE'
        self.log_message("Transitioned to IDLE state.")

    # Process a single job from the job queue.
    def process_job(self):
        # Get the job from the queue; the job is stored as a tuple containing the arrival time.
        job_arrival_time, = yield self.job_queue.get()

        # Determine the service duration and simulate job processing.
        service_duration = self.get_service_duration() 
        yield self.env.timeout(service_duration) 
        
        # Calculate the total response time for the job and store it.  
        total_response_time = self.env.now - job_arrival_time 
        self.response_times.append(total_response_time)  
        
        # Update counters based on whether the job was processed in BUSY (slow) or SCALED (fast) state.
        if self.state == 'BUSY':
            self.jobs_slow_speed_counter += 1
            self.log_message("Job processed at slow speed.")
        else:  # This applies for the SCALED state.
            self.jobs_fast_speed_counter += 1
            self.log_message("Job processed at fast speed.")

    # Determine the service duration using an exponential distribution.
    def get_service_duration(self):
        if self.state == 'SCALED':
            # In SCALED state, the service rate is increased by the scale speed factor.
            return np.random.exponential(1.0 / (self.service_rate * self.scale_speed_factor))
        else:  # In BUSY state, use the standard service rate.
            return np.random.exponential(1.0 / self.service_rate)
    
    # In the IDLE state, wait for a timeout. If no job arrives, transition to OFF; if interrupted, transition to active state.
    def idle_to_off_or_busy(self):
        try:
            yield self.env.timeout(1/self.alpha)  # Wait for a timeout based on the alpha parameter.
            self.state = 'OFF'  
            self.log_message("Transitioned to OFF state from IDLE due to no job arrivals.")
            self.server_off_counter += 1  # Increment the counter for server turning off.
        except simpy.Interrupt:
            self.transition_to_busy_or_scaled()  # If interrupted by a job arrival, go back to BUSY or SCALED.

    # Continuously generate job arrivals based on an exponential inter-arrival time.
    def job_arrival_process(self):
        while True:
            # Wait for the next job arrival time.
            yield self.env.timeout(np.random.exponential(1.0 / self.arrival_rate))

            # Add a new job to the queue. The tuple contains the arrival time.
            self.job_queue.put((self.env.now,))
            self.log_message("Job arrived.")

            # If the server is OFF or IDLE, interrupt the server process to process the new job.
            if self.state in ['OFF', 'IDLE']:
                self.server_process_instance.interrupt()

    # Run the simulation for a specified period and compute performance metrics.
    def run_simulation(self, run_time, warm_up_period=100):
        self.env.run(until=run_time)
        self.state_times[self.state] += self.env.now - self.last_change

        # Calculate the percentage of time spent in each state.
        state_percentages = {state: time / self.env.now * 100 for state, time in self.state_times.items()}

        # Calculate response times and power consumption, excluding the warm-up period.
        E_R, confidence_interval_R = calculate_statistics(self.response_times[warm_up_period:])
        E_P, confidence_interval_P = calculate_statistics(self.powers[warm_up_period:])
        
        # Return the computed metrics along with various counters and the log.
        return (E_R, E_P, confidence_interval_R, confidence_interval_P, state_percentages, 
                self.server_off_counter, self.jobs_slow_speed_counter, self.jobs_fast_speed_counter, self.log)

# ---------------------------
# GA + Bayesian Optimization Components
# ---------------------------

# Global evaluations: list to store (parameter vector, fitness) tuples.
global_evaluations = []

# Parameter ranges
PARAM_RANGES = {
    "arrival_rate": [0.2, 0.5, 0.95],
    "setup_time": [1, 5, 10, 20],
    "scale_speed_factor": [1.5, 2, 3],
    "turn_on_threshold": list(range(1, 21)),
    "scale_threshold": list(range(1, 21)),
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "service_rate": [1.0],
    "idle_power": [0.6]
}

# Encode an individual's parameters into a vector.
def encode_ind(ind):
    return np.array([
        ind["arrival_rate"],
        ind["setup_time"],
        ind["scale_speed_factor"],
        ind["turn_on_threshold"],
        ind["scale_threshold"],
        ind["alpha"]
    ])

# Decode a vector back into a parameter dictionary by selecting the nearest allowed value.
# need to check if argmin is the best way to do this.
def decode_vec(vec):
    new_ind = {}
    keys = ["arrival_rate", "setup_time", "scale_speed_factor", "turn_on_threshold", "scale_threshold", "alpha"]
    for i, k in enumerate(keys):
        arr = np.array(PARAM_RANGES[k])
        idx = np.argmin(np.abs(arr - vec[i]))
        new_ind[k] = arr[idx]
    new_ind["service_rate"] = 1.0
    new_ind["idle_power"] = 0.6
    return new_ind

# Simulate an individual configuration and return its fitness and performance metrics.
def simulate_individual(params, sim_time=15000, warm_up=100):
    # Create a new simulation environment and server instance.
    env = simpy.Environment()
    server = DVSServer(env, **params)

    # Run the simulation and collect the performance metrics.
    (
        E_R, 
        E_P, 
        confidence_interval_R, 
        confidence_interval_P, 
        state_percentages, 
        off_counter,
        slow_counter, 
        fast_counter, 
        logs
    ) = server.run_simulation(sim_time, warm_up)

    # Compute fitness
    fit = E_R * E_P

    # Record the evaluation
    vec = encode_ind(params)
    global_evaluations.append((vec, fit))

    # return the fitness value and performance metrics(slow and fast).
    return fit, E_R, E_P, slow_counter, fast_counter

# GA hyperparameters.
POP_SIZE = 20      # Population size.
NUM_GEN = 50       # Number of generations.
MUT_RATE = 0.1     # Mutation rate.
ELITE_RATE = 0.1   # Elitism rate (fraction of best individuals carried over).

# Generate a random individual from the parameter ranges.
def random_individual():
    return {
        "arrival_rate": random.choice(PARAM_RANGES["arrival_rate"]),
        "setup_time": random.choice(PARAM_RANGES["setup_time"]),
        "scale_speed_factor": random.choice(PARAM_RANGES["scale_speed_factor"]),
        "turn_on_threshold": random.choice(PARAM_RANGES["turn_on_threshold"]),
        "scale_threshold": random.choice(PARAM_RANGES["scale_threshold"]),
        "alpha": random.choice(PARAM_RANGES["alpha"]),
        "service_rate": 1.0,
        "idle_power": 0.6
    }

# Generate an initial population of individuals by randomly selecting parameter values.
def initialize_population(n):
    return [random_individual() for _ in range(n)]

# Select an individual from the population using inverse-proportional fitness values.
def selection(pop, fits):
    arr = np.array(fits)
    
    # If all fitness values are infinite (e.g., due to bad solutions), fall back to a random selection.
    if np.all(np.isinf(arr)):
        return random.choice(pop)
    
    # Compute selection probabilities inversely proportional to fitness values.
    with np.errstate(divide='ignore'):  # Suppress divide-by-zero warnings.
        p = 1.0 / (arr + 1e-9)  # Add a small constant to prevent division by zero.

    # Normalize probabilities so they sum to 1.
    p /= np.sum(p)

    # If the probability array contains NaN values (numerical instability), fall back to random selection.
    if np.isnan(p).any():
        return random.choice(pop)

    # Perform roulette wheel selection: individuals with lower fitness have a higher probability of selection.
    idx = np.random.choice(len(pop), p=p)
    return pop[idx]

# Perform crossover (recombination) between two parents to create two offspring.
def crossover(p1, p2):
    c1, c2 = {}, {}  # Initialize empty dictionaries for offspring.

    # Iterate over each parameter in the parents.
    for k in p1.keys():
        if random.random() < 0.5:  # Randomly choose whether to inherit from parent 1 or parent 2.
            c1[k] = p1[k]
            c2[k] = p2[k]
        else:
            c1[k] = p2[k]
            c2[k] = p1[k]

    return c1, c2  # Return the two offspring with mixed parameter values.

# Apply mutation to an individual by randomly selecting and changing one parameter.
def mutate(ind):
    mutated = copy.deepcopy(ind)  # Create a copy of the individual to avoid modifying the original.
    
    # Exclude 'service_rate' and 'idle_power' from mutation since they remain fixed.
    keys = [k for k in PARAM_RANGES.keys() if k not in ["service_rate", "idle_power"]]
    
    # Select a random parameter to mutate.
    param = random.choice(keys)
    
    # Assign a new random value from the allowed range of that parameter.
    mutated[param] = random.choice(PARAM_RANGES[param])
    
    return mutated  # Return the mutated individual.

# Train a Gaussian Process (GP) surrogate model using previous fitness evaluations.
def fit_surrogate(evals):
    valid = [(v, f) for v, f in evals if np.isfinite(f)]  # Filter out invalid (infinite) fitness values.
    
    # If there are too few data points, Bayesian optimization cannot proceed.
    if len(valid) < 5:
        raise ValueError("Not enough valid data to fit GP.")
    
    # Extract input feature vectors (parameters) and corresponding fitness values.
    X = np.array([v for v, f in valid])
    y = np.array([f for v, f in valid])

    # Define the kernel function for the Gaussian Process (Matern kernel with tunable smoothness).
    kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3))
    
    # Create and fit a Gaussian Process model with the given data.
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    gp.fit(X, y)
    
    return gp  # Return the trained GP model.

# Compute the expected improvement (EI) at a given parameter setting using the Gaussian Process model.
def expected_improvement(x, gp, best_fit):
    x = x.reshape(1, -1)  # Reshape input vector for the model.
    
    # Predict the mean and standard deviation of the fitness at x using the Gaussian Process.
    mu, sig = gp.predict(x, return_std=True)
    mu, sig = mu[0], sig[0]  # Extract scalar values.
    
    # If the model is highly confident (zero variance), return zero improvement.
    if sig == 0:
        return 0.0
    
    # Compute improvement over the best observed fitness.
    imp = best_fit - mu
    Z = imp / (sig + 1e-9)  # Standardize the improvement (avoiding division by zero).
    
    # Compute the probability density function (PDF) and cumulative density function (CDF).
    cdfZ = norm.cdf(Z)
    pdfZ = norm.pdf(Z)

    # Compute the expected improvement using the formula:
    return imp * cdfZ + sig * pdfZ  # Balances exploration (high uncertainty) and exploitation (low expected fitness).

# Generate a new candidate solution by maximizing the expected improvement.
def propose_candidate(gp, best_fit, n_candidates=50):
    if gp is None:
        return random_individual()  # If the GP model is not available, return a random individual.
    
    best_cand = None
    best_ei = -np.inf  # Initialize best EI value to negative infinity.

    # Generate multiple random candidates and evaluate their expected improvement.
    for _ in range(n_candidates):
        cand = random_individual()
        vec = encode_ind(cand)  # Convert the individual to a vector representation.
        ei = expected_improvement(vec, gp, best_fit)  # Compute EI for the candidate.

        # Update the best candidate if the EI is the highest found so far.
        if ei > best_ei:
            best_ei = ei
            best_cand = cand

    return best_cand  # Return the candidate with the highest expected improvement.

# Main genetic algorithm that evolves the population and integrates Bayesian optimization.
def genetic_algorithm():
    population = initialize_population(POP_SIZE)
    best_ind = None
    best_fit = float('inf')
    fit_history = []
    # Iterate over the generations.
    for generation in range(NUM_GEN):
        fits = []
        details = []
        print(f"\n--- Generation {generation} ---")
        # Evaluate each individual in the population.
        for ind in population:
            f, eR, eP, sl, fs = simulate_individual(ind)
            fits.append(f)
            details.append((ind, f, eR, eP, sl, fs))
            if f < best_fit:
                best_fit = f
                best_ind = ind
        fit_history.append(best_fit)
        print("Best fitness this generation:", best_fit)
        
        # If enough evaluations exist, perform a Bayesian step to propose a candidate.
        if len(global_evaluations) >= 5:
            try:
                gp = fit_surrogate(global_evaluations)
                bcand = propose_candidate(gp, best_fit, n_candidates=50)
                print("Bayesian candidate proposed:", bcand)
                bf, _, _, _, _ = simulate_individual(bcand)
                worst = max(fits)
                # If the candidate performs better than the worst individual, inject it into the population.
                if bf < worst:
                    idx = fits.index(worst)
                    details[idx] = (bcand, bf, None, None, None, None)
                    print("Bayesian candidate injected.")
            except Exception as e:
                print("Bayesian step skipped:", e)
        
        # Check convergence based on the recent fitness history.
        if generation > 10 and np.std(fit_history[-10:]) < 1e-3:
            print("Convergence reached.")
            break
        
        # Create the next generation using elitism, selection, crossover, and mutation.
        nElite = max(1, int(ELITE_RATE * POP_SIZE))
        sorted_pop = sorted(details, key=lambda x: x[1])
        elites = [x[0] for x in sorted_pop[:nElite]]
        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            p1 = selection(population, fits)
            p2 = selection(population, fits)
            c1, c2 = crossover(p1, p2)
            if random.random() < MUT_RATE:
                c1 = mutate(c1)
            if random.random() < MUT_RATE:
                c2 = mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop[:POP_SIZE]
    return best_ind, best_fit

# ---------------------------
# Validate the Best Solution
# ---------------------------
def re_test_config(config, runs=5, sim_time=15000, warm_up=100):
    # Runs multiple simulations to compute the average performance metrics.
    # It initializes a new environment and server for each run, collects response time,
    # power consumption, and job processing counts, then averages the results.
    fitness_vals = []
    E_Rs = []
    E_Ps = []
    slow_counts = []
    fast_counts = []

    for i in range(runs):
        env = simpy.Environment()
        server = DVSServer(env, **config)

        # Run the simulation and capture relevant metrics
        (
            eR,
            eP,
            confidence_interval_R,
            confidence_interval_P,
            state_percentages,
            off_counter,
            sl,
            fs,
            logs
        ) = server.run_simulation(sim_time, warm_up)

        fit = eR * eP
        fitness_vals.append(fit)
        E_Rs.append(eR)
        E_Ps.append(eP)
        slow_counts.append(sl)
        fast_counts.append(fs)

    # Compute and return the average values across all runs
    avg_fit = np.mean(fitness_vals)
    std_fit = np.std(fitness_vals)
    avg_er = np.mean(E_Rs)
    avg_ep = np.mean(E_Ps)
    avg_slow = np.mean(slow_counts)
    avg_fast = np.mean(fast_counts)

    return avg_fit, std_fit, avg_er, avg_ep, avg_slow, avg_fast

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # 1) Run the genetic algorithm to find the best configuration.
    best_ind, best_fit = genetic_algorithm()

    # 2) Print final solution from a single simulation run.
    print("\n=== GA Completed ===")
    single_run_fit, eR, eP, sl, fs = simulate_individual(best_ind)
    print("Best Individual (single-run) => Fit:", single_run_fit, "E(R):", eR, "E(P):", eP)
    print("Slow Jobs:", sl, "Fast Jobs:", fs)
    
    # 3) Re-test the best configuration over multiple runs to get average performance.
    avg_fit, std_fit, avg_er, avg_ep, avg_slow, avg_fast = re_test_config(best_ind, runs=5)
    print("\n=== Re-testing Best Config over 5 runs ===")
    print("Avg Fitness:", avg_fit, "±", std_fit)
    print("Avg E(R):", avg_er, "Avg E(P):", avg_ep)
    print("Avg Slow:", avg_slow, "Avg Fast:", avg_fast)
    
    # 4) Interpret the results: Check if the best solution uses both processing speeds.
    if (avg_slow < 1e-9) or (avg_fast < 1e-9):
        print("\nConclusion: The best solution found uses effectively ONE speed.")
    else:
        print("\nConclusion: The best solution found uses BOTH speeds.")
