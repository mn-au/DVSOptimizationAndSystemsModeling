# Overview:
# This code implements a simulation model for a Dynamic Voltage Scaling (DVS) server.
# It uses SimPy for the simulation environment.
# The simulation handles job arrivals, state transitions (OFF, SETUP, BUSY, SCALED, IDLE).
# and tracks metrics such as response times, energy consumption, and state durations.

import simpy 
import numpy as np  
import scipy.stats as stats  
import random  # currently not in use

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

# Define a list of test cases to run the simulation.
test_cases = [
    {
        "description": "GA Possible Winning Test Case", 
        "parameters": {
            "arrival_rate": 0.95,      
            "service_rate": 1.0,    
            "setup_time": 1.0,        
            "alpha": 6.7,              
            "scale_speed_factor": 6,   
            "turn_on_threshold": 39,   
            "scale_threshold": 39,    
            "idle_power": 0.6,         
        },
    }
]

# Execute the simulation for each test case.
for i, test_case in enumerate(test_cases):
    try:
        env = simpy.Environment()  # Create a new simulation environment.

        # Initialize a DVSServer instance with the parameters from the test case.
        server = DVSServer(env, **test_case["parameters"])

        # Run the simulation and collect metrics.
        (E_R, E_P, confidence_interval_R, confidence_interval_P, state_percentages, 
         server_off_counter, jobs_slow_speed_counter, jobs_fast_speed_counter, log) = server.run_simulation(run_time=15000, warm_up_period=100)

        # Output the results for the test case.
        print(f"\nTest case {i+1}: {test_case['description']}")
        print(f"E(R): {E_R}")
        print(f"Confidence interval for Response time: {confidence_interval_R[0]} to {confidence_interval_R[1]}")
        print(f"E(P): {E_P}")
        print(f"Confidence interval for Power: {confidence_interval_P[0]} to {confidence_interval_P[1]}")
        print(f"Percentage of times in each state: {state_percentages}")
        print(f"Number of times server switched off: {server_off_counter}")
        print(f"Number of jobs completed at slow speed: {jobs_slow_speed_counter}")
        print(f"Number of jobs completed at fast speed: {jobs_fast_speed_counter}")
        print(f"Overall Energy Consumed (approx.): {E_P * E_R}")

    except ValueError as ve:
        # Print an error message if an exception is raised during simulation.
        print("An error occurred:", ve)


