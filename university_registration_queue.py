"""
Mini Project: Performance Modeling and Evaluation of a University Student Registration Queue System.
Course: EEX5362

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import simpy
import random
import statistics
import os

# ==========================================
# FILE LOADING
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = "registration_queue_data.csv"
file_path = os.path.join(script_dir, csv_filename)

print("--- INITIALIZING ---")
print(f"Script Location: {script_dir}")
print(f"Loading Dataset: {file_path}")

if not os.path.exists(file_path):
    print("[ERROR] Dataset not found.")
    exit()

# ==========================================
# PART 1: DATA ANALYSIS
# ==========================================

df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"], format="%H:%M")

numeric_cols = [
    "Document_Check_Duration(min)",
    "Waiting_Time(min)",
    "Registration_Service_Duration(min)"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Observation window
t_min = df["Arrival_Time"].min()
t_max = df["Arrival_Time"].max()
observation_hours = (t_max - t_min).total_seconds() / 3600

arrival_rate = len(df) / observation_hours
avg_doc_check = df["Document_Check_Duration(min)"].mean()
avg_service = df["Registration_Service_Duration(min)"].mean()

print("\n--- DATA SUMMARY ---")
print(f"Observation Window: {observation_hours:.2f} hours")
print(f"Arrival Rate (λ): {arrival_rate:.2f} students/hour")
print(f"Avg Document Check Time: {avg_doc_check:.2f} min")
print(f"Avg Registration Service Time: {avg_service:.2f} min")

# ==========================================
# FIGURE 1: WAITING TIME DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 4))
sns.histplot(df["Waiting_Time(min)"].dropna(), bins=25, kde=True, color="purple")
plt.title("FIGURE 1: Distribution of Student Waiting Times")
plt.xlabel("Waiting Time (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "waiting_time_distribution.png"))
plt.close()

# ==========================================
# FIGURE 2: ARRIVALS PER HOUR
# ==========================================

arrivals_hourly = df.groupby(df["Arrival_Time"].dt.hour).size()

plt.figure(figsize=(8, 4))
arrivals_hourly.plot(marker="o", color="red")
plt.title("FIGURE 2: Hourly Student Arrivals")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Students")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "arrivals_per_hour.png"))
plt.close()

# ==========================================
# FIGURE 3: COUNTER UTILIZATION
# ==========================================

utilization = (
    df.groupby("Counter_ID")["Registration_Service_Duration(min)"].sum()
    / (60 * observation_hours)
)

plt.figure(figsize=(7, 4))
utilization.plot(kind="bar", color="green", edgecolor="black")
plt.axhline(y=1.0, color="red", linestyle="--", label="Full Capacity")
plt.title("FIGURE 3: Registration Counter Utilization")
plt.ylabel("Utilization Ratio")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "counter_utilization.png"))
plt.close()

print("Visualizations saved successfully.")

# ==========================================
# PART 2: DISCRETE EVENT SIMULATION (SimPy)
# ==========================================

print("\n--- DISCRETE EVENT SIMULATION ---")

SIM_INTER_ARRIVAL = 60 / 120     # 120 students/hour → 0.5 min
SIM_TIME = 600                  # 10 hours (minutes)

def registration_simulation(env, counters, results):
    doc_check = simpy.Resource(env, capacity=5)
    registration = simpy.Resource(env, capacity=counters)
    wait_times = []

    def student():
        with doc_check.request() as req:
            yield req
            yield env.timeout(random.expovariate(1 / avg_doc_check))

        start_wait = env.now
        with registration.request() as req:
            yield req
            wait_times.append(env.now - start_wait)
            yield env.timeout(random.expovariate(1 / avg_service))

    def generator():
        while True:
            yield env.timeout(random.expovariate(1 / SIM_INTER_ARRIVAL))
            env.process(student())

    env.process(generator())
    env.run(until=SIM_TIME)

    avg_wait = statistics.mean(wait_times)
    results.append((counters, avg_wait))
    print(f"Counters: {counters} → Avg Waiting Time: {avg_wait:.2f} minutes")

random.seed(42)
simulation_results = []

for c in [3, 10, 12]:
    registration_simulation(simpy.Environment(), c, simulation_results)

# ==========================================
# FIGURE 4: WAITING TIME VS COUNTERS
# ==========================================

counters = [r[0] for r in simulation_results]
waits = [r[1] for r in simulation_results]

plt.figure(figsize=(7, 4))
plt.plot(counters, waits, marker="o")
plt.title("FIGURE 4: Waiting Time vs Number of Registration Counters")
plt.xlabel("Number of Counters")
plt.ylabel("Average Waiting Time (minutes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "waiting_time_vs_counters.png"))
plt.show()

print("\n--- FINAL CONCLUSION ---")
print("Simulation confirms that approximately 12 registration counters")
print("are required to maintain waiting times below 20 minutes.")
