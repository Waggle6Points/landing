import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Base pacing model
# ===============================
class PacingModel:
    def control(self, state, context):
        raise NotImplementedError


# ===============================
# Energy + U-shape + Fatigue model
# ===============================
class EnergyUF(PacingModel):
    def __init__(self, energyfactor, ufactor,
                 a0, fatigue_factor, O2_intake, f):
        self.energyfactor = energyfactor
        self.ufactor = ufactor
        self.fatigue_factor = fatigue_factor
        self.a0 = a0
        self.O2_intake = O2_intake
        self.f = f

    def control(self, state, context):
        dE_dt = self.O2_intake - self.f * state['v']
        E = max(state['E'] + dE_dt, 0)

        dF_dt = 5.0 * state['v'] - 3.5
        F = max(state['F'] + dF_dt, 0)

        progress = state['x'] / context['total_distance']

        a_desired = (
            self.energyfactor * E
            + self.ufactor * abs(np.sin(progress - 0.5))
        )

        a_max = self.a0 / (1.0 + self.fatigue_factor * F)

        return np.clip(a_desired, -a_max, a_max)


# ===============================
# Runner
# ===============================
class Runner:
    def __init__(self, pacing_model):
        self.state = {'x': 0.0, 'v': 0.0, 'E': 100.0, 'F': 0.0}
        self.model = pacing_model
        self.finished_time = None


# ===============================
# Helpers
# ===============================
def O2_from_VO2max(VO2max, kg):
    return VO2max * kg * 5.04 * 4.184 /(60 * 22400)


# ===============================
# Simulation
# ===============================
def simulate_race(runners, race_distance=1600,
                  total_time=100000, dt=1.0, sigma=0.05):

    context = {'total_distance': race_distance}
    history = {i: {'v': [], 'x': []} for i in range(len(runners))}

    for t in range(int(total_time / dt)):
        for i, runner in enumerate(runners):
            state = runner.state

            if runner.finished_time is not None:
                continue

            a_control = runner.model.control(state, context)
            a = a_control + sigma * np.random.randn()

            state['v'] = max(state['v'] + a * dt, 0.0)
            state['x'] += state['v'] * dt
            state['E'] = max(state['E'] - abs(a_control) * dt, 0.0)

            history[i]['v'].append(state['v'])
            history[i]['x'].append(state['x'])

            if state['x'] >= race_distance:
                runner.finished_time = t * dt

    return history


# ===============================
# TERMINAL INPUT
# ===============================
kg = 70
num_runners = int(input("Number of runners: "))

runners = []

for i in range(num_runners):
    print(f"\n--- Runner {i+1} ---")
    VO2max = float(input("VO2max (ml/kg/min): "))
    energyfactor = float(input("Energy Weight: "))
    ufactor = float(input("U Pacing Weight:"))
    fatigue_factor = float(input("Fatigue factor: "))
    a0 = float(input("Max accel a0 (from time trial): "))
    Power_avg = float(input("Power_Avg(from other run):"))
    Velocity_avg = float(input("Velocity_Avg(from other run):"))
                      

    model = EnergyUF(
        energyfactor=energyfactor,
        ufactor=ufactor,
        a0=a0,
        fatigue_factor=fatigue_factor,
        O2_intake=O2_from_VO2max(VO2max, kg),
        f = (Power_avg / Velocity_avg)
    )
    runners.append(Runner(model))


# ===============================
# Run race
# ===============================
history = simulate_race(runners)


# ===============================
# Results
# ===============================
print("\n--- Race Results ---")
for i, runner in enumerate(runners):
    print(f"Runner {i+1}: finish time = {runner.finished_time:.1f} s")


# ===============================
# Plots
# ===============================
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))

for i in history:
    ax1.plot(history[i]['v'], label=f"Runner {i+1}")
ax1.set_xlabel("Time step")
ax1.set_ylabel("Velocity (m/s)")
ax1.legend()

for i in history:
    ax2.plot(history[i]['x'], label=f"Runner {i+1}")
ax2.set_xlabel("Time step")
ax2.set_ylabel("Position (m)")
ax2.legend()

plt.tight_layout()
plt.show()