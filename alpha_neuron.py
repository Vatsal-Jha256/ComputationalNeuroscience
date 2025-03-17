import numpy as np
import matplotlib.pyplot as plt

def run_simulation(t_peak):
    h = 1.  # step size
    t_max = 200  # ms, simulation time period
    tstop = int(t_max / h)  # number of time steps
    ref = 0  # refractory period counter

    # Generate random input spikes
    thr = 0.9  # threshold for random spikes
    spike_train = np.random.rand(tstop) > thr

    # alpha function synaptic conductance
    t_a = 100  # Max duration of syn conductance
    g_peak = 0.05  # nS (peak synaptic conductance)
    const = g_peak / (t_peak * np.exp(-1))
    t_vec = np.arange(0, t_a + h, h)
    alpha_func = const * t_vec * (np.exp(-t_vec / t_peak))

    # Initialize parameters
    C = 0.5  # nF
    R = 40  # M ohms
    g_ad = 0
    G_inc = 1 / h
    tau_ad = 2
    E_leak = -60  # mV, equilibrium potential
    E_syn = 0  # Excitatory synapse
    g_syn = 0  # Current syn conductance
    V_th = -40  # spike threshold mV
    V_spike = 50  # spike value mV
    ref_max = 4 / h  # Starting value of ref period counter
    t_list = np.array([], dtype=int)
    V = E_leak
    V_trace = [V]

    for t in range(tstop):
        if spike_train[t]:  # check for input spike
            t_list = np.concatenate([t_list, [1]])

        g_syn = np.sum(alpha_func[t_list])
        I_syn = g_syn * (E_syn - V)

        if np.any(t_list):
            t_list = t_list + 1
            if t_list[0] == t_a:  # Reached max duration of syn conductance
                t_list = t_list[1:]

        if not ref:
            V = V + h * (-((V - E_leak) * (1 + R * g_ad) / (R * C)) + (I_syn / C))
            g_ad = g_ad + h * (-g_ad / tau_ad)  # spike rate adaptation
        else:
            ref -= 1
            V = V_th - 10  # reset voltage after spike
            g_ad = 0

        if (V > V_th) and not ref:
            V = V_spike
            ref = ref_max
            g_ad = g_ad + G_inc

        V_trace.append(V)

    return np.sum(np.array(V_trace) == V_spike)  # Count spikes

# Define range for t_peak
t_peaks = np.arange(0.5, 10.5, 0.5)
spike_counts = []

# Run simulation for each t_peak and record spike count
for t_peak in t_peaks:
    spike_count = run_simulation(t_peak)
    spike_counts.append(spike_count)
    print(f"t_peak = {t_peak}, spike_count = {spike_count}")

# Plot the results
plt.figure()
plt.plot(t_peaks, spike_counts, marker='o')
plt.xlabel('t_peak (ms)')
plt.ylabel('Spike Count')
plt.title('Output Spike Count vs. t_peak')
plt.show()
