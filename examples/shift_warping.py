# %% load libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

os.chdir('C:\dev\workspaces\Ephys-scripts\ephys_analysis')
from Functions.user_def_analysis import *
from Functions.load import *
from Functions.PSTH import *
from Functions.neural_response import *

from scipy.ndimage import gaussian_filter1d
from scipy.signal import firwin, lfilter, filtfilt, hilbert, kaiserord, freqz
from affinewarp import ShiftWarping


os.chdir(r'C:\dev\workspaces\affinewarp\examples\olfaction')
# %% Load example Data

# Raw data (data for pinene odor, 1e-2 M, from one animal).
Z = dict(np.load(r"C:\dev\workspaces\affinewarp\examples\olfaction\pinene_data.npz"))

# Spiking data.
from affinewarp import SpikeData
data = SpikeData(
    Z["trials"],
    Z["spiketimes"],
    Z["neuron_ids"],
    tmin=Z["tmin"],
    tmax=Z["tmax"],
)

# Encode sniff onsets as a fraction between zero (stim onset) and one (trial end).
frac_onsets = Z["sniff_onsets"] / Z["tmax"]

    
# %% debug example data

# Load Data
Z = dict(np.load(r"C:\dev\workspaces\affinewarp\examples\olfaction\pinene_data.npz"))

# Print out data keys and basic shapes, handling scalars and arrays separately
print("Keys in data:", Z.keys())
for key, value in Z.items():
    if isinstance(value, np.ndarray):  # Check if it's an array
        print(f"{key}: shape {value.shape}")
        if value.ndim > 0:  # Only index if it has dimensions
            print(f"First few elements of {key}:\n", value[:5], "\n")
        else:  # For 0-dimensional arrays, print the value directly
            print(f"Value of {key}: {value.item()}\n")
    else:  # Handle non-array types (just in case any are not numpy arrays)
        print(f"{key} (non-array): {value}\n")

# Load and examine unique trials
unique_trials = np.unique(Z["trials"])
num_unique_trials = unique_trials.size

print(f"Number of unique trials: {num_unique_trials}")
print(f"Unique trial IDs: {unique_trials[:]}")

unique_neurons = np.unique(Z["neuron_ids"])
num_unique_neurons = unique_neurons.size

print(f"Number of unique neurons: {num_unique_neurons}")
print(f"Unique neurons IDs: {unique_neurons[:]}")

# %% raster, psths for example data


# Parameters for PSTH
bin_size = 0.01  # Bin size in seconds
tmin, tmax = Z["tmin"], Z["tmax"]
bins = np.arange(tmin, tmax, bin_size)

# Extract data
trials = Z["trials"]
spiketimes = Z["spiketimes"]
neuron_ids = Z["neuron_ids"]
unique_neurons = np.unique(neuron_ids)

# Select a subset of neurons to plot for better readability
selected_neurons = unique_neurons[:6]  # Adjust the range to show a few neurons

# Plot raster plots
plt.figure(figsize=(10, 2 * len(selected_neurons)))
for i, neuron_id in enumerate(selected_neurons):
    neuron_spikes = spiketimes[neuron_ids == neuron_id]
    neuron_trials = trials[neuron_ids == neuron_id]
    
    ax_raster = plt.subplot(len(selected_neurons), 1, i + 1)
    ax_raster.scatter(neuron_spikes, neuron_trials, s=1, color='black')
    ax_raster.set_title(f"Neuron {neuron_id} Raster Plot")
    ax_raster.set_xlim(tmin, tmax)
    ax_raster.set_ylabel("Trial")
    if i == len(selected_neurons) - 1:
        ax_raster.set_xlabel("Time (s)")
    else:
        ax_raster.set_xticks([])

plt.tight_layout()
plt.show()

# Plot PSTHs
plt.figure(figsize=(10, 2 * len(selected_neurons)))
for i, neuron_id in enumerate(selected_neurons):
    neuron_spikes = spiketimes[neuron_ids == neuron_id]
    neuron_trials = trials[neuron_ids == neuron_id]
    
    # Calculate PSTH
    counts, _ = np.histogram(neuron_spikes, bins=bins)
    psth = counts / (len(np.unique(neuron_trials)) * bin_size)  # Firing rate in Hz
    
    ax_psth = plt.subplot(len(selected_neurons), 1, i + 1)
    ax_psth.plot(bins[:-1], psth, color='blue')
    ax_psth.set_title(f"Neuron {neuron_id} PSTH")
    ax_psth.set_xlim(tmin, tmax)
    ax_psth.set_ylabel("Firing Rate (Hz)")
    if i == len(selected_neurons) - 1:
        ax_psth.set_xlabel("Time (s)")
    else:
        ax_psth.set_xticks([])

plt.tight_layout()
plt.show()

# %% Further inspection: individual data keys
# Visualize distributions of the main elements in Z

# Plot spiketimes, trials, and neuron_ids distributions
plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
plt.hist(Z["spiketimes"], bins=50)
plt.title("Spiketimes Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.subplot(2, 2, 2)
plt.hist(Z["trials"], bins=50)
plt.title("Trials Distribution")
plt.xlabel("Trial ID")
plt.ylabel("Frequency")

plt.subplot(2, 2, 3)
plt.hist(Z["neuron_ids"], bins=len(np.unique(Z["neuron_ids"])))
plt.title("Neuron IDs Distribution")
plt.xlabel("Neuron ID")
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.plot(Z["sniff_onsets"], np.arange(len(Z["sniff_onsets"])), 'o', markersize=2)
plt.title("Sniff Onsets Over Trials")
plt.xlabel("Sniff Onset Time")
plt.ylabel("Trial Index")

plt.tight_layout()
plt.show()

# Check sniff onset timings normalized as a fraction of trial duration
frac_onsets = Z["sniff_onsets"] / Z["tmax"]
plt.figure(figsize=(10, 5))
plt.hist(frac_onsets, bins=50, color='blue', alpha=0.7)
plt.title("Normalized Sniff Onsets Distribution")
plt.xlabel("Fraction of Trial Duration (0 to 1)")
plt.ylabel("Frequency")
plt.show()

# Initialize SpikeData object
data = SpikeData(
    Z["trials"],
    Z["spiketimes"],
    Z["neuron_ids"],
    tmin=Z["tmin"],
    tmax=Z["tmax"],
)

# Inspect SpikeData structure by printing key attributes
print("SpikeData structure:")
print(f"Number of trials: {data.n_trials}")
print(f"Number of neurons: {len(np.unique(data.neurons))}")
print(f"Spiketimes (first 10): {data.spiketimes[:10]}")
print(f"Trials (first 10): {data.trials[:10]}")
print(f"Neuron IDs (first 10): {data.neurons[:10]}")

# Visualize a sample raster plot for raw spiking data
plt.figure(figsize=(10, 6))
example_neurons = [2, 6, 20, 22, 28, 9]  # Customize based on observed neurons in data
for i, neuron in enumerate(example_neurons):
    neuron_spikes = data.spiketimes[data.neurons == neuron]
    neuron_trials = data.trials[data.neurons == neuron]
    plt.scatter(neuron_spikes, neuron_trials + i * 0.5, s=2, label=f"Neuron {neuron}")

plt.xlabel("Time")
plt.ylabel("Trial Index (offset by neuron)")
plt.title("Example Raster Plot of Spike Data")
plt.legend(loc="upper right", markerscale=3)
plt.show()

# %% Hyperparameters (can be fiddled with)    
NBINS = 130         # Number of time bins per trial
SMOOTH_REG = 10.0   # Strength of roughness penalty
WARP_REG = 0.0      # Strength of penalty on warp magnitude
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
MAXLAG = 0.5        # Maximum amount of shift allowed.


# Specify model.
from affinewarp import ShiftWarping
shift_model = ShiftWarping(
    maxlag=MAXLAG,
    smoothness_reg_scale=SMOOTH_REG,
    warp_reg_scale=WARP_REG,
    l2_reg_scale=L2_REG,
)

# Fit and apply warping to held out neurons.
from affinewarp.crossval import heldout_transform
validated_alignments = heldout_transform(
    shift_model, data.bin_spikes(NBINS), data, iterations=100)

# Fit model to full dataset (used to align sniffs).
shift_model.fit(data.bin_spikes(NBINS))

# NOTE: various preprocessing and normalizations schemes (z-scoring,
# square-root-transforming the spike counts, etc.) could be tried here.

# Manually specify an alignment to sniff onsets.
from affinewarp import PiecewiseWarping
align_sniff = PiecewiseWarping()
align_sniff.manual_fit(
    data.bin_spikes(NBINS),
    np.column_stack([frac_onsets, np.full(data.n_trials, 0.4)]),
    recenter=False
)

# %% plot

def _plot_column(axc, spks, sniffs):
    """
    Plots column of subplots.
    
    Parameters
    ----------
    axc : array, holding list of axes in column.
    spks : SpikeData object
    sniffs : array, holding sniff time on each trial.
    """
    
    # These are the neurons shown in the paper.
    example_neurons = [2, 6, 20, 22, 28, 9]

    # Plot raster plot for each neuron.
    raster_kws = dict(s=4, c='k', lw=0)
    for n, ax in zip(example_neurons, axc[:-1]):
        ax.scatter(
            spks.spiketimes[spks.neurons == n],
            spks.trials[spks.neurons == n],
            **raster_kws,
        )
        ax.set_ylim(-1, len(sniffs))
        ax.axis('off')

        # Plot blue dots, denoting sniffs, on rasters.
        sniff_kws = dict(c='b', s=5, alpha=.55, lw=0)
        ax.scatter(sniffs, range(sniffs.size), **sniff_kws)

    # Plot histogram at bottom.
    histbins = np.linspace(0, 500, 50)
    if len(np.unique(np.histogram(sniffs, histbins)[0])) == 2:
        axc[-1].axvline(sniffs.mean(), c='b', alpha=.7, lw=2, dashes=[2,2])
    else:
        axc[-1].hist(sniffs, histbins, color='blue', alpha=.65)
    
    # Format bottom subplot.
    axc[-1].spines['right'].set_visible(False)
    axc[-1].spines['top'].set_visible(False)
    axc[-1].set_ylim(0, 15)


# Create figure.
fig, axes = plt.subplots(7, 4, figsize=(9.5, 6))


# First column, raw data.
_plot_column(
    axes[:, 0], data, Z["sniff_onsets"]
)

# Second column, re-sorted trials by warping function.
_plot_column(
    axes[:, 1],
    data.reorder_trials(shift_model.argsort_warps()),
    Z["sniff_onsets"][shift_model.argsort_warps()] 
)

# Third column, shifted alignment.
_plot_column(
    axes[:, 2],
    validated_alignments,
    shift_model.event_transform(
        range(Z["sniff_onsets"].size), frac_onsets) * Z["tmax"],
)

# Final column, aligned to sniff onset.
_plot_column(
    axes[:, 3],
    align_sniff.transform(data),
    align_sniff.event_transform(
        range(Z["sniff_onsets"].size), frac_onsets) * Z["tmax"],
)

# Final formatting.
for ax in axes.ravel():
    ax.set_xlim(-50, 550)
for ax in axes[-1]:
    ax.set_xlabel("time (ms)")

axes[0, 0].set_title("raw data")
axes[0, 1].set_title("sorted by warp")
axes[0, 2].set_title("aligned by model")
axes[0, 3].set_title("aligned to sniff")

fig.tight_layout()
fig.subplots_adjust(hspace=.3)

# %% load data

#root path (local) where the data to analyse is located
analysisDir =  define_directory_analysis()

#Generate a csv with all the data that is going to be anlysed in THIS script
#Should have a copy local of all the preprocessed data
csvDir = os.path.join(analysisDir, 'Inventory', 'gratings_temporal.csv')


database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Protocol": str,
        "Experiment": str,
        "Sorting": str,
    }
)

label = 'mua'

# Create database for analysis
#Inicialize object to colect data from each session
stim = []
neural = []

for i in range(len(database)):
    
    
    #Path
    dataEntry = database.loc[i]
    dataDirectory = os.path.join(analysisDir, dataEntry.Name, dataEntry.Date)
    
    #Check if directory for figures exists
    figureDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date)
    
    if not os.path.isdir(figureDirectory):
        os.makedirs(figureDirectory)
    
    
    #Load stimulus type (for example oddballs)
    s = load_stimuli_data(dataDirectory, dataEntry)
    
    #Load from the csv file WHICH experiments are concatenated for the analysis in INDEX
    #For example, all the oddballs drifting
    
    #TODO: IMPLEMENTED TEST. capture here the cases when there is an only experiment (no indices given)
        
    chopped = chop_by_experiment(s, dataEntry)

    if isinstance(dataEntry.Experiment, str):
        index = [int(e) for e in dataEntry.Experiment.split(',')]
        s = merge_experiments_by_index(chopped, index)
    else:
        s = chopped[0]
        
    stim.append(s)
    
    # Load neural data#########################################################
    
    #Set interval to load just spikes in these limits. Handle merged experiments
    #in an inefficent way. 
    extend = 20
    
    interval = np.array([s['intervals'][0][0] - extend, s['intervals'][-1][0] + extend])
    
    n = load_neural_data_local(dataDirectory, interval)
    
    cluster_info = n['cluster_info']
    cluster_info.set_index('cluster_id', inplace=True)
    SC_depth_limits = n['SC_depth_limits']
    
    #Find somatic units
    if dataEntry.Sorting == 'manual':
        
        label = label.lower()     
        
        # Keep clusters IDs within SC limits, with the label selected before (default good)
        soma = cluster_info[((cluster_info.group == label) &
                                              (cluster_info.depth >= SC_depth_limits[1]) &
                                              (cluster_info.depth <= SC_depth_limits[0]))].index.values
    
    if dataEntry.Sorting == 'bombcell':
        label = label.upper()     
    
        soma = cluster_info[((cluster_info.bc_unitType == label) &
                                              (cluster_info.depth >= SC_depth_limits[1]) &
                                              (cluster_info.depth <= SC_depth_limits[0]))].index.values
    
        #Find non somatic units
        nonSoma = find_units_bomcell(dataDirectory, 'NON-SOMA')
        nonSoma = cluster_info.loc[nonSoma][((cluster_info.depth >= SC_depth_limits[1]) &
                                      (cluster_info.depth <= SC_depth_limits[0]))].index.values
    
    n['cluster_analysis'] = soma
    #n['cluster_analysis'] = np.vstack((soma,nonSoma))
    
    neural.append(n)

# Params

#PSTH params
baseline = 0.2
stimDur = 2
prePostStim = 1
window = np.array([-prePostStim, stimDur + prePostStim])
binSize=0.005
sigma = 0.06
alpha_val = 0.4
groups = np.unique(stim[0]['direction'])
colors = cm.rainbow(np.linspace(0, 1, len(groups)))
xticks = np.arange(-1, 4, 1)

#define windows for determine if neuron is visually responsive
no_stim_window = (-1,0)
stim_window = (0,1)
#stim_window_sensitization = (1.5,2)

#Behaviour params
threshold = 0.9 #for running velocity

#adaptation, fr and supression  params
window_response = np.array([0, 2])
early_interval = np.array([0, 0.5])
late_interval = np.array([1.5, 2])
window_is_suppressed = np.array([0, 1])

#Heatmap params
vmin, vmax = -15,15
norm_interval = np.array([0, 0.5])

# Define behavior

data = []

for st in stim:

    running_state = np.array([])    

    active = 0
    quiet = 0
    not_considered = 0
    
    stimuli_start = st['startTime'].reshape(-1)
    stimuli_end = st['endTime'].reshape(-1)
    ard_time_adjust = st['wheelTimestamps'].reshape(-1)
    velocity = st['wheelVelocity'].reshape(-1)
    
    for start, end in zip(stimuli_start, stimuli_end):
        
        interval_velocity = velocity[np.where((ard_time_adjust >= start) &
                                              (ard_time_adjust <= end))[0]]
        if sum(interval_velocity < 1) > int(len(interval_velocity) * threshold):
            state = 0
            quiet += 1
        elif sum(interval_velocity >= 1) > int(len(interval_velocity) * threshold):
            state = 1
            active += 1 
        else:
            state = np.nan
            not_considered += 1 
        
        running_state = np.hstack((running_state,state))    
        
    st['running_state'] = running_state

# Compute visual responsiveness

def vr_statistic(r_before, r_after, axis):
    return np.mean(r_after, axis=axis) - np.mean(r_before, axis=axis)


for n,st in zip(neural,stim):    

    clusters = n['cluster_analysis']
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    stimuli_start = st['startTime'].reshape(-1)
    direction = st['direction'].reshape(-1)
    
    visual = []
    
    for neuron in clusters:
        
        #Calculate vr using info from one state (because of differences in baseline)
        #Use state with higher response
        
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                            stimuli_start,window)
        #Calculate vr in early response
        r_before, r_after = visual_responsiveness(spAligned,
                                          trials,
                                          no_stim_window,
                                          stim_window,
                                          direction,
                                          baseline)

        res = permutation_test((r_before, r_after), vr_statistic, permutation_type = 'samples',
                                vectorized=True, n_resamples=5000, alternative='two-sided')
                
        if res.pvalue < 0.05:
        
            #Calculate vr in late response (for sensitizing neurons)
            # r_before, r_after = visual_responsiveness(spAligned,
            #                                   trials,
            #                                   no_stim_window,
            #                                   stim_window_sensitization,
            #                                   direction,
            #                                   baseline)

            # res = permutation_test((r_before, r_after), vr_statistic, permutation_type = 'samples',
            #                         vectorized=True, n_resamples=5000, alternative='two-sided')
            # if res.pvalue <= 0.05:
                
            #     visual.append(False)
            # else:
            #     #Test if the firing rate is low
            fr = np.mean(calculate_fr_per_trial(spAligned, trials, window_response, 
                                        direction, baseline))
            if abs(fr) < 0.15:
                visual.append(False)
            else:
                visual.append(True)
            
        else:
            visual.append(False)
   
    #line for selecting and checking the non-visual            
    #visual = [not val for val in visual]     
    n['visual'] = visual     
    
#%% batch raster and psth (our data)

binSize = 0.0005
sigma = 0.07
i = 0

for n,st in zip(neural, stim):    

    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
    
    mask = (st['temporalF'] == 2).reshape(-1)
    stimuli_start = st['startTime'][mask].reshape(-1)
    stimuli_end = st['endTime'][mask].reshape(-1)
    direction = st['direction'][mask].reshape(-1)
    running_state = st['running_state'][mask].reshape(-1)
   
    for neuron in clusters:
        
        fig = plt.figure(figsize=(9, 9))

        # Active
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                            stimuli_start[running_state == 1], 
                                            window)
        
        ax1 = plt.subplot(2,2,1)
        plt.title('ACTIVE', loc='left', fontsize = 10)
        newPlotPSTH(spAligned, trials, window, direction[running_state == 1], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax2 = plt.subplot(2,2,3, sharex = ax1)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 1],
                                              groups, baseline,  binSize, sigma)
        # traces, sem, timeBins = newTracesFromPSTH_trialSmoooth(spAligned, trials, window, direction[running_state == 1],
        #                                       groups, baseline,  binSize, sigma)
        mean_trace_active = np.mean(traces, axis=0)
        
        ax2.plot(timeBins, mean_trace_active, color='black', label='Mean Active', linewidth=2)
        ax2.legend()

        
        

        for t, s, c, l in zip(traces, sem, colors, groups):

            plt.plot(timeBins, t, alpha = alpha_val, c = c, label = str(l) )
            plt.fill_between(timeBins, t - s, t + s, alpha= 0.1, color= c)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

            
        # Quiet
        spAligned, trials = alignData(spike_times[spike_clusters == neuron],
                                            stimuli_start[running_state == 0], 
                                            window)
    
        ax3 = plt.subplot(2,2,2, sharex = ax1)
        plt.title('QUIET', loc='left', fontsize = 10)
        newPlotPSTH(spAligned, trials, window, direction[running_state == 0], groups, colors)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        ax4 = plt.subplot(2,2,4, sharex = ax3, sharey = ax2)
        traces, sem, timeBins = newTracesFromPSTH(spAligned, trials, window, direction[running_state == 0],
                                              groups, baseline, binSize, sigma)
        mean_trace_quiet = np.mean(traces, axis=0)
        
        ax4.plot(timeBins, mean_trace_quiet, color='black', label='Mean Quiet', linewidth=2)
        ax4.legend()


        for t, s, c, l in zip(traces, sem, colors, groups):

            plt.plot(timeBins, t, alpha = alpha_val, c = c, label = str(l) )
            plt.fill_between(timeBins, t - s, t + s, alpha= 0.1, color= c)
        plt.axvline(x=0, c='k', ls='--')
        plt.axvline(x= np.mean(stimuli_end - stimuli_start), c='k', ls='--')
        plt.xticks(xticks)

        
        depth = abs(n['cluster_info']['depth'].loc[neuron] - n['SC_depth_limits'][0]
                    )/(n['SC_depth_limits'][0] - n['SC_depth_limits'][1])*100
        
        fig.suptitle(
            f'Neuron: {neuron} Depth from sSC:{depth:.1f}')
        
        dataEntry = database.loc[i]
               
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, '27-01-25', 'rasterPSTH')
        # saveDirectory = os.path.join(analysisDir, 'Figures', 'FG005', 'raster_PSTH_trial_smooth')
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
    i += 1
    
# %% functions 
def _plot_column(axc, spks, example_neurons):
    """
    Plots column of subplots.
    
    Parameters
    ----------
    axc : array, holding list of axes in column.
    spks : SpikeData object
    sniffs : array, holding sniff time on each trial.
    """
    
    

    # Plot raster plot for each neuron.
    raster_kws = dict(s=4, c='k', lw=0)
    for n, ax in zip(example_neurons, axc):
        ax.scatter(
            spks.spiketimes[spks.neurons == n],
            spks.trials[spks.neurons == n],
            **raster_kws,
        )
        # ax.set_ylim(-1, len(trials))
        ax.axis('off')

def save_shifted_spikes(spks, example_neurons):
    
    shifted_spikes = {}

    for n in example_neurons:
        spiketimes = spks.spiketimes[spks.neurons == n]
        trials = spks.trials[spks.neurons == n]
        
        # Store spike times and trials in a sub-dictionary
        shifted_spikes[n] = {
            'spiketimes': spiketimes,
            'trials': trials
        }
        
    return shifted_spikes

# Function to calculate the range of spike times for each neuron and trial
def calculate_shifted_spike_ranges(shifted_spikes):
    trial_ranges = {}

    for neuron_id, data in shifted_spikes.items():
        spiketimes = data['spiketimes']
        trials = data['trials']
        
        # Create a dictionary to store ranges per trial
        neuron_trial_ranges = {}
        
        for trial in set(trials):
            trial_spiketimes = spiketimes[trials == trial]
            min_time = trial_spiketimes.min()
            max_time = trial_spiketimes.max()
            
            neuron_trial_ranges[trial] = (min_time, max_time)
        
        # Store the ranges for this neuron
        trial_ranges[neuron_id] = neuron_trial_ranges
    
    return trial_ranges

def compute_psth_with_baseline_correction(spiketimes, trials, window, bin_size=0.05, sigma=0.02, baseline=0):
    """
    Compute PSTH with fixed window, Gaussian smoothing, baseline correction, and bin centering.

    Parameters:
    - spiketimes: np.array
        Array of spike times aligned to an event.
    - trials: np.array
        Array of trial IDs corresponding to the spike times.
    - window: tuple
        Time window for PSTH calculation (start, end).
    - bin_size: float
        Size of bins for the PSTH (in seconds).
    - sigma: float
        Standard deviation of the Gaussian smoothing kernel (in seconds).
    - baseline: float
        Duration for baseline calculation (seconds before zero). If 0, no baseline correction.

    Returns:
    - bins: np.array
        Bin centers.
    - smoothed_spike_counts: np.array
        Smoothed PSTH values (spikes/s).
    """
    # Generate fixed bin edges and centers
    if window[0]*window[-1] < 0:
    # Calculate edges and bins centers        
        edges = np.concatenate((-np.arange(0, -window[0], bin_size)[::-1], 
                                np.arange(bin_size, window[1], bin_size)))
    else:
        edges = np.arange(window[0], window[-1], bin_size)
    bins = edges[:-1] + 0.5 * bin_size

    # Initialize an array to accumulate histograms for all trials
    binned_spike_counts = []

    for trial in np.unique(trials):  # Iterate through trials
        spiketimes_in_trial = spiketimes[trials == trial]
        trial_spike_counts, _ = np.histogram(spiketimes_in_trial, bins=edges)
        trial_spike_counts = trial_spike_counts / bin_size  # Convert to spikes/second
        binned_spike_counts.append(trial_spike_counts)

    binned_spike_counts = np.array(binned_spike_counts)

    # Compute mean firing rate across trials
    mean_spike_counts = np.mean(binned_spike_counts, axis=0)

    # Baseline correction
    if baseline > 0:
        baseline_bins = (edges >= -baseline) & (edges < 0)
        baseline_mean = np.mean(mean_spike_counts[baseline_bins[:-1]])  # Match indexing to bins
        mean_spike_counts -= baseline_mean

    # Gaussian smoothing
    sigma_bins = sigma / bin_size
    smoothed_spike_counts = gaussian_filter1d(mean_spike_counts, sigma_bins)

    return bins, smoothed_spike_counts

def plot_raster_and_psth(neuron_id, spiketimes, trials, bins, smoothed_spike_counts, window, save_path=None):
    """
    Plot a combined raster plot and PSTH for a neuron and save the figure.

    Parameters:
    - neuron_id: int
        The ID of the neuron being plotted.
    - spiketimes: np.array
        Array of spike times aligned to an event.
    - trials: np.array
        Array of trial IDs corresponding to the spike times.
    - bins: np.array
        Bin centers for the PSTH.
    - smoothed_spike_counts: np.array
        Smoothed firing rates for the PSTH.
    - window: tuple
        Time window for the plots (start, end).
    - save_path: str or None
        Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 8))

    # Raster plot (top subplot)
    plt.subplot(2, 1, 1)
    for trial in np.unique(trials):  # Iterate through trials
        spiketimes_in_trial = spiketimes[trials == trial]
        plt.scatter(
            spiketimes_in_trial,
            [trial] * len(spiketimes_in_trial),  # Keep the trial index as the y-axis
            s=5,
            color='black'
        )
    plt.title(f"Raster Plot for Neuron {neuron_id}")
    plt.ylabel("Trials")
    plt.xlim(window)

    # PSTH (bottom subplot)
    plt.subplot(2, 1, 2)
    plt.plot(bins, smoothed_spike_counts, label="Smoothed PSTH", color="blue")
    plt.title(f"PSTH for Neuron {neuron_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (Spikes/s)")
    plt.xlim(window)
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        # Create the directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the file with neuron_id appended to the filename
        filename = os.path.join(save_path, f"neuron_{neuron_id}.png")
        plt.savefig(filename)
    plt.close()

# %% package data per direction
desired_direction = 0

spAligned_list = []
trials_list = []
neuron_ids_list = []
direction_array = []

i = 0

for n, st in zip(neural, stim):
    clusters = n['cluster_analysis'][n['visual']]
    spike_times = n['spike_times']
    spike_clusters = n['spike_clusters']
     
    mask = (st['temporalF'] == 2).reshape(-1)
    stimuli_start = st['startTime'][mask].reshape(-1)
    stimuli_end = st['endTime'][mask].reshape(-1)
    direction = st['direction'][mask].reshape(-1)
    running_state = st['running_state'][mask].reshape(-1)
    dataEntry = database.loc[i]
    
    # clusters = [344, 349, 355, 358, 360, 365, 366, 370, 371, 373, 374, 375, 376, 377, 379, 
    #             386, 389, 394, 396, 398, 400, 404, 407, 408, 409, 410, 415, 417, 418, 419, 
    #             423, 424, 426, 427, 429, 430, 435]
    
    clusters = [398, 407, 409, 415, 417, 419, 424, 426, 427, 430]
    
    for neuron_index, neuron_id in enumerate(clusters):
        identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}')
        # Specify the direction of interest
        desired_direction = desired_direction  
        # Mask to filter trials for the desired direction
        direction_mask = (direction == desired_direction)

        spAligned_original, trials_original = alignData( # total trials per state
            spike_times[spike_clusters == neuron_id],
            stimuli_start[direction_mask], 
            window)
        
        # Append data to lists
        spAligned_list.append(spAligned_original)
        trials_list.append(trials_original)
        neuron_ids_list.extend([neuron_index] * len(spAligned_original))
        direction_array.append(direction)
        
    
spAligned_flat = np.concatenate(spAligned_list, axis=0)
trials_flat = np.concatenate(trials_list, axis=0)
    
spAligned_array = np.array(spAligned_flat, dtype=float)  
trials_array = np.array(trials_flat, dtype=int)         
neuron_ids_array = np.array(neuron_ids_list, dtype=int)




tmin = 0.4
tmax = 2
    
np.savez('our_data.npz',
         spiketimes=spAligned_array,
         trials=trials_array,
         neuron_ids=neuron_ids_array,
         tmin = tmin,
         tmax = tmax
         )


our_data = dict(np.load('our_data.npz'))


from affinewarp import SpikeData
our_data = SpikeData(
    our_data["trials"],
    our_data["spiketimes"],
    our_data["neuron_ids"],
    tmin=our_data["tmin"],
    tmax=our_data["tmax"],
)


# %% Hyperparameters + fit the model   
NBINS = 160        # Number of time bins per trial (time window btwn tmin and tmax)
SMOOTH_REG = 10   # Strength of roughness penalty
WARP_REG = 0.0      # Strength of penalty on warp magnitude
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
MAXLAG = 0.5       # Maximum amount of shift allowed.


# Specify model.
shift_model = ShiftWarping(
    maxlag=MAXLAG,
    smoothness_reg_scale=SMOOTH_REG,
    warp_reg_scale=WARP_REG,
    l2_reg_scale=L2_REG,
)

# Fit and apply warping to held out neurons.
from affinewarp.crossval import heldout_transform
validated_alignments = heldout_transform(
    shift_model, our_data.bin_spikes(NBINS), our_data, iterations=100)

template = shift_model.template
plt.figure()
plt.plot(template)

# binned_spikes = our_data.bin_spikes(NBINS)
# plt.figure()
# plt.plot(binned_spikes[0,:,0]) 

# Fit model to full dataset (used to align sniffs).
# shift_model.fit(our_data.bin_spikes(NBINS))


# %% plot column our data


# Create figure.
fig, axes = plt.subplots(10, 3, figsize=(9.5, 8))
# clusters = [398, 407, 409, 415, 417, 419, 424, 426, 427, 430]
example_neurons = [0,1,2,3,4,5,6,7,8,9] #[19, 22, 24, 26, 27, 29, 31, 32, 33, 35]


# First column, raw data.
_plot_column(
    axes[:, 0],
    our_data,
    example_neurons
)

# Second column, re-sorted trials by warping function.
_plot_column(
    axes[:, 1],
    our_data.reorder_trials(shift_model.argsort_warps()),
    example_neurons
)

# Third column, shifted alignment.
_plot_column(
    axes[:, 2],
    validated_alignments,
    example_neurons
)

# Final formatting.
for ax in axes.ravel():
    ax.set_xlim(-0.5, 3)
# for ax in axes[-1]:
#     ax.set_xlabel("time (ms)")

axes[0, 0].set_title("raw data")
axes[0, 1].set_title("sorted by warp")
axes[0, 2].set_title("aligned by model")


fig.tight_layout()
fig.subplots_adjust(hspace=.3)


# %% save the shifted alignment spikes
        
example_neurons = [0,1,2,3,4,5,6,7,8,9]        
shifted_spikes = save_shifted_spikes(validated_alignments, example_neurons)
original_spikes = save_shifted_spikes(our_data, example_neurons)


# %% plot raster and psth (before and after shift)

smoothed_psths_original = {}

for neuron_id, data in original_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    
    # Define the analysis window and parameters
    stimDur = 2
    prePostStim = 1
    window = np.array([-prePostStim, stimDur + prePostStim])
    bin_size = 0.05  # Bin size for PSTH
    sigma = 0.06  # Smoothing parameter
    baseline = 0.2  # Baseline duration for correction

    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )

    smoothed_psths_original[neuron_id] = smoothed_psth
    # Save path for the figure
    save_path = rf"Q:\Analysis\Figures\{dataEntry.Name}/{dataEntry.Date}/27-01-25/trial_shifts\before_shift" 
    # save_path = None
    # Plot combined raster and PSTH
    plot_raster_and_psth(
        neuron_id,
        spiketimes,
        trials,
        bins,
        smoothed_psth,
        window,
        save_path=save_path
    )
    
# Initialize a dictionary to store smoothed PSTH for all neurons
smoothed_psths_shifted = {}
    
for neuron_id, data in shifted_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    
    # Define the analysis window and parameters
    stimDur = 2
    prePostStim = 1
    window = np.array([-prePostStim, stimDur + prePostStim])
    bin_size = 0.05  # Bin size for PSTH
    sigma = 0.06  # Smoothing parameter
    baseline = 0.2  # Baseline duration for correction

    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )
    
    # Inside your loop, after computing the smoothed PSTH:
    smoothed_psths_shifted[neuron_id] = smoothed_psth

    # Save path for the figure
    save_path = rf"Q:\Analysis\Figures\{dataEntry.Name}/{dataEntry.Date}/27-01-25/trial_shifts\after_shift"
    # save_path = None

    # Plot combined raster and PSTH
    plot_raster_and_psth(
        neuron_id,
        spiketimes,
        trials,
        bins,
        smoothed_psth,
        window,
        save_path=save_path
    )


# %% Data for filtering
# Define the analysis window and parameters
stimDur = 2
prePostStim = 0.5
window = np.array([-prePostStim, stimDur + prePostStim])
bin_size = 0.0005  # Bin size for PSTH
sigma = 0.06  # Smoothing parameter
baseline = 0.2  # Baseline duration for correction
    
smoothed_psths_original = {}

for neuron_id, data in original_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    

    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )

    smoothed_psths_original[neuron_id] = smoothed_psth
    
# Initialize a dictionary to store smoothed PSTH for all neurons
smoothed_psths_shifted = {}
    
for neuron_id, data in shifted_spikes.items():
    spiketimes = data['spiketimes']
    trials = data['trials']
    
    # Compute PSTH for the neuron
    bins, smoothed_psth = compute_psth_with_baseline_correction(
        spiketimes, trials, window, bin_size=bin_size, sigma=sigma, baseline=baseline
    )
    
    # Inside your loop, after computing the smoothed PSTH:
    smoothed_psths_shifted[neuron_id] = smoothed_psth
    
    

timeBins = bins
neuron_id = 8
psth_trace = smoothed_psths_shifted[neuron_id]
plt.figure()
plt.plot(psth_trace)
# %% Filter design


# Calculate sampling frequency from time bins
time_diffs = np.diff(timeBins)  # Calculate time differences between consecutive points
fs = 1 / np.mean(time_diffs)  # Sampling frequency is the reciprocal of the mean time difference

# Filter specifications
cutoff = [1, 3]  # Desired cutoff frequencies in Hz
bandpass = cutoff[1]-cutoff[0]
ripple_db = 60  # Desired stopband attenuation in dB - 30 -> attentuated to 1/30th of original amplitude
width = 2/fs   # 
width = 0.002
# width = 0.1*bandpass # Transition width - 10% of passband width ('cutoff') - numtaps become too low 
numtaps, beta = kaiserord(ripple_db, width) # essentially the length of the filter 
numtaps = 5500
print(f"Calculated number of taps: {numtaps}")
group_delay = numtaps / (2 * fs) # how much the delay would be in seconds
print(f'group delay: {group_delay}')

# Generate filter coefficients using the Kaiser window
fir_coeff = firwin(numtaps, cutoff, window=('kaiser', beta), pass_zero=False, fs=fs)

# Calculate frequency response of the filter
w, h = freqz(fir_coeff, worN=8000, fs=fs)

# Plot the frequency response
plt.figure()
plt.plot(w, abs(h), label=f'Kaiser window numtaps={numtaps}')
plt.title('Frequency Response with Kaiser Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.xlim(0, 10)
plt.grid(True)
plt.legend(loc='best')
plt.show()


# %% padding the signal - edge value - which is essentially 0

sigma = 0.06
# padding from 0s
expansion_amount = 3 * sigma

start_time = 0 - expansion_amount
end_time = 2 + expansion_amount

start_idx = np.searchsorted(timeBins, start_time)
end_idx = np.searchsorted(timeBins, end_time)
stim_window_psth = psth_trace[start_idx:end_idx]
pad_length =  6000  # Pad length is half the number of taps
padded_data = np.pad(stim_window_psth, pad_length, mode='edge')
timeBins_stim_window = timeBins[start_idx:end_idx]
time_step = np.mean(np.diff(timeBins))
timeBins_padded = np.arange(
    timeBins[start_idx] - pad_length * time_step,
    timeBins[end_idx - 1] + pad_length * time_step + time_step,
    time_step
)[:len(padded_data)]


plt.figure(figsize=(15, 10))
plt.subplot(311)
plt.plot(timeBins, psth_trace)
plt.title('Shifted mean PSTH')

plt.subplot(312)
plt.plot(timeBins_stim_window, stim_window_psth)
plt.title('Shifted mean PSTH window')

plt.subplot(313)
plt.plot(timeBins_padded, padded_data)
plt.title('Shifted mean PSTH window padded')

plt.tight_layout()
plt.show()

# %% applying the filter
# filtered_data = lfilter(fir_coeff, 1.0, psth_trace)
filtered_padded_data = filtfilt(fir_coeff, 1.0, padded_data)
filtered_data = filtered_padded_data[pad_length:-pad_length]

# Plotting the original and filtered signal for comparison

plt.figure(figsize=(16, 14))
ax1 = plt.subplot(411)
plt.plot(timeBins, psth_trace)
plt.title('Shifted mean PSTH', fontsize=14, fontweight='bold')

ax2 = plt.subplot(412, sharex=ax1)
plt.plot(timeBins_padded, padded_data)
plt.title('Shifted mean PSTH window padded', fontsize=14, fontweight='bold')

ax3 = plt.subplot(413, sharex=ax1)
plt.plot(timeBins_padded, filtered_padded_data)
plt.title('Filtered Signal Padded', fontsize=14, fontweight='bold')

ax4 = plt.subplot(414, sharex=ax1)
plt.plot(timeBins_stim_window, filtered_data)
plt.title('Filtered Signal', fontsize=14, fontweight='bold')

plt.show()

# %% find time shifts per trial

binsize_shift_model = (tmax-tmin)/NBINS
shifts = shift_model.shifts
shifts_s = shifts*binsize_shift_model


# checking the difference between original and shifted spikes to confirm
# it seems that they are not always exactly as the binsize suggests - difference of 0.01 here and there and in different neurons
# but consistent per trial per in a given neuron
original_spikes_1 = original_spikes[1]['spiketimes']
shifted_spikes_1 = shifted_spikes[1]['spiketimes']
trials_1 = original_spikes[1]['trials']

diff_spikes_1 = original_spikes_1 - shifted_spikes_1



# %% subtract F1 from original PSTH

pad_value = 639
padded_F1 = np.pad(filtered_data, pad_width=pad_value, mode='constant', constant_values=0)
F1_trial = padded_F1/20

original_psth_nonshift = smoothed_psths_original[neuron_id]

binsize_signal = timeBins[-1]-timeBins[-2]

F1_trial_shifted = np.zeros((len(shifts_s), len(F1_trial)))  # Placeholder for all shifted versions

for i, shift in enumerate(shifts_s):
    shift_bins = int(shift / binsize_signal)  # Convert shift in seconds to bins
    # shift_bins = - shift_bins
    if shift_bins >= 0:
        F1_trial_shifted[i, shift_bins:] = F1_trial[:len(F1_trial) - shift_bins]
    else:
        F1_trial_shifted[i, :shift_bins] = F1_trial[-shift_bins:]

F1_subtracted_signal = original_psth_nonshift.copy()

# Subtract each trial's shifted signal from the psth_trace
for i in range(len(shifts_s)):
    F1_subtracted_signal -= F1_trial_shifted[i]



plt.figure()
plt.plot(timeBins, original_psth_nonshift, label = 'Original')
plt.plot(timeBins, padded_F1, label = 'F1')
plt.plot(timeBins, F1_trial, label = 'F1 trial no shift')
plt.plot(timeBins, F1_trial_shifted[2], label = 'F1 trial shift 0.14 s')
plt.plot(timeBins, F1_subtracted_signal, label = 'F1 subtracted')
plt.legend()


plt.figure()
plt.plot(timeBins, F1_trial, label = 'F1 trial no shift', linewidth=5, color='black')
for idx, trial in enumerate(F1_trial_shifted, start=1):
    plt.plot(timeBins, trial, label=f'Trial {idx}')
plt.legend()

# %% sliced plot

pad_value = 850
# Define slicing indices
slice_start = pad_value
slice_end = -pad_value if pad_value > 0 else None  # Avoid issues if pad_value is 0

# Slice the relevant arrays
F1_trial_sliced = F1_trial[slice_start:slice_end]
F1_trial_shifted_sliced = F1_trial_shifted[:, slice_start:slice_end]
F1_subtracted_signal_sliced = F1_subtracted_signal[slice_start:slice_end]
timeBins_sliced = timeBins[slice_start:slice_end]
filtered_data_sliced = filtered_data[211:-211]
# Plot sliced versions
plt.figure()
plt.plot(timeBins_sliced, original_psth_nonshift[slice_start:slice_end], label='Original', linewidth=2)
plt.plot(timeBins_sliced, filtered_data_sliced, label = 'F1', linewidth=2)
plt.plot(timeBins_sliced, F1_trial_sliced, label='F1 trial no shift', linewidth=2)
plt.plot(timeBins_sliced, F1_trial_shifted_sliced[2], label='F1 trial shift 0.14 s', linewidth=2)
plt.plot(timeBins_sliced, F1_subtracted_signal_sliced, label='F1 subtracted', linewidth=2)
plt.title('F1 subtraction')
plt.legend()

plt.figure()
plt.plot(timeBins_sliced, F1_subtracted_signal_sliced, label='F1 subtracted', linewidth=2)
plt.title('F1 subtracted response')
# %% F1 subtracted directly

F1_subtracted_wo_trial = original_psth_nonshift - padded_F1

F1_subtracted_wo_trial=F1_subtracted_wo_trial[211:-211]
plt.figure()
# plt.plot(timeBins_sliced, original_psth_nonshift[slice_start:slice_end], label = 'Original')
# plt.plot(timeBins, padded_F1, label = 'F1')
plt.plot(timeBins_sliced, F1_subtracted_wo_trial, label = 'F1 subtracted directly')
# plt.plot(timeBins, F1_subtracted_signal, label = 'F1 subtracted each trial')

plt.legend()


# %% batch raster, PSTH for shifted data all trials


def plot_neuron_psth_and_raster(shifted_spikes, direction, trials_total, bin_size=0.05, sigma=0.06, save_path=None):
    """
    Plots a raster plot and PSTH for each neuron in the shifted_spikes dictionary, 
    with trials in the raster plot colored by direction.

    Parameters:
        shifted_spikes (dict): Dictionary containing neuron data, with neuron IDs as keys and
                               sub-dictionaries containing 'spiketimes' and 'trials'.
        direction (np.array): Array of stimulus directions, aligned with trials_total.
        trials_total (np.array): Array of trial indices, aligned with direction.
        bin_size (float): Size of bins for the PSTH in the same units as spiketimes.
        sigma (float): Standard deviation for Gaussian smoothing.
        save_path (str): Path to the folder where plots will be saved. If None, plots will not be saved.
    """
    # Create a colormap to represent different directions
    unique_directions = np.unique(direction)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_directions)))  # Use a colormap with distinct colors
    direction_color_map = dict(zip(unique_directions, colors))  # Map each direction to a color

    for neuron_id, data in shifted_spikes.items():
        spiketimes = data['spiketimes']
        trials = data['trials']
        
        # Determine time range
        time_range = (np.min(spiketimes), np.max(spiketimes))

        # Bin edges for PSTH
        bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)

        # Create the figure
        plt.figure(figsize=(9, 9))

        # Raster plot direction sorted
        # plt.subplot(2, 1, 1)
        # k = 0
        # for d in unique_directions:
        #     # Find trials corresponding to the current direction
        #     direction_mask = (direction == d)
        #     trials_in_direction = trials_total[direction_mask]
            
            
        #     # Plot each trial in this direction with the assigned color
        #     for trial in trials_in_direction:
        #         spiketimes_in_trial = spiketimes[trials == trial]
        #         plt.scatter(spiketimes_in_trial, [k] * len(spiketimes_in_trial), 
        #                     s=5, color=direction_color_map[d], label=f'Direction {d}' if trial == trials_in_direction[0] else "")
                
        #         k+=1
        
        # plt.title(f'Raster Plot for Neuron {neuron_id}')
        # plt.ylabel('Trials')
        # plt.xlim(time_range)
        # plt.legend(title='Stimulus Direction')
        
        # Raster plot (top subplot)
        plt.subplot(2, 1, 1)
        
        for trial in np.unique(trials):  # Iterate through trials in their original order
            spiketimes_in_trial = spiketimes[trials == trial]
            plt.scatter(
                spiketimes_in_trial,
                [trial] * len(spiketimes_in_trial),  # Keep the trial index as the y-axis
                s=5,
                color='black',  # Set the color to black
                label=f'Trial {trial}' if trial == np.min(trials) else ""
            )

        
        plt.title(f'Raster Plot for Neuron {neuron_id}')
        plt.ylabel('Trials')
        plt.xlim(time_range)
        plt.legend(title='Trials (as ordered)')


        # # PSTH (bottom subplot) direction sorted
        # plt.subplot(2, 1, 2)
        # for d in unique_directions:
        #     direction_mask = (direction == d)
        #     trials_in_direction = trials_total[direction_mask]

        #     # Initialize an array to accumulate histograms for each trial
        #     trial_histograms = []

        #     for trial in trials_in_direction:
        #         spiketimes_in_trial = spiketimes[trials == trial]
        #         trial_spike_counts, _ = np.histogram(spiketimes_in_trial, bins=bins)
        #         trial_spike_counts = trial_spike_counts / bin_size
        #         trial_histograms.append(trial_spike_counts)

        #     trial_histograms = np.array(trial_histograms)
        #     mean_spike_counts = np.mean(trial_histograms, axis=0)
        #     smoothed_spike_counts = gaussian_filter1d(mean_spike_counts, sigma=sigma)

        #     # Plot the smoothed PSTH
        #     plt.plot(bins[:-1], smoothed_spike_counts, label=f'Direction {d}', color=direction_color_map[d])

        # plt.title(f'PSTH for Neuron {neuron_id}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Spikes/s')
        # plt.xlim(time_range)
        # plt.legend(title='Stimulus Direction')
        # plt.tight_layout()
        
        # PSTH (bottom subplot)
        plt.subplot(2, 1, 2)
        
        # Initialize an array to accumulate histograms for all trials
        all_trial_histograms = []
        
        for trial in np.unique(trials):  # Iterate through trials in their natural order
            spiketimes_in_trial = spiketimes[trials == trial]
            trial_spike_counts, _ = np.histogram(spiketimes_in_trial, bins=bins)
            trial_spike_counts = trial_spike_counts / bin_size  # Convert to spikes/second
            all_trial_histograms.append(trial_spike_counts)
        
        all_trial_histograms = np.array(all_trial_histograms)
        mean_spike_counts = np.mean(all_trial_histograms, axis=0)
        smoothed_spike_counts = gaussian_filter1d(mean_spike_counts, sigma=sigma)
        
        # Plot the smoothed PSTH
        plt.plot(bins[:-1], smoothed_spike_counts, label="Mean PSTH", color="blue")
        
        plt.title(f'PSTH for Neuron {neuron_id}')
        plt.xlabel('Time (s)')
        plt.ylabel('Spikes/s')
        plt.xlim(time_range)
        plt.legend(title='Aligned Trials')

        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'neuron_{neuron_id}.png')
            plt.savefig(file_path)

        plt.close()


for n,st in zip(neural, stim):    

    mask = (st['temporalF'] == 2).reshape(-1)
    stimuli_start = st['startTime'][mask].reshape(-1)
    stimuli_end = st['endTime'][mask].reshape(-1)
    direction = st['direction'][mask].reshape(-1)
    trials_total = np.arange(len(direction))
    running_state = st['running_state'][mask].reshape(-1)
    

    plot_neuron_psth_and_raster(original_spikes, direction, trials_total, bin_size=0.05, sigma=0.06, save_path = r"Q:\Analysis\Figures\trial_shifts\before_shift")
    plot_neuron_psth_and_raster(shifted_spikes, direction, trials_total, bin_size=0.05, sigma=0.06, save_path = r"Q:\Analysis\Figures\trial_shifts\after_shift")
    
# %% batch raster, PSTH for shifted data per direction


def plot_neuron_psth_and_raster(shifted_spikes, direction, trials_total, bin_size=0.05, sigma=0.06, save_path=None):

    # Create a colormap to represent different directions
    unique_directions = np.unique(direction)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_directions)))  # Use a colormap with distinct colors
    direction_color_map = dict(zip(unique_directions, colors))  # Map each direction to a color

    for neuron_id, data in shifted_spikes.items():
        spiketimes = data['spiketimes']
        trials = data['trials']
        
        # Determine time range
        time_range = (np.min(spiketimes), np.max(spiketimes))

        # Bin edges for PSTH
        bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)

        # Create the figure
        plt.figure(figsize=(9, 9))
        
        # Raster plot (top subplot)
        plt.subplot(2, 1, 1)
        
        for trial in np.unique(trials):  # Iterate through trials in their original order
            spiketimes_in_trial = spiketimes[trials == trial]
            plt.scatter(
                spiketimes_in_trial,
                [trial] * len(spiketimes_in_trial),  # Keep the trial index as the y-axis
                s=5,
                color='black',  # Set the color to black
                label=f'Trial {trial}' if trial == np.min(trials) else ""
            )

        
        plt.title(f'Raster Plot for Neuron {neuron_id}')
        plt.ylabel('Trials')
        plt.xlim(time_range)
        plt.legend(title='Trials (as ordered)')
        
        # PSTH (bottom subplot)
        plt.subplot(2, 1, 2)
        
        # Initialize an array to accumulate histograms for all trials
        all_trial_histograms = []
        
        for trial in np.unique(trials):  # Iterate through trials in their natural order
            spiketimes_in_trial = spiketimes[trials == trial]
            trial_spike_counts, _ = np.histogram(spiketimes_in_trial, bins=bins)
            trial_spike_counts = trial_spike_counts / bin_size  # Convert to spikes/second
            all_trial_histograms.append(trial_spike_counts)
        
        all_trial_histograms = np.array(all_trial_histograms)
        mean_spike_counts = np.mean(all_trial_histograms, axis=0)
        smoothed_spike_counts = gaussian_filter1d(mean_spike_counts, sigma=sigma)
        
        # Plot the smoothed PSTH
        plt.plot(bins[:-1], smoothed_spike_counts, label="Mean PSTH", color="blue")
        
        plt.title(f'PSTH for Neuron {neuron_id}')
        plt.xlabel('Time (s)')
        plt.ylabel('Spikes/s')
        plt.xlim(time_range)
        plt.legend(title='Aligned Trials')

        
        # Save the plot if save_path is provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'neuron_{neuron_id}.png')
            plt.savefig(file_path)

        plt.close()


for n,st in zip(neural, stim):    

    mask = (st['temporalF'] == 2).reshape(-1)
    stimuli_start = st['startTime'][mask].reshape(-1)
    stimuli_end = st['endTime'][mask].reshape(-1)
    direction = st['direction'][mask].reshape(-1)
    trials_total = np.arange(len(direction))
    running_state = st['running_state'][mask].reshape(-1)
    

    plot_neuron_psth_and_raster(original_spikes, direction, trials_total, bin_size=0.05, sigma=0.06, save_path = rf"Q:\Analysis\Figures\{dataEntry.Name}/{dataEntry.Date}/27-01-25/trial_shifts\before_shift_1" )
    plot_neuron_psth_and_raster(shifted_spikes, direction, trials_total, bin_size=0.05, sigma=0.06, save_path = rf"Q:\Analysis\Figures\{dataEntry.Name}/{dataEntry.Date}/27-01-25/trial_shifts\after_shift_1" )