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

os.chdir(r'C:\dev\workspaces\affinewarp\examples\olfaction')
# %% Load Data

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

    
# %% debug data

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

# %% raster, psths


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

# %% load our data

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
binSize=0.04
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
    
#%% batch raster and psth

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
               
        saveDirectory = os.path.join(analysisDir, 'Figures', dataEntry.Name, dataEntry.Date, 'raster_PSTH_12_11_24')
        # saveDirectory = os.path.join(analysisDir, 'Figures', 'FG005', 'raster_PSTH_trial_smooth')
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
        filename = os.path.join(saveDirectory, f'{neuron}_PSTH.png')
        plt.savefig(filename)
        plt.close()
        
    i += 1
# %% open data 

spAligned_dict = {}
trials_dict = {}
neuron_id_for_spikes = {}

spAligned_list = []
trials_list = []
neuron_ids_list = []

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
    
    clusters = [344, 349, 355, 358, 360, 365, 366, 370, 371, 373, 374, 375, 376, 377, 379, 
                386, 389, 394, 396, 398, 400, 404, 407, 408, 409, 410, 415, 417, 418, 419, 
                423, 424, 426, 427, 429, 430, 435]
    
    for neuron_index, neuron_id in enumerate(clusters):
        identifier=f'{dataEntry.Name}_{dataEntry.Date}_{neuron_id}'
        print(f'Processing neuron {identifier}')
        
        spAligned_original, trials_original = alignData( # total trials per state
            spike_times[spike_clusters == neuron_id],
            stimuli_start, 
            window)
        
        # Store in dictionaries with neuron identifier as keys
        spAligned_dict[f'spAligned_{neuron_id}'] = spAligned_original
        trials_dict[f'trials_{neuron_id}'] = trials_original
        neuron_id_for_spikes[neuron_id] = [neuron_id] * len(spAligned_original)
        
        
        # Append data to lists
        spAligned_list.append(spAligned_original)
        trials_list.append(trials_original)
        neuron_ids_list.extend([neuron_index] * len(spAligned_original))
        
    
spAligned_flat = np.concatenate(spAligned_list, axis=0)
trials_flat = np.concatenate(trials_list, axis=0)
    
spAligned_array = np.array(spAligned_flat, dtype=float)  
trials_array = np.array(trials_flat, dtype=int)         
neuron_ids_array = np.array(neuron_ids_list, dtype=int)

tmin = 0.2
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


# %% Hyperparameters (our data)    
NBINS = 180         # Number of time bins per trial
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
    shift_model, our_data.bin_spikes(NBINS), our_data, iterations=100)

# Fit model to full dataset (used to align sniffs).
# shift_model.fit(our_data.bin_spikes(NBINS))

# NOTE: various preprocessing and normalizations schemes (z-scoring,
# square-root-transforming the spike counts, etc.) could be tried here.

# Manually specify an alignment to sniff onsets.
# from affinewarp import PiecewiseWarping
# align_sniff = PiecewiseWarping()
# align_sniff.manual_fit(
#     data.bin_spikes(NBINS),
#     np.column_stack([frac_onsets, np.full(data.n_trials, 0.4)]),
#     recenter=False
# )    


# %% plot our data

def _plot_column(axc, spks):
    """
    Plots column of subplots.
    
    Parameters
    ----------
    axc : array, holding list of axes in column.
    spks : SpikeData object
    sniffs : array, holding sniff time on each trial.
    """
    
    example_neurons = [36, 35, 34, 33, 32, 31]

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

        # Plot blue dots, denoting sniffs, on rasters.
        # sniff_kws = dict(c='b', s=5, alpha=.55, lw=0)
        # ax.scatter(sniffs, range(sniffs.size), **sniff_kws)

    # Plot histogram at bottom.
    # histbins = np.linspace(0, 2, 1)
    # if len(np.unique(np.histogram(sniffs, histbins)[0])) == 2:
    #     axc[-1].axvline(sniffs.mean(), c='b', alpha=.7, lw=2, dashes=[2,2])
    # else:
    # axc[-1].hist(histbins, color='blue', alpha=.65)
    
    # Format bottom subplot.
    # axc[-1].spines['right'].set_visible(False)
    # axc[-1].spines['top'].set_visible(False)
    # axc[-1].set_ylim(0, 15)


# Create figure.
fig, axes = plt.subplots(6, 3, figsize=(9.5, 8))


# First column, raw data.
_plot_column(
    axes[:, 0],
    our_data
)

# Second column, re-sorted trials by warping function.
_plot_column(
    axes[:, 1],
    our_data.reorder_trials(shift_model.argsort_warps())
)

# Third column, shifted alignment.
_plot_column(
    axes[:, 2],
    validated_alignments
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

# should save from validates
def save_shifted_spikes(spks):
    
    shifted_spikes = {}

    for n in range(spks.n_neurons):
        neuron_id = n  # Assuming n is the neuron ID
        spiketimes = spks.spiketimes[spks.neurons == neuron_id]
        trials = spks.trials[spks.neurons == neuron_id]
        
        # Store spike times and trials in a sub-dictionary
        shifted_spikes[neuron_id] = {
            'spiketimes': spiketimes,
            'trials': trials
        }
        
        return shifted_spikes
        
        
shifted_spikes = save_shifted_spikes(validated_alignments)
            

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