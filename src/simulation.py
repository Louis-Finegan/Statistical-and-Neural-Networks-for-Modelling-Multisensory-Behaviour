"""Shared sampling and evaluation routines extracted from the notebooks."""
from .common import np, torch, relative_bias
from . import map as MAP, mle as MLE

def PerceivedUniSensoryData(x, sd, NTrials):
    return np.random.normal(x, sd, NTrials)

def UniSensorySimulation1(stimuli, sd, NTrials):
    return np.array([np.random.normal(stimulus, sd, NTrials) for stimulus in stimuli])

def jumble_data(stimuli, n_points=100):
    jumbled_data = []
    for i in range(len(stimuli)):
        samples = np.random.choice(stimuli[i], n_points, replace=False)
        jumbled_data.extend(samples)
    jumbled_data = np.array(jumbled_data)
    return jumbled_data

def evaluate_ann(Visual_stimuli_many, Audio_stimuli_many, sources, NN):
    stimuli_means = []
    stimuli_stds = []
    for i in range(len(Visual_stimuli_many)):
        Visual_stimuli = Visual_stimuli_many[i]
        Audio_stimuli = Audio_stimuli_many[0]
        X = torch.tensor(np.column_stack((Visual_stimuli, Audio_stimuli)), dtype=torch.float32)
        X_test = NN(X)
        stimuli_means.append(relative_bias(X_test.detach().numpy().mean(), sources[i]))
        stimuli_stds.append(X_test.detach().numpy().std())
    return (np.array(stimuli_means), np.array(stimuli_stds))

def mle_reference(Visual_stimuli_many, Audio_stimuli_many, sources):
    model_means = []
    model_stds = []
    for i in range(len(Visual_stimuli_many)):
        Visual_stimuli = Visual_stimuli_many[i]
        Audio_stimuli = Audio_stimuli_many[0]
        Model = MLE.ModelMLE(Audio_stimuli, Visual_stimuli)
        _model = Model.model()
        model_means.append(relative_bias(np.mean(_model), sources[i]))
        model_stds.append(np.std(_model))
    return (np.array(model_means), np.array(model_stds))

def map_reference():
    sources = np.arange(-25, 26, step=5)
    audio_loc = np.array([0])
    model_Std_unity = []
    model_Std_nonunity = []
    model_means_unity = []
    model_means_nonunity = []
    model_prop = []
    audio_locs = UniSensorySimulation1(audio_loc, 8, 2000)
    visual_locs = UniSensorySimulation1(sources, 2.5, 2000)
    for i in range(len(sources)):
        audio_loc = audio_locs[0].reshape(-1)
        visual_loc = visual_locs[i].reshape(-1)
        Model = MAP.ModelMAP(visual_loc, audio_loc, np.abs(sources[i] - 0))
        model_prop.append(np.mean(np.abs(visual_loc - audio_loc) < 8))
        Int_unity = Model.model_unity()
        Int_nonunity = Model.model_nonunity()
        model_Std_unity.append(np.std(Int_unity))
        model_Std_nonunity.append(np.std(Int_nonunity))
        model_means_unity.append(relative_bias(np.mean(Int_unity), sources[i]))
        model_means_nonunity.append(relative_bias(np.mean(Int_nonunity), sources[i]))
    return (model_Std_unity, model_Std_nonunity, model_means_unity, model_means_nonunity, model_prop)

def evaluate_map(Visual_stimuli_many, Audio_stimuli_many, sources, NN):
    stimuli_means_unity = []
    stimuli_means_nonunity = []
    stimuli_stds_unity = []
    stimuli_stds_nonunity = []
    stimuli_prop = []
    for i in range(len(Visual_stimuli_many)):
        Visual_stimuli = Visual_stimuli_many[i]
        Audio_stimuli = Audio_stimuli_many[0]
        X_test = torch.tensor(np.column_stack((Visual_stimuli, Audio_stimuli)), dtype=torch.float32)
        Y_test = NN(X_test)
        unity = Y_test.detach().numpy()[:, 0]
        nonunity = Y_test.detach().numpy()[:, 1]
        prop = Y_test.detach().numpy()[:, 2]
        stimuli_means_unity_temp = []
        stimuli_means_nonunity_temp = []
        stimuli_stds_unity_temp = []
        stimuli_stds_nonunity_temp = []
        stimuli_prop_temp = []
        for j in range(len(Visual_stimuli)):
            stimuli_prop_temp.append(prop[j])
            if np.abs(Visual_stimuli[j] - Audio_stimuli[j]) < 8:
                stimuli_means_unity_temp.append(unity[j])
                stimuli_stds_unity_temp.append(unity[j])
            else:
                stimuli_means_nonunity_temp.append(nonunity[j])
                stimuli_stds_nonunity_temp.append(nonunity[j])
        stimuli_means_unity.append(relative_bias(np.mean(stimuli_means_unity_temp), sources[i]))
        stimuli_stds_unity.append(np.std(stimuli_stds_unity_temp))
        stimuli_means_nonunity.append(relative_bias(np.mean(stimuli_means_nonunity_temp), sources[i]))
        stimuli_stds_nonunity.append(np.std(stimuli_stds_nonunity_temp))
        stimuli_prop.append(np.mean(stimuli_prop_temp))
    return (np.array(stimuli_means_unity), np.array(stimuli_stds_unity), np.array(stimuli_means_nonunity), np.array(stimuli_stds_nonunity), np.array(stimuli_prop))
