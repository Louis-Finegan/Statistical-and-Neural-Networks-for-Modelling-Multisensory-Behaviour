"""Reproduce figures from Chapter 2-v2.ipynb."""
from src.common import np, plt, Trainer, relative_bias, configure, save_figure
from src import networks as ANN, mle as MLE, map as MAP, plotting as pw

def main():
    options = configure('chapter3')
    sources = np.arange(-25, 26, 5)
    Visual_stimuli_many = [np.random.normal(loc=source, scale=2.5, size=5000) for source in sources]
    Audio_stimuli_many = [np.random.normal(loc=0, scale=8, size=5000) for source in sources]
    accuracies_UNITY = []
    accuracies_NONUNITY = []
    problist = []
    bias_UNITY = []
    bias_NONUNITY = []
    disparity_threshold = 8
    for i in range(len(sources)):
        Visual_stimuli = Visual_stimuli_many[i]
        Audio_stimuli = Audio_stimuli_many[i]
        Spacial_Disparity = np.abs(Visual_stimuli - Audio_stimuli)
        mask = Spacial_Disparity < disparity_threshold
        Visual_stimuli_unity = Visual_stimuli[mask]
        Audio_stimuli_unity = Audio_stimuli[mask]
        Visual_stimuli_nonunity = Visual_stimuli[~mask]
        Audio_stimuli_nonunity = Audio_stimuli[~mask]
        dec = len(Audio_stimuli_unity) / (len(Audio_stimuli_nonunity) + len(Audio_stimuli_unity))
        print(len(Visual_stimuli_unity) / (len(Visual_stimuli_nonunity) + len(Visual_stimuli_unity)))
        problist.append(dec)
        if len(Visual_stimuli_unity) > 0 and len(Audio_stimuli_unity) > 0:
            w1_unity = 1 / np.std(Audio_stimuli_unity) ** 2 / (1 / np.std(Audio_stimuli_unity) ** 2 + 1 / np.std(Visual_stimuli_unity) ** 2)
            w2_unity = 1 / np.std(Visual_stimuli_unity) ** 2 / (1 / np.std(Audio_stimuli_unity) ** 2 + 1 / np.std(Visual_stimuli_unity) ** 2)
            Integration_unity = dec * (w1_unity * Audio_stimuli_unity + w2_unity * Visual_stimuli_unity) + (1 - dec) * Audio_stimuli_unity
            bias_unity = relative_bias(np.mean(Integration_unity), sources[i])
            bias_UNITY.append(bias_unity)
            accuracies_UNITY.append(np.std(Integration_unity))
        else:
            accuracies_UNITY.append(np.nan)
            bias_UNITY.append(np.nan)
        if len(Visual_stimuli_nonunity) > 0 and len(Audio_stimuli_nonunity) > 0:
            Integration_nonunity = Audio_stimuli_nonunity
            accuracies_NONUNITY.append(np.std(Integration_nonunity))
            bias_nonunity = relative_bias(np.mean(Integration_nonunity), sources[i])
            bias_NONUNITY.append(bias_nonunity)
        else:
            accuracies_NONUNITY.append(np.nan)
            bias_NONUNITY.append(np.nan)
    plt.figure(figsize=(10, 5))
    plt.plot(sources, bias_UNITY, 'o-', markersize=8, label='Unity')
    plt.plot(sources, bias_NONUNITY, 'x-', markersize=8, label='Non-Unity')
    plt.xlabel('Spatial Disparity (Degrees)')
    plt.ylabel('Mean Bias')
    plt.xlim(-30, 30)
    plt.legend()
    save_figure()
    plt.figure(figsize=(10, 5))
    plt.plot(sources, accuracies_UNITY, 'o-', markersize=8, label='Unity')
    plt.plot(sources, accuracies_NONUNITY, 'x-', markersize=8, label='Non-Unity')
    plt.xlabel('Spatial Disparity (Degrees)')
    plt.ylabel('Standard Deviation (Degrees)')
    plt.ylim(0, 13)
    plt.xlim(-30, 30)
    plt.legend()
    save_figure()
    probs = np.array(problist)
    print(probs)
    print(np.mean(probs * (1 / (30 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (sources / 30) ** 2) + 1 / 180)))
    print(len(sources))

if __name__ == "__main__":
    main()
