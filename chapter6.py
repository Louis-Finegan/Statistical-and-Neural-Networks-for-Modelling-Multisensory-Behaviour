"""Reproduce figures from MAP_ANN_Single_Output.ipynb."""
from src.common import np, plt, torch, nn, Trainer, configure, save_figure, save_checkpoint
from src import networks as ANN, mle as MLE, map as MAP, plotting as pw
from src.simulation import PerceivedUniSensoryData, UniSensorySimulation1, map_reference, evaluate_ann

def main():
    options = configure('chapter6')
    def make_trainer(model, criterion, optimizer):
        return Trainer(model, criterion, optimizer, epochs=options.epochs)
    Neural_Net = ANN.ANN3()
    visual_mean = 10
    audio_mean = 0
    Visual_stimuli = PerceivedUniSensoryData(visual_mean, 2.5, 2000)
    Audio_stimuli = PerceivedUniSensoryData(audio_mean, 8, 2000)
    w_audio = 2.5 ** 2 / (2.5 ** 2 + 8 ** 2)
    w_visual = 8 ** 2 / (2.5 ** 2 + 8 ** 2)
    MLE = w_audio * Audio_stimuli + w_visual * Visual_stimuli
    mean_prop = 0.2904090909090909
    Integration_unity = mean_prop * MLE + (1 - mean_prop) * Audio_stimuli
    Integration_nonunity = Audio_stimuli
    Integration = np.concatenate((Integration_unity, Integration_nonunity))
    mask = np.abs(Visual_stimuli - Audio_stimuli) < 8
    Visual_stimuli = np.concatenate((Visual_stimuli, Visual_stimuli))
    Audio_stimuli = np.concatenate((Audio_stimuli, Audio_stimuli))
    X = torch.tensor(np.column_stack((Visual_stimuli, Audio_stimuli)), dtype=torch.float32)
    Y = torch.tensor(Integration, dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(Neural_Net.parameters(), lr=0.01)
    trainer_nueral_net1 = make_trainer(Neural_Net, criterion, optimizer)
    trainer_nueral_net1.train(X, Y)
    weights_nn1 = trainer_nueral_net1.get_weights_history()
    loss_nn1 = trainer_nueral_net1.get_loss_history()
    save_checkpoint(Neural_Net, 'chapter6_ANN.pth')
    plt.figure(figsize=(10, 6))
    plt.plot(loss_nn1, linewidth=3, color='black')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Loss')
    plt.grid(True)
    save_figure()
    sources = np.arange(-25, 26, step=5)
    audio_loc = np.array([0])
    Visual_stimuli_many = UniSensorySimulation1(sources, 2.5, 2000)
    Audio_stimuli_many = UniSensorySimulation1(audio_loc, 8, 2000)
    stimuli_means_nn1, stimuli_stds_nn1 = evaluate_ann(Visual_stimuli_many, Audio_stimuli_many, sources, Neural_Net)
    model_Std_unity, model_Std_nonunity, model_means_unity, model_means_nonunity, model_prop = map_reference()
    plt.figure(figsize=(10, 5))
    plt.plot(sources, stimuli_means_nn1, marker='o', linestyle='-', label='Mean bias')
    plt.plot(sources, model_means_unity, marker='x', linestyle='--', label='Unity', alpha=0.5, color='red')
    plt.plot(sources, model_means_nonunity, marker='x', linestyle='--', label='Non-unity', alpha=0.5, color='green')
    plt.ylim(-1.5, 2)
    plt.xlabel('Spatial Disparity')
    plt.ylabel('Mean Bias')
    plt.legend()
    plt.tight_layout()
    save_figure()
    plt.figure(figsize=(10, 5))
    plt.plot(sources, stimuli_stds_nn1, marker='o', linestyle='-', label='Standard deviation')
    plt.plot(sources, model_Std_unity, marker='x', linestyle='--', label='Unity', alpha=0.5, color='red')
    plt.plot(sources, model_Std_nonunity, marker='x', linestyle='--', label='Non-unity', alpha=0.5, color='green')
    plt.xlabel('Spatial Disparity')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.ylim(0, 13)
    plt.tight_layout()
    save_figure()

if __name__ == "__main__":
    main()
