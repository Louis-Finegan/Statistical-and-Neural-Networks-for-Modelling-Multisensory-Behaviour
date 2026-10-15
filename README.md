# Statistical and Neural Networks for Modelling Multisensory Behaviour

This code was developed for my undergraduate dissertation in Mathematical Sciences at Technological University Dublin. The project investigated audiovisual multisensory integration using statistical models and artificial neural networks. It draws on the Bayesian model of audiovisual unity perception developed by Sato et al. (2008), the model of statistically optimal sensory integration presented by Ernst et al. (2002), and the experiments on multisensory unity judgements conducted by Wallace et al. (2004). The code generates simulated audiovisual data, trains neural networks and compares their predictions with the statistical models.

## Files and outputs

| File | Purpose and outputs |
| --- | --- |
| `chapter3.py` | Simulates unity and non-unity judgements, producing bias and standard deviation plots. |
| `chapter5.py` | Trains neural networks to approximate the maximum likelihood estimation (MLE) model. Produces stimulus distributions, training losses, weight histories and comparisons with the statistical model. |
| `chapter6.py` | Trains a single-output neural network using maximum a posteriori (MAP) model targets. Produces training loss, bias and standard deviation plots. |
| `chapter7.py` | Trains a three-output neural network to predict unity and non-unity location estimates and the probability of unity. Produces training losses, location distributions and comparisons with the statistical model. |
| `src/simulation.py` | Generates stimuli and evaluates statistical models and neural networks. |
| `src/networks.py` | Defines the neural network architectures. |
| `src/training.py` | Provides the scalar-output neural network training engine. |
| `src/mle.py` | Defines the MLE statistical model. |
| `src/map.py` | Defines the MAP statistical model. |
| `src/plotting.py` | Produces neural network weight evolution plots. |
| `src/common.py` | Handles configuration, random seeds and figure saving. |
| `main.py` | Runs all four chapter scripts. |

The scripts produce **35 figures**, saved as PNG and PDF files in `figures/`. Re-running a chapter overwrites its figures. The trained neural networks are saved in `models/`.

## How to run

Requires Python 3.10 or later. From the project directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run all chapters:

```bash
python main.py
```

Or run a chapter individually:

```bash
python chapter5.py
```

To specify the random seed and number of training epochs:

```bash
python main.py --seed 42 --epochs 1000 --map-epochs 3000
```

`--epochs` controls the scalar-output networks, while `--map-epochs` controls the three-output network in `chapter7.py`. These are the default values.

To save figures to a different directory:

```bash
python main.py --output-dir figures/custom
```