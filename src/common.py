"""Configuration, deterministic execution and figure output shared by scripts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from .training import Trainer

ROOT = Path(__file__).resolve().parents[1]
_output = ROOT / 'figures'
_experiment = ''
_count = 0
_records = []

def relative_bias(mean, disparity):
    """Notebook mean/disparity statistic; undefined at zero disparity."""
    return mean / disparity if disparity != 0 else np.nan

def configure(experiment):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--map-epochs', type=int, default=3000)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    options = parser.parse_args()
    if options.epochs < 1 or options.map_epochs < 1:
        parser.error('Epoch counts must be positive.')
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.set_num_threads(1)
    global _output, _experiment, _count, _records
    _output = options.output_dir.resolve()
    _output.mkdir(parents=True, exist_ok=True)
    _experiment, _count, _records = experiment, 0, []
    return options

def save_figure():
    """Save each original figure in sequence, then release its memory."""
    global _count
    _count += 1
    fig = plt.gcf()
    stem = f'{_experiment}_{_count:02d}'
    for extension in ('png', 'pdf'):
        temporary = _output / f'{stem}.tmp.{extension}'
        fig.savefig(temporary, dpi=180, bbox_inches='tight')
        temporary.replace(_output / f'{stem}.{extension}')
    _records.append({'file': stem, 'titles': [a.get_title() or a.get_title(loc='left') for a in fig.axes]})
    (_output / f'{_experiment}_manifest.json').write_text(json.dumps(_records, indent=2)+'\n')
    plt.close(fig)

def save_checkpoint(model, filename):
    destination = ROOT / 'models'
    destination.mkdir(exist_ok=True)
    torch.save(model.state_dict(), destination / filename)
