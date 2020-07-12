# gmms_inpainting

Source code for [Estimating conditional density of missing values using deep Gaussian mixture model](https://openreview.net/forum?id=VR6mXmaHacL).

## Requirements

Python 3.7 or higher is required.
Models have been implemented with PyTorch.

To install the requirements, running:

```bash
pip install -r requirements.txt
```

should suffice.

## Running

To train the DMFA model, see the script:

```bash
python scripts/train.py --h
```
### Classical MFA baseline

The implementation of classical MFA baseline we compare ourselves to in the paper can be found [here](https://github.com/mareksmieja/gmm_missing).

### Notebooks

There are several Jupyter Notebooks in the [notebooks](https://github.com/mprzewie/gmms_inpainting/tree/master/notebooks) directory. 
They were used for initial experiments with the DMFA models, as well as analysis of the results and calculating metrics reported in the paper.

The notebooks are not guaranteed to run 100% correctly due to subsequent code refactor.
