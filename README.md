# SotR-ICUA-2026
This repository contains the code used to conduct all data processing steps before manual labelling in the paper: 'Sounding out the river: and end-to-end framework for monitoring bioacoustic events and sediment movement in freshwater soundscapes' submitted to the International Conference on Underwater Acoustics 2026, and to be published on the Institute of Acoustics website. Pre-print available at: https://doi.org/10.32942/X2QT0D

## How to use

Both soundfile_selection.py and soundfile_selection.ipynb contain the exact same material, as an executable python script and a jupyter notebook.

To run the python script, first edit the 'DEFINE PATHS AND GLOBAL VARIABLES' section (lines 11-88) to modify parameters, add your output paths, and enter your key for the SEPA API. Then simply run:

```
python3 soundfile_selection.py
```

The jupyter notebook can be run cell by cell, and is meant as a more accessible complement to the script.

## Requirements #

Python >=3.9

Libraries:
- os
- requests
- shutil
- numpy
- pandas
- scikit-maad
- tqdm
