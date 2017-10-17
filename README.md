# kaggle-redefining-cancer-treatment-ailabs
This repository contains solution to Kaggle competition "Personalized Medicine: Redefining Cancer Treatment" by Taiwan AI Labs

## Prerequisites
* Python 3.4 or above
* virtualenv
* required packages are listed in requirements.txt

## Description
Detailed description can be found in `description.pdf`

## How to run
After cloning this repository, we recommend setting up a python virtual environment

    virtualenv kaggle

Remember to activate the virtual environment before running

    source kaggle/bin/activate

For running all the experiments, simply use `run.sh`:

    bash run.sh

`run.sh` will run `pretrain.py` and then `train.py`. 

After a successful run, you should find two submission file placed in `result/` directory.
