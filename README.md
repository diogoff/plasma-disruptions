# A Recurrent Neural Network for Disruption Prediction

This repository contains a neural network to predict disruptions from bolometer data in two different ways:

* Time to disruption: predicts the remaining time towards an impending disruption. In this case, the network is trained on disruptive pulses only.

* Probability of disruption: predicts whether the current pulse is disruptive or not. In this case, the network is trained on both disruptive and non-disruptive pulses.

## Requirements

- Keras, TensorFlow

## Instructions

- Run `dst_bolo.py` to get the disruption time and bolometer data for every pulse.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - The disruption time is defined as the moment when the current gradient reaches 20 MA/s. For non-disruptive pulses, the disruption time is zero.

    - The bolometer data is down-sampled from 5 kHz to 200 Hz (1 sample every 5 ms).

    - An output file `dst_bolo.hdf` will be created.

- Run `model_train.py` on each folder (`ttd` and `pd`) to train the model. In both cases:

    - Training will finish automatically once the validation loss no longer improves.
    
    - The model and its weights will be saved to `model.hdf`.
    
    - A log file with the loss and validation loss will be saved to `train.log`.

- During (or after) training, run `plot_train.py` to plot the loss and validation loss.

    - The script will also indicate the epoch where the minimum validation loss was achieved.
