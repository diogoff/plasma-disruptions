# Recurrent Neural Networks for Disruption Prediction

This repository contains neural networks to predict disruptions from bolometer data in two different ways:

* Time to disruption: predicts the remaining time towards an impending disruption. This network is trained on disruptive pulses only.

* Disruption probability: predicts whether the current pulse is disruptive or not. This network is trained on both disruptive and non-disruptive pulses.

## Requirements

- Keras, TensorFlow

## Instructions

### Time to disruption

- Run `dst_bolo.py` to get the disruption time and bolometer data for every disruptive pulse.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - The disruption time is defined as the moment when the current gradient reaches 20 MA/s.

    - The bolometer data is down-sampled from 5 kHz to 200 Hz (1 sample every 5 ms).

    - An output file `dst_bolo.hdf` will be created.

- Run `model_train.py` to train the model.

    - Before running this script, set the number of GPUs and the batch size accordingly.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The best model parameters will be saved in `model_weights.hdf`.

- During training, run `plot_loss.py` to see how the loss and validation loss are evolving.

    - The script will indicate the epoch where the minimum validation loss was achieved.

### Disruption probability

- Run `bolo_data.py` to get the bolometer data for every pulse.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - The bolometer data is down-sampled from 5 kHz to 200 Hz (1 sample every 5 ms).

    - An output file `bolo_data.hdf` will be created.
