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

- Depending on the desired prediction (either time to disruption or probability of disruption), change to the corresponding directory (`ttd` or `pd`, respectively).

- Run `model_train.py` to train the model.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The model will be saved in `model.hdf`.

- During training, run `plot_loss.py` to see how the loss and validation loss are evolving.

    - This script will indicate the epoch where the minimum validation loss was achieved.
