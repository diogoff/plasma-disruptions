# Under construction

## A Recurrent Neural Network for Disruption Prediction

### Requirements

- Keras, TensorFlow


### Instructions

- Run `dst_bolo.py` to get the disruption time and bolometer data for every disruptive pulse.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - The disruption time is defined as the moment when the current gradient reaches 20 MA/s.

    - The bolometer data is subsampled to 1 kHz.

    - An output file `dst_bolo.hdf` will be created.
