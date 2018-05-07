# A Recurrent Neural Network for Disruption Prediction

### Requirements

- Keras, TensorFlow


### Instructions

- Run `dst.py` to find all the disruptive pulses and their disruption times.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - The disruption time is defined as the moment when the current gradient reaches 20 MA/s.

    - An output file `dst.txt` will be created with the pulse numbers and disruption times.
