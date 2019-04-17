## A Recurrent Neural Network for Disruption Prediction

This repository contains a neural network to predict disruptions from bolometer data in two different ways:

* Time to disruption (`ttd`): predicts the remaining time towards an impending disruption. In this case, the model is trained on disruptive pulses only.

* Probability of disruption (`prd`): predicts whether a pulse is disruptive or not. In this case, the model is trained on both disruptive and non-disruptive pulses.

### Requirements

- Keras 2.1.2, TensorFlow 1.4.1 (minimum)

- Configure `~/.keras/keras.json` as follows:

```
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
    "image_data_format": "channels_last",
}

```

### Instructions

- Run `dst_bolo.py` to get the disruption time and bolometer data from every pulse.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - The disruption time is is obtained from the JET disruption database.

    - The bolometer data is down-sampled from 5 kHz to 200 Hz (1 sample every 5 ms).

    - An output file `dst_bolo.hdf` will be created.

- Run `model_train.py` on each folder (`prd` and `ttd`) to train the corresponding model.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The model and its weights will be saved to `model.hdf`.
    
    - A log file with the loss and validation loss will be saved to `train.log`.

- During training, run `plot_train.py` to see how the loss and validation loss are evolving.

    - The script will also indicate the epoch where the minimum validation loss was achieved.

- After training both models, run `model_valid.py` to test the models on the validation and test sets.

    - This script will plot the time to disruption and the probability of disruption for each test pulse.
    
    - Each plot will be saved to an `images/` folder as separate PNG file.
    
    - An output file `dst_pred.hdf` will be created with the output of both networks for each validation pulse.
