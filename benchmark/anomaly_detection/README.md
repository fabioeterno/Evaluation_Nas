# MLPerf Tiny anomaly detection reference model

This is the MLPerf Tiny anomaly detection reference model.

- Model: autoencoder
- Dataset: ToyCar

## Quick start

To run the code and replicate the results it is suggested to create a new python virtual environment and install the required libraries:
``` Bash

# Python virtual environment creation
python3 -m venv <environment-name>
# Installation of the required libraries
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the following commands to go through the whole training and validation process:

``` Bash

# If the dataset has not been already download, run the following script
./download_dataset.sh
```
``` Bash

# Train and test the model
./toycar_autoencoder.sh
```
