# NetABR: Advancing Multistream Fairness via Bilevel Joint Adaptive Bitrate and Bandwidth Allocation

This is a python implementation of NetABR.

## Requirements

Install the required packages using the follow command. Conda environment is recommended.
 
```
pip install -r requirements.txt
```

## Running NetABR

The main function is in `/train.py`. You can use the following command to run NetABR.

```
python train.py
```

Parameter Settings

- These are the parameters during the training process.

`--save-dir`: Directory to save the results during the training process. 

`--seed`: Seed to initialize the neural network parameters.

`--total-epoch`: Total number of epoch to be trained.

`--video-size-file-dir`: Path to video size files.

`--val-trace-dir`: Directory of the trace data.

`--k`: Number of clusters during pre-processing.

`--time-slot`: Decision frequency ratio between upper and lower levels.

`--device-type`: Type of terminal devices participating in experiment.



- These are the parameters during the evaluating process.

`--pretrained-abr`: Whether to load the ABR model.

`--pretrained-abr_dir`: Directory of the ABR model.

`--pretrained-net`: Whether to load the Net model.

`--pretrained-net-dir`: Directory of the Net model.
