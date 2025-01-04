# 🎯 Goals for week 14

1. Practice working with recurrent neural networks.

## Task 01

**Description:**

Let's tackle the problem of predicting electricity consumption based on past patterns. In our `DATA` folder you'll find the folder `electricity_consumption`. It contains electricity consumption in kilowatts, or kW, for a certain user recorded every 15 minutes for four years.

Define a function `create_sequences` that takes a `pandas` `Dataframe` and a sequence length and returns two `NumPy` arrays, one with input sequences and the other one with the corresponding targets. To test the correctness of the function, apply it on a dataset containing two columns with the numbers from `0` to `100` and output the first five entries in both arrays.

Apply the function on the training set of the electricity consumption data. Output the shape of the `X` and `y` splits and output the length of the `TensorDataset` for training the model.

**Acceptance criteria:**

1. A new function `create_sequences` is defined.
2. The first five entries after applying `create_sequences` on the `electricity_consumption` data are displayed.
3. An object of type `TensorDataset` is created and its length is outputted.

**Test case:**

```console
python task01.py
```

```console
First five training examples: [[0 1 2 3 4]
 [1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]]
First five target values: [5 6 7 8 9]

X_train.shape=(105119, 96)
y_train.shape=(105119,)
Length of training TensorDataset: 105,119
```

## Task 02

**Description:**

The model that we'll use to predict the electricity consumption would need to have a sequence-to-vector architecture.

Define a neural network that will have the following layers:

- two stacked `LSTM` layers with memory size of `32`. Initially, both types of memory will be all zeros;
- a linear layer that will take the recurrent layer's last output.

Apply the model on the data:

- train for `3` epochs with batches of `32` samples;
- use the mean squared error loss from the `torchmetrics` module;
- use the optimizer `AdamW` with a learning rate of `0.0001`.

Output the loss per epoch when training and the loss on the test set.

**Acceptance criteria:**

1. Two `LSTM` layers are stacked.
2. The correct shape for the linear layer is identified.
3. The mean squared error loss function from the `torchmetrics` module is used.
4. `tqdm` is used to visualize the progress through the batches.

**Test case:**

```console
python task02.py
```

```console
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3284/3284 [00:59<00:00, 55.45it/s]
Epoch 1, Average MSE loss: 0.19529657065868378
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3284/3284 [01:32<00:00, 35.69it/s]
Epoch 2, Average MSE loss: 0.12671414017677307
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3284/3284 [01:04<00:00, 50.63it/s]
Epoch 3, Average MSE loss: 0.1027834564447403
Test MSE: 0.09648480266332626
```

## Task 03

**Description:**

Try to create a better model than the one we obtained in the previous tasks. Treat this task as a playground for conducting experiments to understand what makes a good neural network.

**Acceptance criteria:**

1. A training process that results in a model giving lower MSE on the test set and a reasonable R-squared metric.
