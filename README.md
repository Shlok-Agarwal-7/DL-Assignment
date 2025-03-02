# MNIST Model Experimentation

This repository contains experimentation on different model architectures and hyperparameter tuning for the MNIST dataset. The goal is to identify the best-performing model based on accuracy and efficiency.

## Project Structure

- `mnist_experiments.ipynb` - Jupyter Notebook containing various model architectures and their evaluations.
- `README.md` - This documentation file.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy matplotlib
```

## Model Training

### Using Jupyter Notebook

1. Open `mnist_experiments.ipynb` in Jupyter Notebook.
2. Run the notebook cells sequentially to train and evaluate different models.
3. The best model parameters are recorded in the notebook.

### Using the Modular Function

You can train a model using the `create_model` function :

```python

model = create_model(input_shape=(28, 28),
                     hidden_layers=[128, 64, 32],
                     activation='relu',
                     output_units=10,
                     optimizer='adam',
                     learning_rate=0.001,
                     output_activation = 'softmax',
                     use_xavier = True)
```

Compile and train the model using:

```python
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

## Model Evaluation

After training, evaluate the model performance using:

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Coding Style and Clarity

- Functions are modular and reusable.
- Hyperparameters are clearly defined and adjustable.
- The README provides clear instructions for training and evaluating models.

## Comparing the cross entropy loss with the squared error loss
- Mean Squared error doesnt perform well because its meant for regression models and not doesnt performance well with classification tasks

## Results and Best Model

The best model configuration found in experiments is:
- Hidden Layers: `[128, 64, 64, 64, 32]`
- Output units : `10`
- Output activation : `softmax`
- Activation: `ReLU`
- Optimizer: `Adam`
- Learning Rate: `0.001`
- Kernel Intialization: `Xavier`

## 3 recommendations for MNIST dataset :
**Model 1**
- Hidden Layers: `[128, 64, 64, 64, 32]`
- Output units : `10`
- Output activation : `softmax`
- Activation: `ReLU`
- Optimizer: `Adam`
- Learning Rate: `0.001`
- Kernel Intialization: `Xavier`
- Accuracy : `98%`

**Model 1**
- Hidden Layers: `[128, 64, 64, 64, 32]`
- Output units : `10`
- Output activation : `softmax`
- Activation: `ReLU`
- Optimizer: `Adam`
- Learning Rate: `0.001`
- Kernel Intialization: `Xavier`
- Accuracy : `98%`

## Conclusion

This experimentation provides insights into optimal hyperparameters for MNIST classification. Future improvements may include adding convolutional layers (CNNs) for better feature extraction.


