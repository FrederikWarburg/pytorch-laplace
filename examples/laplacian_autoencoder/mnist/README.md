# MNIST dataset

All notebooks in this folder train on MNIST hand-written-digits dataset, a collection of black and white 28x28 images.
In order to keep it as simple and fast as possible, only the first 100 images are used (+ other 100 for testing)

### Neural Network Architecture

All notebooks share the same ___autoencoder___ architecture. It is a [784, 50, 2, 50, 784] fully connected with Tanh non-linearities. Moreover there is a l2 normalization layer before the latent space, such that the latent representation actually lays on the 1-D unit circle.

The ***encoder*** is defined as
```python
encoder = nnj.Sequential(
                nnj.Flatten(),
                nnj.Linear(28*28, 50),
                nnj.Tanh(),
                nnj.Linear(50, 2),
                nnj.L2Norm(),
                add_hooks = True
        )
```

The ***decoder*** is defined as
```python
decoder = nnj.Sequential(
                nnj.Flatten(),
                nnj.Linear(2, 50),
                nnj.Tanh(),
                nnj.Linear(50, 28*28),
                nnj.Reshape(1, 28, 28),
                add_hooks = True
        )
```

They are combined in the ***model*** with
```python
model = nnj.Sequential(
                encoder,
                decoder,
                add_hooks = True
        )
```
