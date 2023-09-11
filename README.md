# Neural Networks
Implementations of various neural networks from scratch in C#. This includes implementation of my own matrix operation library, activation and loss functions.

### Feedforward neural networks
I reimplemented my digit recognition project that I did in Python a few months ago in C# with a general-purpose feedforward neural network.
This supports both stochastic gradient descent and individual training, which achieved 88% and 91% after 5 epochs and 2 minutes of training respectively.
Going forward I can implement customisability to my network to see which one works best with the MNIST database. I can also train my network on other things, such as doodles from the Google Quick Draw dataset.

Since this is very similar to the project I did before, you can have a look at the documentation for that project [here](https://jamesywu.notion.site/Digit-Recognition-a55f1887d0f14154b3caf80fda85a538?pvs=4).

### Recurrent neural networks
I'm currently learning how these work and I'm implementing a traditional RNN as well as the LSTM version.
I tried to train it to classify words from 6 European languages but is only on 55% accuracy so far.
