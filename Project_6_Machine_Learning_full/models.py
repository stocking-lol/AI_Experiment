import nn
import numpy as np

# Question 1
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Useful function:
        update(self, direction, multiplier) in class Parameter in nn.py
        retrieve seft.data in nn.Constant, e.g., if x is an object of nn.Constant, get its data via x.data
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                true_label = nn.as_scalar(y)
                pre_label = self.get_prediction(x)
                if pre_label != true_label:
                    converged = False
                    direction = x if true_label == 1 else nn.Constant(-x.data)
                    self.w.update(direction, 1.0)


# Question 2
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchsize = 50
        self.learning_rate = 0.01
        self.hidden_size = 200

        self.w1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.w3 = nn.Parameter(self.hidden_size, 1)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, self.hidden_size)
        self.b3 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        hidden1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        hidden2 = nn.ReLU(nn.AddBias(nn.Linear(hidden1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(hidden2, self.w3), self.b3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batchsize):
            loss = self.get_loss(x, y)
            gradiant = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

            # 更新梯度
            for param, grad in zip([self.w1, self.b1, self.w2, self.b2], gradiant):
                param.update(grad, -self.learning_rate)

            if nn.as_scalar(loss) < 0.01:
                return


# Question 3
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchsize = 50
        self.learning_rate = 0.05
        self.hidden_size = 200

        self.w1 = nn.Parameter(784, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.w3 = nn.Parameter(self.hidden_size, 10)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, self.hidden_size)
        self.b3 = nn.Parameter(1, 10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        hidden2 = nn.ReLU(nn.AddBias(nn.Linear(hidden1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(hidden2, self.w3), self.b3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for i, (x, y) in enumerate(dataset.iterate_forever(self.batchsize)):
            loss = self.get_loss(x, y)
            gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

            # Update parameters
            for param, grad in zip([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], gradients):
                param.update(grad, -self.learning_rate)

            # Stop condition
            if i % 100 == 0 and dataset.get_validation_accuracy() > 0.975:
                return

# Question 4
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.batch_size = 100
        self.learning_rate = 0.1
        self.hidden_size = 100

        # Initialize parameters
        self.w_xh = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hh = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_h = nn.Parameter(1, self.hidden_size)
        self.w_hy = nn.Parameter(self.hidden_size, 5)
        self.b_y = nn.Parameter(1, 5)
    def run(self, xs):
        """
        Runs the model for a batch of examples.
        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).
        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.
        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Constant(np.zeros((xs[0].data.shape[0], self.hidden_size)))
        for x in xs:
            h = nn.ReLU(nn.AddBias(
                nn.Add(nn.Linear(x, self.w_xh),
                       nn.Linear(h, self.w_hh)
                       ), self.b_h))

        return nn.AddBias(nn.Linear(h, self.w_hy), self.b_y)
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for i, (xs, y) in enumerate(dataset.iterate_forever(self.batch_size)):
            loss = self.get_loss(xs, y)
            params = [self.w_xh, self.w_hh, self.b_h, self.w_hy, self.b_y]
            gradients = nn.gradients(loss, params)

            # Update parameters
            for param, grad in zip(params, gradients):
                param.update(grad, -self.learning_rate)

            # Stop condition
            if i % 100 == 0 and dataset.get_validation_accuracy() > 0.86:
                return