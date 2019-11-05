import numpy as np

# This serves as our input data.
np.random.seed(1)
data = np.random.normal(5, 10, size=(10,))

# Import nodes for the distribution analysis we are interested in.
from bayespy.nodes import GaussianARD, Gamma

# mu and tau are the variables modeled by our engine. The distributions calculated through
# these functions are the GaussianARD (Gaussian distribution with ARD prior), and the
# standard Gamma distribution. ARD (Automatic Relevance Determination) is a technique taken
# from neural network literature to simplify autoregressive models by setting any unnecessary
# variables to zero.
mu = GaussianARD(0, 1e-6)
tau = Gamma(1e-6, 1e-6)

# P(y|mu,tau) is the result of our Bayesian Inference. The plates keyword here refers
# to plate notation, which is a method of representing variables that repeat in a graphical
# model. Here, we have created 10 plates.
y = GaussianARD(mu, tau, plates=(10,))
y.observe(data)

# Now that we have created a model and provided our data, we can estimate
# the posterior distribution of our data. The posterior distribution is
# a way to summarize what we know about uncertain quantities in Bayesian analysis.

# Import the Variational Bayesian (VB) engine to perform inference on our set.
from bayespy.inference import VB

# At this point we run our Bayesian Inference on the engine P(y|mu,tau) and have the
# algorithm run through 10 iterations or until the point of algorithmic convergence.
Q = VB(mu, tau, y)
Q.update(repeat=10)

# This code simply provides a graphical depiction of our models and calculations.
import bayespy.plot as bpplt
bpplt.pyplot.subplot(2, 1, 1)
bpplt.pdf(mu, np.linspace(-10, 20, num=100), color='k', name=r'\mu')
bpplt.pyplot.subplot(2, 1, 2)
bpplt.pdf(tau, np.linspace(1e-6, 0.08, num=100), color='k', name=r'\tau')
bpplt.pyplot.tight_layout()
bpplt.pyplot.show()