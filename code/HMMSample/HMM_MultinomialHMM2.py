#problem 2 sample
import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]])
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))
