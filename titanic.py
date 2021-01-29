# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os

import matplotlib
matplotlib.use("agg")    # must select backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

# D-Wave Ocean tools
import dimod
from dwave.system import DWaveCliqueSampler


# Define MI calculations
def prob(dataset):
    """Joint probability distribution P(X) for the given data."""

    # bin by the number of different values per feature
    num_rows, num_columns = dataset.shape
    bins = [len(np.unique(dataset[:, ci])) for ci in range(num_columns)]

    prob, _ = np.histogramdd(dataset, bins)
    return prob / np.sum(prob)


def shannon_entropy(p):
    """Shannon entropy H(X) is the negative sum of P(X)log(P(X)) for probability
    distribution P(X).
    """
    p = p.flatten()
    return -sum(pi*np.log2(pi) for pi in p if pi)


def conditional_shannon_entropy(p, *conditional_indices):
    """Conditional Shannon entropy H(X|Y) = H(X,Y) - H(Y)."""

    axis = tuple(i for i in np.arange(len(p.shape))
                 if i not in conditional_indices)

    return shannon_entropy(p) - shannon_entropy(np.sum(p, axis=axis))


def mutual_information(prob, j):
    """Mutual information between variables X and variable Y.

    Calculated as I(X; Y) = H(X) - H(X|Y)."""

    return (shannon_entropy(np.sum(prob, axis=j))
            - conditional_shannon_entropy(prob, j))


def conditional_mutual_information(p, j, *conditional_indices):
    """Mutual information between variables X and variable Y conditional on variable Z.

    Calculated as I(X;Y|Z) = H(X|Z) - H(X|Y,Z)"""

    return (conditional_shannon_entropy(np.sum(p, axis=j), *conditional_indices)
            - conditional_shannon_entropy(p, j, *conditional_indices))

def maximum_energy_delta(bqm):
    """Compute conservative bound on maximum change in energy when flipping a single variable"""
    delta_max = 0
    for i in bqm.iter_variables():
        delta = abs(bqm.get_linear(i))
        for j in bqm.iter_neighbors(i):
            delta += abs(bqm.get_quadratic(i,j))
        if delta > delta_max:
            delta_max = delta
    return delta_max


def run_demo():
    # Read the feature-engineered data into a pandas dataframe
    # Data obtained from http://biostat.mc.vanderbilt.edu/DataSets
    demo_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(demo_path, 'data', 'formatted_titanic.csv')
    dataset = pd.read_csv(data_path)

    # Rank the MI between survival and every other variable
    scores = {}
    features = list(set(dataset.columns).difference(('survived',)))
    for feature in features:
        scores[feature] = mutual_information(prob(dataset[['survived', feature]].values), 0)

    labels, values = zip(*sorted(scores.items(), key=lambda pair: pair[1], reverse=True))

    # Plot the MI between survival and every other variable
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Mutual Information")
    ax1.set_ylabel('MI Between Survival and Feature')
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.bar(np.arange(len(labels)), values)

    # The Titanic dataset provides a familiar, intuitive example available in the public
    # domain. In itself, however, it is not a good fit for solving by sampling. Run naively on
    # this dataset, it finds numerous good solutions but is unlikely to find the exact optimal solution.
    # There are many techniques for reformulating problems for the D-Wave system that can
    # improve performance on various metrics, some of which can help narrow down good solutions
    # to closer approach an optimal solution.
    # This demo solves the problem for just the highest-scoring features.

    # Select 8 features with the top MI ranking found above.
    keep = 8

    sorted_scores = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    dataset = dataset[[column[0] for column in sorted_scores[0:keep]] + ["survived"]]
    features = list(set(dataset.columns).difference(('survived',)))

    # Build a QUBO that maximizes MI between survival and a subset of features
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Add biases as (negative) MI with survival for each feature
    for feature in features:
        mi = mutual_information(prob(dataset[['survived', feature]].values), 1)
        bqm.add_variable(feature, -mi)

    # Add interactions as (negative) MI with survival for each set of 2 features
    for f0, f1 in itertools.combinations(features, 2):
        cmi_01 = conditional_mutual_information(prob(dataset[['survived', f0, f1]].values), 1, 2)
        cmi_10 = conditional_mutual_information(prob(dataset[['survived', f1, f0]].values), 1, 2)
        bqm.add_interaction(f0, f1, -cmi_01)
        bqm.add_interaction(f1, f0, -cmi_10)

    # Set up a QPU sampler with a fully-connected graph of all the variables
    sampler = DWaveCliqueSampler()

    # For each number of features, k, penalize selection of fewer or more features
    selected_features = np.zeros((len(features), len(features)))

    # Specify the penalty based on the maximum change in the objective
    # that could occur by flipping a single variable.  This ensures
    # that the ground state will satisfy the constraints.
    penalty = maximum_energy_delta(bqm)

    for k in range(1, len(features) + 1):
        kbqm = bqm.copy()
        kbqm.update(dimod.generators.combinations(features, k,
                                                  strength=penalty))  # Determines the penalty

        sample = sampler.sample(kbqm,
                                label='Example - MI Feature Selection',
                                num_reads=10000).first.sample

        for fi, f in enumerate(features):
            selected_features[k-1, fi] = sample[f]

    # Plot the best feature set per number of selected features
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Best Feature Selection")
    ax2.set_ylabel('Number of Selected Features')
    ax2.set_xticks(np.arange(len(features)))
    ax2.set_xticklabels(features, rotation=90)
    ax2.set_yticks(np.arange(len(features)))
    ax2.set_yticklabels(np.arange(1, len(features)+1))
    # Set a grid on minor ticks
    ax2.set_xticks(np.arange(-0.5, len(features)), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(features)), minor=True)
    ax2.grid(which='minor', color='black')
    ax2.imshow(selected_features, cmap=colors.ListedColormap(['white', 'red']))

    plots_path = os.path.join(demo_path, "plots.png")
    plt.savefig(plots_path, bbox_inches="tight")
    print("Your plots are saved to {}".format(plots_path))


if __name__ == "__main__":
    run_demo()
