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

    # Sanity check on validity of conditional_indices.  In particular,
    # try to trap issues in which dimensions have been removed from
    # probability table through marginalization, but
    # conditional_indices were not updated accordingly.
    assert(all(ci < p.ndim for ci in conditional_indices))

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

    # Compute an updated version of the conditional indices for use
    # when the probability table is marginalized over dimension j.
    # This marginalization removes one dimension, so any conditional
    # indices pointing to dimensions after this one must be adjusted
    # accordingly.
    marginal_conditional_indices = [i-1 if i > j else i for i in conditional_indices]

    return (conditional_shannon_entropy(np.sum(p, axis=j), *marginal_conditional_indices)
            - conditional_shannon_entropy(p, j, *conditional_indices))


def maximum_energy_delta(bqm):
    """Compute conservative bound on maximum change in energy when flipping a single variable"""
    return max(abs(bqm.get_linear(i))
               + sum(abs(bqm.get_quadratic(i,j))
                     for j, _ in bqm.iter_neighborhood(i))
               for i in bqm.variables)


def mutual_information_bqm(dataset, features, target):
    """Build a BQM that maximizes MI between survival and a subset of features"""
    variables = ((feature, -mutual_information(prob(dataset[[target, feature]].values), 1))
                 for feature in features)
    interactions = ((f0, f1, -conditional_mutual_information(prob(dataset[[target, f0, f1]].values), 1, 2))
                    for f0, f1 in itertools.permutations(features, 2))
    return dimod.BinaryQuadraticModel(variables, interactions, 0, dimod.BINARY)


def add_combination_penalty(bqm, k, penalty):
    """Create a new BQM with an additional penalty biased towards k-combinations"""
    kbqm = dimod.generators.combinations(bqm.variables, k, strength=penalty)
    kbqm.update(bqm)
    return kbqm


def mutual_information_feature_selection(dataset, features, target, num_reads=5000):
    """Run the MIFS algorithm on a QPU solver"""
    
    # Set up a QPU sampler that embeds to a fully-connected graph of all the variables
    sampler = DWaveCliqueSampler()

    # For each number of features, k, penalize selection of fewer or more features
    selected_features = np.zeros((len(features), len(features)))

    bqm = mutual_information_bqm(dataset, features, target)

    # This ensures that the soltion will satisfy the constraints.
    penalty = maximum_energy_delta(bqm)

    for k in range(1, len(features) + 1):
        kbqm = add_combination_penalty(bqm, k, penalty)
        sample = sampler.sample(kbqm,
                                label='Example - MI Feature Selection',
                                num_reads=num_reads).first.sample
        for fi, f in enumerate(features):
            selected_features[k-1, fi] = sample[f]
    return selected_features


def run_demo(dataset, target):
    """Compute MIFS for each value of k and visualize results"""

    # Rank the MI between survival and every other variable
    scores = {feature: mutual_information(prob(dataset[[target, feature]].values), 0)
              for feature in set(dataset.columns) - {target}}

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
    features = sorted(list(set(dataset.columns) - {target}))
    selected_features = mutual_information_feature_selection(dataset, features, target)
    
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


if __name__ == "__main__":
    # Read the feature-engineered data into a pandas dataframe
    # Data obtained from http://biostat.mc.vanderbilt.edu/DataSets
    demo_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(demo_path, 'data', 'formatted_titanic.csv')
    dataset = pd.read_csv(data_path)
    run_demo(dataset, 'survived')
    plots_path = os.path.join(demo_path, "plots.png")
    plt.savefig(plots_path, bbox_inches="tight")
    print("Your plots are saved to {}".format(plots_path))

