# LPatternIdentification


https://pypi.org/project/LPatternIdentification/


The formal mathematical definition of the l-Pattern Identification Problem is as follows:

## Input:

A finite alphabet Σ, two disjoint sets Good, Bad ⊆ Σn of strings and an integer l > 0

## Problem question:

Is there a set of P patterns such that: |P| ≤ l and P → (Good, Bad)?


## Demonstration:

See demo.ipynb


## 1. Install:

```
pip install LPatternIdentification
```

## 2. Load:

```
from LPatternIdentification import feature_set, split_data, get_patterns_from_feature_set, reduce_pattern_set
```

## 3. Prepare data:

Sort dataset by class labels

Separate observations into numpy ndarray

Separate labels into list

## 4. Find set of features

```
features = feature_set(observations, labels)
```

## 5. Split observations by classes 

Here, classes are named 'Good' and 'Bad', the 'Good' class being the class of our interest.

```
split_point, Good, Bad  = split_data(elections_X, elections_y)
```

## 6. Return a set of patterns that contain the features and are Good

```
Patterns = get_patterns_from_feature_set(Good, elections_feature_set)
```

## 7. Identify 'L' number of patterns such that all patterns are uniquely Good and not similar to Bad patterns

```
L = 7
Patterns_identified = reduce_pattern_set(Patterns, Bad, L)
```