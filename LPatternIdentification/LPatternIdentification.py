
import pandas as pd
import numpy as np


def feature_set(input_matrix, target_vector):
    
    # generate list of different-class pairs
    values, counts = np.unique(target_vector, return_counts=True)
    class_1_count, class_2_count = counts[0], counts[1]
    x = np.array(list(range(class_1_count)))
    y = np.array(list(range(class_1_count, class_1_count + class_2_count)))
    pairs = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    # generate matrix of feature difference between different-class pairs
    difference_matrix = np.zeros((pairs.shape[0], input_matrix.shape[1]), dtype=np.int8)

    for i, pair in enumerate(pairs):
        sample_1, sample_2 = input_matrix[pair[0]], input_matrix[pair[1]]
        mask = np.equal(sample_1, sample_2)
        indices = np.nonzero(~mask)
        difference_matrix[i, indices] = 1

    # find infeasible rows that contain only zeros
    infeasible_rows = [] 

    for i, row in enumerate(difference_matrix):
        if not np.any(row):
            infeasible_rows.append(i)

    # remove infeasible rows from consideration
    universe = set(range(difference_matrix.shape[0])) - set(infeasible_rows)
    subsets = []

    # pass columns as subsets to set cover algorithm
    for i in range(difference_matrix.shape[1]):
        column = difference_matrix[:,i]
        subset = np.nonzero(column)[0]
        subsets.append(set(subset))
        
    def set_cover(universe, subsets):
        elements = set(e for s in subsets for e in s)

        # Check the subsets cover the universe
        if elements != universe:
            return None
        covered = set()
        global cover
        cover = []

        # Greedily add the subsets with the most uncovered points
        while covered != elements:
            subset = max(subsets, key=lambda s: len(s - covered))
            cover.append(subsets.index(subset))
            covered |= subset
 
        return cover

    cover = set_cover(universe, subsets)

    return sorted(cover)


def split_data(input_matrix, target_vector):
    global Good, Bad, split_point
    split_point = target_vector.index(0)
    Good = input_matrix[:split_point]
    Bad = input_matrix[split_point:]
    return split_point, Good, Bad


def get_patterns_from_feature_set(Good, feature_set):
    Patterns = []
    for g in Good:
        p = np.empty(len(g), dtype='unicode_')
        for i in range(len(g)):
            if i in feature_set:
                p[i] = str(g[i])
            else:
                p[i] = '*'
        Patterns.append(p)

    return np.unique(Patterns, axis=0)


def reduce_pattern_set(Patterns, Bad, l):
    while Patterns.shape[0] > l:
        distances = get_distances(Patterns)
        Patterns_copy = Patterns.copy()

        for i in range(len(distances)):
            instance = distances[i]
            pattern1_index = instance[0]
            pattern2_index = instance[1]
            indices = instance[-1]
            pattern1 = Patterns[pattern1_index]
            pattern2 = Patterns[pattern2_index]
            pattern3 = make_compatible(pattern1, pattern2, indices)
            Patterns[pattern2_index] = pattern3
            Patterns = remove_redundant_patterns(Patterns)
            
            if not is_Bad_compatible(Patterns, Bad):
                break
            else:
                Patterns = Patterns_copy

    return Patterns


def make_compatible(pattern1, pattern2, indices):                   
    pattern3 = pattern2.copy()
    for i in indices:
        pattern3[i] = '*'
    if is_string_compatible(pattern3, pattern1):
        return pattern3
    else:
        return pattern2
    

def is_string_compatible(pattern, good):
    for i in range(len(pattern)):
        if pattern[i] == '*':
            continue
        if not pattern[i] == good[i]:
            return False
    return True
    

def remove_redundant_patterns(Patterns):
    indices = []
    removed = []
    for i in range(Patterns.shape[0]):
        for j in range(Patterns.shape[0]):
            if i == j:
                continue
            if is_string_compatible(Patterns[i], Patterns[j]):
                indices.append(j)

    for i in indices:
        removed.append(list(Patterns[i]))
    Patterns_reduced = np.delete(Patterns, indices, axis=0)

    return Patterns_reduced


def is_Good_compatible(Patterns, Good):
    Good_is_compatible = True
    
    for i in range(Good.shape[0]):
        g = Good[i]
        compatible = False
        for j in range(Patterns.shape[0]):
            p = Patterns[j]
            if is_string_compatible(p, g):
                compatible = True
                break
        if not compatible:
            Good_is_compatible = False
            return Good_is_compatible

    return Good_is_compatible


def is_Bad_compatible(Patterns, Bad):
    Bad_is_compatible = False
    
    for b in Bad:
        for p in Patterns:
            if is_string_compatible(p, b):
                Bad_is_compatible = True
                break

    return Bad_is_compatible


def get_distances(Patterns):
    distances = []
    for i, sample_1 in enumerate(Patterns):
        for j, sample_2 in enumerate(Patterns):
            distance, indices = hamming_distance_unicode(sample_1, sample_2)
            if not i == j:
                distances.append([i, j, distance, indices])

    return sorted(distances, key=lambda x: x[2])


def hamming_distance_unicode(sample_1, sample_2):
    distance = 0
    indices = []
    for k, (i, j) in enumerate(zip(sample_1, sample_2)):
        if not i == j:
            distance += 1
            indices.append(k)
    return distance, indices

