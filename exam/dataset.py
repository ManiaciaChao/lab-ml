from numpy import genfromtxt, char, mean, var


def init(filename="voice.csv", delimiter=','):
    feature_cols = [i for i in range(0, 20)]
    features = genfromtxt(filename, delimiter=delimiter,
                          usecols=feature_cols, skip_header=True)
    labels_raw = genfromtxt(
        filename, delimiter=delimiter, usecols=20, dtype="U", skip_header=True)
    labels = char.strip(labels_raw, "\"")
    return features, labels
