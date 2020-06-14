from numpy import genfromtxt, char


def init(filename="voice.csv", delimiter=','):
    feature_cols = [i for i in range(0, 21)]
    features = genfromtxt(filename, delimiter=delimiter,
                          usecols=feature_cols, names=True, )
    labels = genfromtxt(filename, delimiter=delimiter, usecols=20, dtype="U")
    stripped = char.strip(labels, "\"")
    return features, stripped
