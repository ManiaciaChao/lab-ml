from scipy.stats import norm
import numpy as np
from dataset import init
from GaussianNB import GaussianNB
from utils import shuffle_split
from random import randint, shuffle
from math import floor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

features_list = init()
split_ratio = 0.7
# training_set, testing_set = split_dataset(features_list, 0.7)
# training_labels, testing_labels = split_dataset(labels, 0.7)
training_set, testing_set, training_labels, testing_labels = shuffle_split(
    *init(), split_ratio, shuffle=True)

# scaling
sc = StandardScaler()
sc.fit(training_set)
training_set = sc.transform(training_set)
testing_set = sc.transform(testing_set)

# naive bayes
gnb = GaussianNB()
gnb.fit(training_set, training_labels)

# naive bayes
bnb = BernoulliNB()
bnb.fit(training_set, training_labels)

# SVM
svc = SVC()
svc.fit(training_set, training_labels)

# LogisticRegression
# lir = LinearRegression()
# lir.fit(training_set, training_labels)

# LogisticRegression
lor = LogisticRegression()
lor.fit(training_set, training_labels)

rf = RandomForestClassifier()
rf.fit(training_set, training_labels)


classifiers = [lor]


def run(classifier):
    # here we assume that male is positive
    TP = FP = TN = FN = 0
    for i, test in enumerate(testing_set):
        predict = classifier.predict([test])
        # print(predict)
        if testing_labels[i] == predict:
            if predict == "male":
                TP += 1
            else:
                TN += 1
        else:
            if predict == "male":
                FP += 1
            else:
                FN += 1
    # print("ends with TP={} FP={} TN={} FN={}".format(
    #     TP, FP, TN, FN))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP+FN)
    precision = TP / (TP + FP)
    f1_score = (2*recall*precision)/(recall+precision)
    cli_output = """\
model:          {model:s}
training name:  {train_name:s}
testing name:   {test_name:s}
split ratio:    {ratio:f}
training size:  {train_size:d}
testing size:   {test_size:d}
accuracy:       {accuracy:f}
recall:         {recall:f}
precision:      {precision:f}
f1_score:       {f1_score:f}
    """.format(model=classifier.__class__.__name__,
               train_name="voice", test_name="voice", ratio=split_ratio,
               train_size=len(training_set), test_size=len(testing_set),
               accuracy=accuracy, recall=recall,
               precision=precision, f1_score=f1_score)
    csv_output = ("{model:s},""{train_name:s},""{test_name:s},""{ratio:f},""{train_size:d},""{test_size:d},"
                  "{accuracy:f},""{recall:f},""{precision:f},""{f1_score:f}"
                  ).format(model=classifier.__class__.__name__,
                           train_name="voice", test_name="voice", ratio=split_ratio,
                           train_size=len(training_set), test_size=len(testing_set),
                           accuracy=accuracy, recall=recall,
                           precision=precision, f1_score=f1_score)
    print(cli_output)


for classifier in classifiers:
    run(classifier)
