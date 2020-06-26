from scipy.stats import norm
import numpy as np
from dataset import init
from utils import shuffle_split
from random import randint, shuffle
from math import floor
from time import time

from GaussianNB import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
bnb = BernoulliNB()
svc = SVC()
lor = LogisticRegression()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
bc = BaggingClassifier()
dt = DecisionTreeClassifier()

classifiers = [gnb, bnb, svc, lor, knn, rf, bc, dt]


def run(classifier, f):
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
    f2_score = (2*recall*precision)/(recall+precision)
    cli_output = """\
model:          {model:s}
training name:  {train_name:s}
testing name:   {test_name:s}
split ratio:    {ratio:f}
training size:  {train_size:d}
accuracy:       {accuracy:f}
recall:         {recall:f}
precision:      {precision:f}
f2_score:       {f2_score:f}
""".format(model=classifier.__class__.__name__,
           train_name="voice", test_name="voice", ratio=split_ratio,
           train_size=len(training_set), test_size=len(testing_set),
           accuracy=accuracy, recall=recall,
           precision=precision, f2_score=f2_score)
    csv_output = ("{model:s},""{train_name:s},""{test_name:s},""{ratio:f},""{train_size:d},""{test_size:d},"
                  "{accuracy:f},""{recall:f},""{precision:f},""{f2_score:f}"
                  ).format(model=classifier.__class__.__name__,
                           train_name="voice", test_name="voice", ratio=split_ratio,
                           train_size=len(training_set), test_size=len(testing_set),
                           accuracy=accuracy, recall=recall,
                           precision=precision, f2_score=f2_score)
    print(cli_output)
    f.write(csv_output+"\n")


filename = "result_%s.csv" % floor(time())
f = open(filename, "a")
header = "model,train_name,test_name,ratio,train,test,accuracy,recall,precision,f2_score"
f.write(header+"\n")
for classifier in classifiers:
    classifier.fit(training_set, training_labels)
    run(classifier, f)
f.close()
