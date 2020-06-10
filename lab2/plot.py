import matplotlib.pyplot as plt


def drawLine(ax, x, y, c, l):
    ax.plot(x, y, c=c, linewidth=1, alpha=0.5, label=l)
    ax.scatter(x, y, c=c, s=15, alpha=0.5)


name = "enron_X - english_mail"

# accuracy, recall, precision, f1_score
# , 0.660000, 0.920000, 0.605263, 0.730159
# , 0.560000, 1.000000, 0.531915, 0.694444
# , 0.820000, 1.000000, 0.735294, 0.847458

ratio = [200, 1000, 3000]
# total_words = [127, 189, 327, 417, 436]
accuracy = [0.660000, 0.560000, 0.820000]
recall = [0.920000, 1.000000, 1.000000]
precision = [0.605263, 0.531915, 0.735294]
f1_score = [0.730159, 0.694444, 0.847458]

plt.figure('%s' % name)
plt.title('%s' % name)
ax = plt.gca()
ax.set_xlabel('enron_X')
# ax.set_ylabel('misclassification ratio')
# ax.set_ylim(bottom=0,top=20)
# drawLine(ax, ratio, total_words, "b", "total words")
drawLine(ax, ratio, accuracy, "green", "accuracy")
drawLine(ax, ratio, recall, "purple", "recall")
drawLine(ax, ratio, precision, "blue", "precision")
drawLine(ax, ratio, f1_score, "red", "f1-score")


plt.legend()
plt.show()
