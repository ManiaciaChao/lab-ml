import matplotlib.pyplot as plt


name = "english_email"

x_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

y_list = [0.14545454545454545,
          0.07272727272727272,
          0.01818181818181818,
          0.01818181818181818,
          0.00909090909090909,
          0.00909090909090909]

plt.figure('misclassification rate for %s' % name)
plt.title("misclassification rate with different split rate")
ax = plt.gca()
ax.set_xlabel('k')
ax.set_ylabel('misclassification rate')
# ax.set_ylim(bottom=0,top=20)
ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
ax.scatter(x_list, y_list, c='b', s=20, alpha=0.5)
plt.show()
