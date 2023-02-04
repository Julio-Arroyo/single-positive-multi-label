import matplotlib.pyplot as plt


test_map = [64.87, 68.92, 69.89, 68.95, 63.83]
avg_labels_per_img = [8.8, 4, 2.7, 2, 1.3]
threshold = [0.55, 0.65, 0.75, 0.85, 0.95]

plt.plot(threshold, test_map, marker='D', color='#FF8C00')
plt.plot(threshold, [70.96]*5, '--')
plt.plot(threshold, [62.3]*5, '-.', color='r')
plt.ylabel("Test MAP COCO")
plt.title("Model Performance")
plt.xlabel("Threshold TeacherNet Predictions")
plt.legend(["StudentNet", "TeacherNet (EM loss)", "Traditional SPML w/ AN loss"])
plt.xticks(threshold)
plt.savefig('pseudo_multi_label_thresh.png')
plt.clf()

plt.plot(threshold, avg_labels_per_img, marker='^', color='g')
plt.plot(threshold, [2.9]*5, '--')
plt.ylabel("Avg Positive Labels per Example")
plt.xlabel("Threshold TeacherNet Predictions")
plt.xticks(threshold)
plt.legend(["TeacherNet Labels", "Ground Truth"])
plt.savefig('pseudo_multi_label_avg_positives.png')
plt.clf()

# fig, ax1 = plt.subplots()
# ax1.plot(threshold, test_map, marker='D')
# ax1.set_xlabel("Threshold TeacherNet Predictions")
# ax1.set_ylabel("Test MAP COCO")
# ax1.set_xticks(threshold)
# ax2 = ax1.twinx()
# ax2.plot(threshold, avg_labels_per_img, marker='^')
# ax2.set_xlabel("Threshold TeacherNet Predictions")
# ax2.set_ylabel("Avg Positive Labels per Example")
# plt.savefig('pseudo_multi_label_performance_and_supervision.png')
