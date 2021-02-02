from glob import glob
import pickle
import matplotlib.pyplot as plt


def plot_one_graph(lst, pths, c_arr, t):
    plt.figure(figsize=(7.5, 10))
    for arr, pth, c in zip(lst, pths, c_arr):
        plt.plot(arr, color=c, label=pth.split('/')[-2])
        if len(arr) > 20:
            plt.scatter(19, arr[-1], color=c)
            plt.annotate("2_ep{}".format(len(arr)), (19, arr[-1]))
    plt.xlim([0,19])
    plt.title(t)
    plt.legend()
    plt.savefig('plots/one_{}.jpg'.format(t))
    plt.show()


def plot_two_graph(lst, v_lst, pths, c_arr, v_c_arr, t):
    plt.figure(figsize=(7.5, 10))
    for arr, v_arr, pth, c, v_c in zip(lst, v_lst, pths, c_arr, v_c_arr):
        plt.plot(arr, color=c, label=pth.split('/')[-2])
        plt.plot(v_arr, color=v_c[0], linestyle=v_c[1], label=pth.split('/')[-2]+"_val")
        if len(arr) > 20:
            plt.scatter(19, arr[-1], color=c)
            plt.scatter(19, v_arr[-1], color=c)
            plt.annotate("2_ep{}".format(len(arr)), (19, arr[-1]))
            plt.annotate("val_2_ep{}".format(len(arr)), (19, v_arr[-1]))
    plt.xlim([0, 19])
    plt.title(t)
    plt.legend()
    plt.savefig('plots/two_{}.jpg'.format(t))
    plt.show()

# Retrieve file names
pths = sorted(glob("models/*/trainHistoryDict.pickle"))
print(pths)

# retrieve list
acc = []
v_acc = []
loss = []
v_loss = []
fbeta = []
v_fbeta = []

line = ['b', 'k', 'r', 'darkorange', 'g']
v_line = [('b', '--'), ('k', '--'), ('r', '--'), ('darkorange', '--'), ('g', '--')]

for pth in pths:
    with open(pth, 'rb') as f:
        tmp = pickle.load(f)
        acc.append(tmp['acc'])
        v_acc.append(tmp['val_acc'])
        loss.append(tmp['loss'])
        v_loss.append(tmp['val_loss'])
        fbeta.append(tmp['fbeta'])
        v_fbeta.append(tmp['val_fbeta'])

# Accuracy
plot_one_graph(acc, pths, line, "Training Accuracy")
plot_one_graph(v_acc, pths, line, "Validation Accuracy")
plot_two_graph(acc, v_acc, pths, line, v_line, "Accuracy")

# Loss
plot_one_graph(loss, pths, line, "Training Loss")
plot_one_graph(v_loss, pths, line, "Validation Loss")
plot_two_graph(loss, v_loss, pths, line, v_line, "Loss")

# Fbeta
plot_one_graph(fbeta, pths, line, "Training fbeta")
plot_one_graph(v_fbeta, pths, line, "Validation fbeta")
plot_two_graph(fbeta, v_fbeta, pths, line, v_line, "fbeta")
