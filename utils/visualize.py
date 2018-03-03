from matplotlib import pyplot as plt
import time


def draw_curves(epochs_all, epoch, train_loss, test_loss, train_accuracy, test_accuracy,
                save_name=time.strftime('%Y%m%d%H%M%S', time.localtime())):
    plt.ion()  # turn on the interactive mode
    plt.close()  # close the former figure and update it
    plt.subplot(121)
    plt.plot(epoch, train_loss, 'b')
    plt.plot(epoch, test_loss, 'r')
    plt.title('loss')
    plt.subplot(122)
    plt.plot(epoch, train_accuracy, 'b')
    plt.plot(epoch, test_accuracy, 'r')
    plt.title('accuracy')
    if len(epoch) == epochs_all:    # pause and save it
        plt.ioff()
        plt.savefig('save/' + save_name)
    plt.show()
    plt.pause(1)