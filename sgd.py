import autograd.numpy as np
import dataset
import matplotlib.pyplot as plt
from autograd import grad
import imageio as io
import os

class SGD():
    def __init__(self, x, y, xtest=None, ytest=None):
        self.xtrain = x
        self.ytrain = y
        self.xtest = xtest
        self.ytest = ytest
        self.n = y.shape[0]
        self.gifnames = []
        self.savegif = True
        if self.xtest is not None and self.ytest is not None:
            self.eval = True
        else:
            self.eval = False

        print(f'Difference between gradients for some different theta:')
        theta = [(-1.0,-1.0,-1.0), (0.0,0.0,0.0), (1.0,1.0,1.0)]
        for th in theta:
            print(f'theta: {th}, ' \
                f'diff: {self.compare_grads(x, y, np.array(th))}')

    def train(self, theta=None, alpha=1e-2, epochs=1000, eps=0,
              alpha_dec=False, viz=True):
        if not os.path.exists('figs'):
            os.makedirs('figs')
        self.gifnames = []
        plt.figure()
        if theta is None:
            theta = np.array([1.0,1.0,1.0])
        test_error = []
        train_error = []
        deltas = []
        prev_err = 0
        if alpha_dec:
                alpha_orig = alpha
        for e in range(epochs):
            if alpha_dec:
                alpha = alpha_orig / np.sqrt(e+1)
            indices = np.arange(self.n)

            while len(indices) > 0:
                i = np.random.choice(indices)
                indices = np.delete(indices, indices==i)
                x = self.xtrain[:, i]
                y = self.ytrain[i]

                theta = theta - alpha * self.gradient(x, y, theta)

            err = self.error(self.xtrain, self.ytrain, theta)
            delta = abs(prev_err - err)
            if viz:
                self.viz(theta, alpha, e)
            if self.eval:
                train_error.append(err)
                test_error.append(self.error(self.xtest, self.ytest, theta))
                deltas.append(delta)

            if delta < eps:
                break
            prev_err = err

        if viz:
            fname = f'figs/boundary-{alpha:.3f}.eps' if not alpha_dec \
                else 'figs/boundary-dec.eps'
            plt.savefig(fname, format='eps')

        if self.eval:
            plt.figure()
            plt.semilogy(train_error, label='Training error')
            plt.semilogy(test_error, label='Test error')
            if alpha_dec:
                plt.title(r'Errors throughout training, decreasing $\alpha$',
                          fontsize=18)
            else:
                plt.title(r'Errors throughout training, $\alpha = $' \
                    f'{alpha:.3f}', fontsize=18)
            plt.xlabel('Epochs', fontsize=16)
            plt.legend(fontsize=16, loc=1)
            fn = f'figs/errors-{alpha:.3f}.eps' if not alpha_dec \
                else 'figs/errors-dec.eps'
            plt.savefig(fn, format='eps')
            plt.figure()
            plt.semilogy(deltas)
            plt.xlabel('Epochs', fontsize=16)
            if alpha_dec:
                plt.title(r'Training: |$error_{k-1} - error_k$|, ' \
                    r'decreasing $\alpha$', fontsize=18)
            else:
                plt.title(r'Training: |$error_{k-1} - error_k$|, $\alpha = $' \
                    f'{alpha:.3f}', fontsize=18)
            fn = f'figs/deltas-{alpha:.3f}.eps' if not alpha_dec \
                else 'figs/deltas-dec.eps'
            plt.savefig(fn, format='eps')

        if self.savegif and viz:
            fname = f'boundary-{alpha}.gif' if not alpha_dec \
                else 'boundary-dec.gif'
            with io.get_writer(fname, mode='I', duration=0.05) as writer:
                for pic in self.gifnames:
                    im = io.imread(pic)
                    writer.append_data(im)
                    os.remove(pic)

        plt.show(block=False)

    def get_plot_boundary(self, theta):
        x1range = np.array([min(self.xtrain[0,:]) - 0.1,
                            max(self.xtrain[0,:] + 0.1)])
        x2min = min(self.xtrain[1,:]) - 0.1

        return x1range, -theta[0]/theta[1] * x1range - theta[2]/theta[1], x2min

    def viz(self, theta, alpha, epoch):
        plt.clf()
        x, y, x2min = self.get_plot_boundary(theta)
        plt.plot(x, y)
        plt.fill_between(x, y, x2min)
        plt.scatter(self.xtrain[0, self.ytrain==1],
                    self.xtrain[1, self.ytrain==1],
                    label='Training data, class 1')
        plt.scatter(self.xtrain[0, self.ytrain==-1],
                    self.xtrain[1, self.ytrain==-1],
                    label='Training data, class -1')
        if self.eval:
            plt.scatter(self.xtest[0, self.ytest==1],
                        self.xtest[1, self.ytest==1],
                        label='Test data, class 1')
            plt.scatter(self.xtest[0, self.ytest==-1],
                        self.xtest[1, self.ytest==-1],
                        label='Test data, class -1')

        plt.title(r'Decision Boundary, $\alpha = $' f'{alpha:.3f}',
                  fontsize=18)
        plt.legend(fontsize=12, loc=1)
        plt.show(block=False)
        if self.savegif:
            fn = f'figs/{alpha:.3f}-{epoch}.png'
            self.gifnames.append(fn)
            plt.savefig(fn)
        plt.pause(0.0001)

    def error(self, x, y, theta):
        self.x = x
        self.y = y
        return self._error(theta)

    def compare_grads(self, x, y, theta):
        self.x = x
        self.y = y
        auto_diff = grad(self._error)
        analytical = self.gradient(x, y, theta)
        auto = auto_diff(theta)

        return np.linalg.norm(analytical - auto)

    def _error(self, theta):
        xhat = np.append(self.x, np.ones([1, self.x.shape[1]]), axis=0)
        return np.mean(np.log(1 + np.exp(-self.y*np.matmul(xhat.T, theta))))

    def gradient(self, x, y, theta):
        if len(x.shape) > 1:
            xhat = np.append(x, np.ones([1, x.shape[1]]), axis=0)
        else:
            xhat = np.append(x, 1)
        if len(x.shape) > 1:
            return -np.mean(y*xhat * np.exp(-y*np.matmul(xhat.T, theta)) / \
            (1 + np.exp(-y*np.matmul(xhat.T, theta))), axis=1)
        else:
            return -y*xhat * np.exp(-y*np.matmul(xhat.T, theta)) / \
                (1 + np.exp(-y*np.matmul(xhat.T, theta)))


x1, y1, x2, y2 = dataset.generate_halfmoon(100, 100)
x = np.append(x1, x2, axis=1)
y = np.append(y1, y2, axis=0)

x1, y1, x2, y2 = dataset.generate_halfmoon(100, 100)
xt = np.append(x1, x2, axis=1)
yt = np.append(y1, y2, axis=0)

sgd = SGD(x, y, xtest=xt, ytest=yt)
alpha = [0.001, 0.01, 0.05, 0.1, 1.0]
epochs = 300

viz = True
theta = np.array([0.0,0.0,0.0])
for a in alpha:
    sgd.train(theta=theta, alpha=a, epochs=epochs, viz=viz)

sgd.train(theta=theta, alpha=0.05, epochs=epochs, alpha_dec=True, viz=viz)
plt.show()
