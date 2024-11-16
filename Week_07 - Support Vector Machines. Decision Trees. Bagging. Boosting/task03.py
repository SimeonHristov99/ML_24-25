import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def make_meshgrid(x, y, h=.02, lims=None):
    """
    Create a mesh of points to plot in.

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    if lims is None:
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
    else:
        x_min, x_max, y_min, y_max = lims
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, proba=False, **params):
    """
    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    if proba:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, -1]
        Z = Z.reshape(xx.shape)
        out = ax.imshow(Z,
                        extent=(np.min(xx), np.max(xx), np.min(yy),
                                np.max(yy)),
                        origin='lower',
                        vmin=0,
                        vmax=1,
                        **params)
        ax.contour(xx, yy, Z, levels=[0.5])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_classifier(X,
                    y,
                    clf,
                    ax=None,
                    ticks=False,
                    proba=False,
                    lims=None):  # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    cs = plot_contours(ax, clf, xx, yy, alpha=0.8, proba=proba)

    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y == labels[0]],
                   X1[y == labels[0]],
                   s=60,
                   c='b',
                   marker='o',
                   edgecolors='k')
        ax.scatter(X0[y == labels[1]],
                   X1[y == labels[1]],
                   s=60,
                   c='r',
                   marker='^',
                   edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())


def main():
    X = np.array([[1.7886284734303186, 0.43650985051198943],
                  [0.09649746807200862, -1.8634927033644908],
                  [-0.27738820251439905, -0.35475897926898675],
                  [-3.0827414814824596, 2.3729993231761526],
                  [-3.0438181689759283, 2.5227819696404974],
                  [-1.3138647533626822, 0.8846223804995846],
                  [-2.1186819577924703, 4.7095730636529485],
                  [-2.94996635782314, 2.595322585399109],
                  [-3.5453599476195303, 1.4535226844170317],
                  [0.9823674342581601, -1.1010676301114757],
                  [-1.1850465270201729, -0.20564989942254108],
                  [-1.5138516449254098, 3.2367162672269125],
                  [-4.023785139926468, 2.2870067998879504],
                  [0.6252449661628293, -0.1605133631869239],
                  [-3.76883635031923, 2.769969277722061],
                  [0.7450562664053708, 1.9761107831263025],
                  [-1.244123328955937, -0.6264169111883692],
                  [-0.8037660945765764, -2.4190831731786697],
                  [-0.9237920216957886, -1.0238757608428377],
                  [1.1239779589574683, -0.1319142328009009]])
    y = np.array([
        -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1
    ])

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    model.coef_ = np.array([[0, 1]])
    model.intercept_ = np.array([0])

    plot_classifier(X, y, model)
    plt.tight_layout()
    plt.show()

    num_err = np.sum(y != model.predict(X))
    print("Number of errors:", num_err)


if __name__ == '__main__':
    main()
