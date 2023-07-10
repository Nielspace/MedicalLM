from embeddings import embeddings

import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline

class model_viz:

    @staticmethod
    def model_comp_viz(df, x_feature, y_feature, model_name:list(), model_architecture:list()):
        isinstance(model_name, list)
        isinstance(model_architecture, list)
        assert len(model_name) == len(model_architecture), """Length of model name must be equal to 
                                                            lenght of model architecture. Ensure that 
                                                            both are lists."""
        figure = plt.figure(figsize=(8, 15))
        i = 1


        for ds_cnt, _ in enumerate(df):
            (X_train, X_test, y_train, y_test, 
            vectorizer,feature_names, pca, labels, category_list) = embeddings.embeddings_gen(df, 
                                                                                            x_feature, 
                                                                                            y_feature )
            x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
            y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])

            ax = plt.subplot(4, int((len(model_architecture)+1)/4), i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            for name, clf in zip(model_name, model_architecture):
                ax = plt.subplot(4, int((len(model_architecture)+1)/4), i)

                clf = make_pipeline(clf)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                DecisionBoundaryDisplay.from_estimator(
                    clf, X_train, cmap=cm, alpha=0.8, ax=ax, eps=0.5
                )

                # Plot the training points
                ax.scatter(
                    X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
                )
                # Plot the testing points
                ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    cmap=cm_bright,
                    edgecolors="k",
                    alpha=0.6,
                )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())
                ax.axis("on")

                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(
                    x_max - 0.3,
                    y_min + 0.3,
                    ("%.2f" % score).lstrip("0"),
                    size=15,
                    horizontalalignment="right",
                )

                if i == (len(model_architecture) + 1):
                    i = 1
                else:
                    i += 1

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_decision_boundary(model, X, y):
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        x1_values, x2_values = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                                        np.arange(x2_min, x2_max, 0.02))

        grid_points = np.c_[x1_values.ravel(), x2_values.ravel()]
        y_pred = model.predict(grid_points)

        y_pred = y_pred.reshape(x1_values.shape)
        plt.contourf(x1_values, x2_values, y_pred, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
        plt.axis("off")

        plt.show()
