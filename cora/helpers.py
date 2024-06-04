# %%
import getml
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn import metrics

FloatMatrix = npt.NDArray[np.float64]
IntMatrix = npt.NDArray[np.int64]
Table = getml.data.DataFrame | getml.data.View


def load_if_needed(conn: getml.database.Connection, name: str) -> getml.DataFrame:
    """
    Loads the data from the relational learning
    repository, if the data frame has not already
    been loaded.
    """
    if not getml.data.exists(name):
        data_frame = getml.data.DataFrame.from_db(name=name, table_name=name, conn=conn)
        data_frame.save()
    else:
        data_frame = getml.data.load_data_frame(name)

    return data_frame


def define_datamodel(view_paper: Table, paper: Table, cites: Table, content: Table):
    """
    actually, we could also use paper.to_placeholder,
    as only the join keys matter for the data model.
    For the sake of clarity, how the abstract datamodel item correpsond to the actual tables,
    we use view_paper.to_placeholder
    """
    dm = getml.data.DataModel(population=view_paper.to_placeholder("population"))

    """
    declare the peripheral tables of the datamodel, cites[0], cites[1], content, and paper.
    """
    dm.add(getml.data.to_placeholder(cites=[cites] * 2, content=content, paper=paper))

    """
    Now that all tables are declared in the datamodel, we can join them
    """
    dm.population.join(dm.cites[0], on=("paper_id", "cited_paper_id"))

    dm.cites[0].join(dm.content, on=("citing_paper_id", "paper_id"))
    dm.cites[0].join(dm.paper, on=("citing_paper_id", "paper_id"), relationship=getml.data.relationship.many_to_one)

    dm.population.join(dm.cites[1], on=("paper_id", "citing_paper_id"))
    dm.cites[1].join(dm.content, on=("cited_paper_id", "paper_id"))
    dm.cites[1].join(dm.paper, on=("cited_paper_id", "paper_id"), relationship=getml.data.relationship.many_to_one)

    dm.population.join(dm.content, on="paper_id")

    return dm


def set_roles_on_dataframes(
    paper: getml.data.DataFrame, cites: getml.data.DataFrame, content: getml.data.DataFrame
) -> None:
    paper.set_role("paper_id", getml.data.roles.join_key)

    paper.set_role("class_label", getml.data.roles.categorical)

    cites.set_role(["cited_paper_id", "citing_paper_id"], getml.data.roles.join_key)

    content.set_role("paper_id", getml.data.roles.join_key)
    content.set_role("word_cited_id", getml.data.roles.categorical)


def probs_1_vs_all_to_label_via_argmax(
    probability_vectors_1_vs_all: FloatMatrix, class_labels: np.ndarray[float]
) -> np.ndarray[int]:
    """
    1. Argmax

    Simply return the class label with the highest value, hence effectively transforming
    v -> [0, 0, ..., 1, ..., 0] where the 1 is at the index of the highest value in v.
    """
    ix_max = np.argmax(probability_vectors_1_vs_all, axis=1)
    predicted_labels = np.asarray([class_labels[ix] for ix in ix_max])

    return predicted_labels


# %%
def softmax(vec: FloatMatrix) -> FloatMatrix:
    """
    stable softmax function, e^xi / sum_i e^xi -> e^(xi-max(x)) / sum_i e^(xi-max(x))
    """
    exps = np.exp(vec - np.max(vec, axis=1, keepdims=True))
    sum_exp_scores = np.sum(exps, axis=1, keepdims=True)
    probabilities = exps / sum_exp_scores
    return probabilities


def plot_dists_all_values(values: FloatMatrix, title: str, show: bool):
    # Plot the hist of the sum of the probability vectors
    plt.hist(values.flatten(), bins=50, edgecolor="black")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if show:
        plt.show()
    plt.close()


def calc_auc(gt: IntMatrix, pred: IntMatrix) -> float:
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, pos_label=2)
    return metrics.auc(fpr, tpr)
