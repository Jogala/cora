# %%
""""
Multiclass classification problem:
- The goal is to predict seven different class_label.
- Call the number off different class labels L

Labels:
    'Case_Based','Genetic_Algorithms','Neural_Networks','Probabilistic_Methods',
    'Reinforcement_Learning','Rule_Learning','Theory'

We generate a target column for each of the class labels.
Each column encodes in a boolean array if the row belongs to the specific class label.

1 column -> L columns, 1 for each class_label

Note that make_target_columns returns a new view on the data, hence no data is copied.
"""
from typing import Literal

import getml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cora.helpers as helpers

getml.engine.launch()
getml.engine.set_project("cora_all_test")

conn = getml.database.connect_mysql(
    host="db.relational-data.org",
    dbname="CORA",
    port=3306,
    user="guest",
    password="relational",
)

paper = helpers.load_if_needed(conn, "paper")
cites = helpers.load_if_needed(conn, "cites")
content = helpers.load_if_needed(conn, "content")

with open("zuordnung.txt", "r") as f:
    zuordnung = f.read()
    zuordnung = eval(zuordnung)

df_zu = pd.DataFrame(data=zuordnung, columns=["paper_id", "row_num"])
df_zu["paper_id"] = df_zu["paper_id"].astype(int)
df_paper = paper.to_pandas()
df_paper["paper_id"] = df_paper["paper_id"].astype(int)
df_paper = df_paper.merge(df_zu, on="paper_id").sort_values("row_num")

paper_train = getml.data.DataFrame.from_pandas(df_paper[:1708], name="paper_train")
paper_val = getml.data.DataFrame.from_pandas(df_paper[1708 : 1708 + 500], name="validation")
paper_test = getml.data.DataFrame.from_pandas(df_paper[1708 + 500 : 1708 + 1000], name="paper_test")
paper, split = getml.data.split.concat("population", train=paper_train, validation=paper_val, test=paper_test)

helpers.set_roles_on_dataframes(paper=paper, cites=cites, content=content)

view_paper_multi_varget = getml.data.make_target_columns(paper, "class_label")

container = getml.data.Container(population=view_paper_multi_varget, split=split)
container.add(cites=cites, content=content, paper=paper)
container.freeze()


dm = helpers.define_datamodel(
    view_paper=view_paper_multi_varget, paper=view_paper_multi_varget, cites=cites, content=content
)

rel_boost = getml.feature_learning.Relboost(
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss, num_threads=10, num_features=50
)

fast_prop = getml.feature_learning.FastProp(
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
    num_threads=10,
    aggregation=getml.feature_learning.aggregations.fastprop.Minimal,
    num_features=595,
)

multi_rel = getml.feature_learning.Multirel(
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
    num_threads=10,
    num_features=50,
)

fast_boost = getml.feature_learning.Fastboost(
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
    num_threads=10,
    num_features=100,
)

rel_boost = getml.feature_learning.Relboost(
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
    num_threads=10,
    num_features=100,
)


feature_learners = [fast_prop]

"""
The mapping preprocessors has a huge impact on this problem, 
it does the target encoding of the catgeorical variables for you.
However, it is only available in the enterpise version.
Comment the line, if you do not have access to the enterprise version.
"""
preprocessors = [getml.preprocessors.Mapping()]

"""
Under the hood, there is a feature table build for each label separately.
Then a xgboost binary classifier (binary:logistic) is trained for each target column, 
hence L times a 1 vs all model is used. 

We sue getml objective: binary:logistic, generating Features for each class label with relboost
and with fastprop one set of features for all class labels.

If we choose as objective binary:logistic, the probabilities that the row belongs to 
a specific class label (an not anything else), are combined by getml into a single vector,
here named prob_vecs_1_vs_all_test.
"""


getml_accuracy = []
multi_softmax_accuracy = []
multi_softmax_classification_report = []
pipe = getml.pipeline.Pipeline(
    tags=["fast_prop_mapping"],
    preprocessors=preprocessors,
    data_model=dm,
    feature_learners=feature_learners,
    predictors=[getml.predictors.XGBoostClassifier(objective="binary:logistic")],
)

pipe.check(container.train)
pipe.fit(container.train)

prob_vecs_1_vs_all_test = pipe.predict(container.test)

predicted_labels_test = helpers.probs_1_vs_all_to_label_via_argmax(
    prob_vecs_1_vs_all_test, class_labels=paper.class_label.unique()
)
gt_labels_test = paper[split == "test"].class_label.to_numpy()
accuracy = (gt_labels_test == predicted_labels_test).sum() / len(gt_labels_test)
getml_accuracy.append(accuracy)

############################################################################################################
# XGBOOST objective: multi:softmax
############################################################################################################


def encode_cat(paper, test_train: Literal["test", "train"]):
    y = paper[split == test_train].class_label.to_numpy()
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder


X_test = pipe.transform(container.test)
X_train = pipe.transform(container.train)
y_test, label_encoder_test = encode_cat(paper, "test")
y_train, label_encoder_train = encode_cat(paper, "train")

labels = list(set(label_encoder_test.classes_ + label_encoder_train.classes_))

print("X.shape", X_test.shape)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "multi:softmax",  # Use 'multi:softprob' for probabilities
    "num_class": 7,  # Number of classes in the dataset
    "max_depth": 3,
}

bst = xgb.train(params, dtrain, num_boost_round=20)

y_pred = bst.predict(dtest)
accuracy = (y_test == y_pred).sum() / len(y_pred)
multi_softmax_accuracy.append(accuracy)

report = classification_report(y_test, y_pred, target_names=labels)
multi_softmax_classification_report.append(report)


# Comparison of the two methods
def print_results(name, accuracies):
    print(
        name,
        " accuracy: mean: ",
        f"{np.mean(accuracies):.3f}",
        "max: ",
        f"{np.max(accuracies):.3f}",
        "median: ",
        f"{np.median(accuracies):.3f}",
        "std: ",
        f"{np.std(accuracies):.3f}",
    )


print_results("getml  ", getml_accuracy)
print_results("xgboost", multi_softmax_accuracy)
