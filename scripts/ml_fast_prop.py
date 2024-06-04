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

import getml
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cora.helpers as helpers

getml.engine.launch()
getml.engine.set_project("cora")

accuracy_per_ansatz = {}
d_prob_vecs = {}

num_features = None
show_plots = True

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

helpers.set_roles_on_dataframes(paper=paper, cites=cites, content=content)

view_paper_multi_varget = getml.data.make_target_columns(paper, "class_label")

# define build the datamodel
dm = helpers.define_datamodel(
    view_paper=view_paper_multi_varget, paper=view_paper_multi_varget, cites=cites, content=content
)

split = getml.data.split.random(train=0.7, test=0.3)

# first set the population table
container = getml.data.Container(population=view_paper_multi_varget, split=split)

# then add the peripheral tables needed for the data model
container.add(cites=cites, content=content, paper=paper)
container.freeze()

# let us check if the data model is compatible with the data container
pipe_check = getml.pipeline.Pipeline(
    data_model=dm,
)

pipe_check.check(container.train)

if num_features is None:
    fast_prop = getml.feature_learning.FastProp(
        loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss, num_threads=1
    )
else:
    fast_prop = getml.feature_learning.FastProp(
        loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss, num_threads=1, num_features=num_features
    )

feature_learners = [fast_prop]


preprocessors = []
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
"""
############################################################################################################
# getml objective: binary:logistic
############################################################################################################
"""
If we choose as objective binary:logistic, the probabilities that the row belongs to 
a specific class label (an not anything else), are combined by getml into a single vector,
here named prob_vecs_1_vs_all_test.
"""

pipe = getml.pipeline.Pipeline(
    tags=["fast_prop_mapping"],
    preprocessors=preprocessors,
    data_model=dm,
    feature_learners=feature_learners,
    predictors=[getml.predictors.XGBoostClassifier(objective="binary:logistic")],
)

pipe.fit(container.train)
pipe.score(container.test)
pipe.scores.filter(lambda score: score.set_used == "test")
prob_vecs_1_vs_all_test = pipe.predict(container.test)

if prob_vecs_1_vs_all_test[prob_vecs_1_vs_all_test < 0].size > 0:
    raise ValueError("There are negative values in the probability vectors. This should not happen.")

if prob_vecs_1_vs_all_test[prob_vecs_1_vs_all_test > 1.0].size > 0:
    raise ValueError("There are values larger 1 in the probability vectors. This should not happen.")


d_prob_vecs["getml_prob_vecs"] = prob_vecs_1_vs_all_test

"""
Let us call the probability vector for a single row as v = [v_1, v_2, ..., v_L].
Note that sum(v) != 1 in general.
"""
helpers.plot_dists_all_values(
    np.sum(prob_vecs_1_vs_all_test, axis=1), "1 vs all prob vectors sum(v) != 1 in general", show=show_plots
)

predicted_labels_test = helpers.probs_1_vs_all_to_label_via_argmax(
    prob_vecs_1_vs_all_test, class_labels=paper.class_label.unique()
)

gt_labels_test = paper[split == "test"].class_label.to_numpy()
accuracy = (gt_labels_test == predicted_labels_test).sum() / len(gt_labels_test)
accuracy_per_ansatz["L_one_vs_all_getml"] = accuracy

print("Share of accurately predicted class labels getml, binary:logistic:")
print(accuracy)

"""
To probabilities by normalization of each probability vector
"""

probs_simple_norm = np.asarray(
    [prob_vecs_1_vs_all_test[i] / np.sum(prob_vecs_1_vs_all_test[i]) for i in range(len(prob_vecs_1_vs_all_test))]
)
helpers.plot_dists_all_values(probs_simple_norm, "Dist norm(v)", show=show_plots)

d_prob_vecs["getml_prob_vecs_norm_simple"] = probs_simple_norm

# %%
############################################################################################################
# getml objective: binary:logitraw
############################################################################################################
"""
If we choose as objective binary:logistic, the probabilities that the row belongs to 
a specific class label (an not anything else), are combined by getml into a single vector,
here named prob_vecs_1_vs_all_test.
"""

pipe = getml.pipeline.Pipeline(
    tags=["fast_prop_mapping"],
    preprocessors=preprocessors,
    data_model=dm,
    feature_learners=feature_learners,
    predictors=[getml.predictors.XGBoostClassifier(objective="binary:logitraw")],
)

pipe.fit(container.train)
pipe.score(container.test)
pipe.scores.filter(lambda score: score.set_used == "test")
logit_vecs_1_vs_all_test = pipe.predict(container.test)

helpers.plot_dists_all_values(np.sum(logit_vecs_1_vs_all_test, axis=1), "logit_vecs_1_vs_all_test", show=show_plots)

prob_softmax_from_logit = helpers.softmax(logit_vecs_1_vs_all_test)
helpers.plot_dists_all_values(prob_softmax_from_logit, "getml_prob_softmax_from_logit", show=show_plots)

d_prob_vecs["getml_prob_softmax_from_logit"] = prob_softmax_from_logit

predicted_labels_test = helpers.probs_1_vs_all_to_label_via_argmax(
    prob_softmax_from_logit, class_labels=paper.class_label.unique()
)

accuracy = (gt_labels_test == predicted_labels_test).sum() / len(gt_labels_test)
accuracy_per_ansatz["L_one_vs_all_getml_logit_softmax"] = accuracy

print("Share of accurately predicted class labels getml:")
print(accuracy)


# %%

# get the feeature table
X = pipe.transform(
    population_table=view_paper_multi_varget, peripheral_tables={"cites": cites, "content": content, "paper": paper}
)

if (X.shape[1] != num_features) and num_features is not None:
    raise ValueError("The number of features must be the same as specified for fast_prop.")

print("X.shape", X.shape)


############################################################################################################
# XGBOOST objective: multi:softmax
############################################################################################################
# %%

y_categorical_strings = paper.class_label.to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_categorical_strings)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    "objective": "multi:softmax",  # Use 'multi:softprob' for probabilities
    "num_class": 7,  # Number of classes in the dataset
    "max_depth": 3,
}

bst = xgb.train(params, dtrain, num_boost_round=20)

# Make predictions
y_pred = bst.predict(dtest)
helpers.plot_dists_all_values(y_pred, "xgboost accuracy dist", show=show_plots)

# Accuracy manualy calculated
accuracy = (y_test == y_pred).sum() / len(y_pred)

accuracy_per_ansatz["xgboost_multi_softmax"] = accuracy

print("Share of accurately predicted class labels xgboost:")
print(accuracy)

# accuracy via sklearn accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy sklearn accuracy_score: {accuracy:.2f}")

# Print classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# Comparison of the two methods
print(accuracy_per_ansatz)

############################################################################################################
# XGBOOST objective: multi:softmax
############################################################################################################
# %%
y_categorical_strings = paper.class_label.to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_categorical_strings)

categorical_mapping = dict(zip(label_encoder.classes_, y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    "objective": "multi:softprob",  # Use 'multi:softprob' for probabilities
    "num_class": 7,  # Number of classes in the dataset
    "max_depth": 3,
}

bst = xgb.train(params, dtrain, num_boost_round=20)

# Make predictions
y_pred_probs = bst.predict(dtest)

print("max value sum(v):", np.sum(y_pred_probs, axis=1).max())
print("min value sum(v):", np.sum(y_pred_probs, axis=1).min())
print("ok... their probabilities sum to 1...")

helpers.plot_dists_all_values(y_pred_probs, "xgboost predictions dist ", show=show_plots)

d_prob_vecs["xgboost_prob_vecs_norm"] = y_pred_probs


# %%

y_max = 100
sel = ["getml_prob_vecs_norm_simple", "getml_prob_softmax_from_logit", "xgboost_prob_vecs_norm"]

for key, values in d_prob_vecs.items():
    if key in sel:
        plt.hist(values.flatten(), bins=300, edgecolor="black", label=key, alpha=0.5)

plt.ylim(0, y_max)
plt.xlim(0.9, 1)
plt.legend()
plt.title("prob vectors")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

for key, values in d_prob_vecs.items():
    if key in sel:
        plt.hist(values.flatten(), bins=300, edgecolor="black", label=key, alpha=0.5)

plt.ylim(0, y_max * 35)
plt.xlim(0, 0.05)
plt.legend()
plt.title("prob vectors")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

for key, values in d_prob_vecs.items():
    if key in sel:
        plt.hist(values.flatten(), bins=300, edgecolor="black", label=key, alpha=0.5)

plt.legend()
plt.title("prob vectors")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
