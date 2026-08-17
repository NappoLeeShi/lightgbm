from preprocessing import *
from boosting import *
from metrics import *


# =====================
# LOAD DATA
# =====================
df = load_data(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# =====================
# FEATURE / TARGET
# =====================
X = df.drop(
    "Churn",
    axis=1
).values

y = df["Churn"].values


# =====================
# TRAIN TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================
# MODEL
# =====================
model = GradientBoostingLightGBM(
    n_estimators=30,
    learning_rate=0.1,
    max_depth=3
)


# =====================
# TRAIN
# =====================
model.fit(
    X_train,
    y_train
)


# =====================
# PREDICT
# =====================
pred = model.predict(
    X_test
)


# =====================
# EVALUATION
# =====================
acc = accuracy(
    y_test,
    pred
)

tp, tn, fp, fn = confusion_matrix(
    y_test,
    pred
)

precision = tp / (tp + fp + 1e-15)

recall = tp / (tp + fn + 1e-15)

f1 = (
    2 * precision * recall
) / (
    precision + recall + 1e-15
)


# =====================
# RESULT
# =====================
print("\n===== RESULT =====")

print(
    f"Accuracy : {acc:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\n===== CONFUSION MATRIX =====")

print(
    f"TP = {tp}"
)

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)