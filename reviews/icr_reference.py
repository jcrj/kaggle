"""Leak-free reference pipeline for ICR (Identifying Age-Related Conditions).

A deliberately simple, classical-ML rewrite of ICR-Competition-Notebook.ipynb
demonstrating the fixes from ICR-code-review.md:

  1. All data-dependent preprocessing lives INSIDE the per-fold pipeline,
     so nothing is ever estimated from validation rows.
  2. class_weight="balanced" replaces RandomUnderSampler: same effect on the
     balanced-log-loss target (probabilities calibrated to a 50/50 prior),
     but no training rows are discarded.
  3. Shallow, regularized trees sized for ~550 training rows per fold.
  4. Every random component is explicitly seeded; the CV score is reported
     as mean +/- std across repeats, so you know your noise floor before
     comparing two ideas.
  5. The inference path asserts its assumptions instead of failing silently.

Intended to run in the Kaggle ICR environment. Requires: pandas, numpy,
scikit-learn, lightgbm.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

PATH = "/kaggle/input/icr-identify-age-related-conditions/"

MASTER_SEED = 42      # one seed to rule them all -> fully reproducible
N_REPEATS = 5         # repeated CV: the spread across repeats = noise estimate
N_FOLDS = 10


def balanced_log_loss(y_true, y_pred):
    """Exact competition metric: log loss with each class reweighted to 50%."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    loss_0 = -np.mean(np.log(1 - y_pred[y_true == 0]))
    loss_1 = -np.mean(np.log(y_pred[y_true == 1]))
    return 0.5 * (loss_0 + loss_1)


def load_features(df):
    """Fixed, data-independent feature prep — safe to apply outside CV.

    The rule: only steps with NO estimated parameters may happen before the
    split. Mapping EJ with a constant dict qualifies; fitting a scaler or
    imputer does not.
    """
    X = df.drop(columns=["Id"], errors="ignore")
    X = X.drop(columns=["Class"], errors="ignore")
    X["EJ"] = X["EJ"].map({"A": 0, "B": 1})
    return X.astype(float)


def make_model(seed):
    """One pipeline = preprocessing + model, fitted as a unit per fold.

    Trees are invariant to monotone transforms, so no log/Box-Cox/scaling
    is needed here. Median imputation is included for the few NaN columns
    (LightGBM also handles NaN natively; the imputer keeps the pipeline
    portable to other estimators).

    Capacity is sized for ~550 rows: few leaves, real regularization, a
    slow learning rate. class_weight="balanced" targets the 50/50 prior
    that the balanced log loss scores against.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=8,
            min_child_samples=20,
            colsample_bytree=0.6,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        ),
    )


def cross_validate(X, y):
    """Repeated stratified CV with out-of-fold predictions.

    Returns per-repeat OOF scores and the repeat-averaged OOF probabilities.
    The mean of `scores` estimates a single model's skill; the score of the
    averaged probabilities estimates the seed-bagged ensemble's skill. The
    std of `scores` is your noise floor: improvements smaller than it are
    not evidence.
    """
    rng = np.random.default_rng(MASTER_SEED)
    scores, oof_sum = [], np.zeros(len(y))

    for _ in range(N_REPEATS):
        seed = int(rng.integers(0, 2**31 - 1))
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for train_idx, val_idx in skf.split(X, y):
            pipe = make_model(seed)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(balanced_log_loss(y, oof))
        oof_sum += oof

    return np.array(scores), oof_sum / N_REPEATS


def main():
    train = pd.read_csv(PATH + "train.csv")
    test = pd.read_csv(PATH + "test.csv")

    X, y = load_features(train), train["Class"]

    scores, oof_avg = cross_validate(X, y)
    print(f"CV balanced log loss: {scores.mean():.4f} +/- {scores.std():.4f} "
          f"(over {N_REPEATS} repeats of {N_FOLDS}-fold)")
    print(f"Seed-averaged OOF score: {balanced_log_loss(y, oof_avg):.4f}")

    # Final model: refit on ALL training data (the CV models existed only to
    # measure skill). A small seed bag keeps the variance reduction that the
    # original notebook's 200-model ensemble was buying, at 2.5% of the cost.
    X_test = load_features(test)
    test_proba = np.zeros(len(X_test))
    for seed in range(5):
        pipe = make_model(MASTER_SEED + seed)
        pipe.fit(X, y)
        test_proba += pipe.predict_proba(X_test)[:, 1]
    test_proba /= 5

    # Fail loudly, not silently: a bad rerun should crash, not score garbage.
    assert np.isfinite(test_proba).all(), "non-finite test predictions"
    assert ((test_proba >= 0) & (test_proba <= 1)).all()

    submission = pd.DataFrame({
        "Id": test["Id"],
        "class_0": 1 - test_proba,
        "class_1": test_proba,
    })
    submission.to_csv("submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
