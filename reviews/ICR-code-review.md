# Code review: ICR-Competition-Notebook.ipynb (159th place, LB 0.39)

A retrospective review focused on the mistakes that matter most, why they matter,
and the classical ML / statistics concepts behind each fix. Companion reference
implementation: [`icr_reference.py`](icr_reference.py).

Scoreboard context: CV said **0.228**, the private leaderboard said **0.39**.
That gap is not bad luck — most of it is explainable, and the explanations below
are ranked by how much each one contributed.

---

## 1. Preprocessing leakage: transforms fitted before the CV split (the big one)

```python
X_preliminary = preliminary_preprocess.fit_transform(train.drop(["Class", 'Id'], axis=1))
...
for fold, (train_idx, val_idx) in enumerate(skf.split(X_preliminary, y)):
```

The entire preprocessing pipeline — `StandardScaler`, `PowerTransformer`
(Box-Cox/Yeo-Johnson lambdas), and especially `KNNImputer` — is fitted on **all
617 rows** before cross-validation begins. Every fold's "validation" rows have
already influenced the scaler means, the power-transform lambdas, and (worst)
the KNN-imputed values: with `KNNImputer`, actual feature values from validation
rows are copied into training rows as imputations.

**The concept.** Cross-validation simulates the real situation: "I have training
data, and tomorrow new data arrives that I have never seen." Anything *estimated
from data* — a mean, a lambda, a set of nearest neighbours, a feature selection,
a hyperparameter — is part of the model. If any of it saw the validation rows,
the simulation is broken and the CV score is biased optimistic. The rule is
mechanical, not a judgment call:

> **Everything estimated from data must be re-estimated inside each fold,
> using only that fold's training rows.**

This kind of leakage (no target involved) is "soft" — it inflates CV by a
modest amount rather than wrecking it — but on 617 rows with a KNN imputer it
is not negligible, and it compounds with issue 2.

**The fix** is structural and actually *less* code: put preprocessing inside the
pipeline that gets fitted per fold, and pass **raw** data to `skf.split`:

```python
pipe = make_pipeline(preliminary_preprocess, estimator)   # one object
for train_idx, val_idx in skf.split(X_raw, y):
    pipe.fit(X_raw.iloc[train_idx], y.iloc[train_idx])    # transforms fit here only
    oof[val_idx] = pipe.predict_proba(X_raw.iloc[val_idx])[:, 1]
```

This is exactly what sklearn's `Pipeline` exists for. Ironically the stale
comment in the submission cell — *"Each classifier contains preprocessing, so we
pass raw test dataset"* — describes the correct design that the code doesn't
implement.

Smaller instance of the same disease: the per-column transform *selection*
(the probplot R² "Winner" table) was also decided on the full dataset. It's
unsupervised, so the damage is small, but a purist would re-select per fold or
accept it as a fixed prior decision.

---

## 2. Tuning and scoring on the same CV: the winner's curse

The hyperparameters are clearly Optuna output
(`learning_rate=0.1656219205521477`), and they were tuned against the same
folds used to report the 0.228. That makes 0.228 the **maximum of many noisy
draws**, not an unbiased estimate.

**The concept (adaptive overfitting / winner's curse).** Each CV score is a
noisy measurement: true skill + noise. If you run 200 Optuna trials and pick
the best score, you have selected partly for skill and partly for *lucky
noise*. The expected value of the maximum of N noisy estimates exceeds the
true skill of the chosen configuration — and the smaller the dataset, the
bigger the noise, the bigger the bias. With 617 rows and a metric as twitchy
as balanced log loss, fold-level noise is huge, so the bias is huge.

The tuned parameters themselves confirm it: `max_depth=10/11`,
`n_estimators=582/684`, `learning_rate≈0.17`, regularization `1e-4`–`1e-5`
(i.e. none). After `RandomUnderSampler`, each fold trains on roughly
**2 × 97 ≈ 195 rows** — and you're fitting 600 trees of depth 10 to them.
Optuna found a configuration that memorizes the folds in a way that happened
to survive this particular CV. On 195 rows, sensible gradient-boosting
settings look like: `num_leaves` 4–16, heavy `min_child_samples`,
`reg_lambda ≥ 1`, learning rate ≤ 0.05.

**The fixes, in increasing rigor:**

1. **Sanity prior:** if tuning improves CV from, say, 0.30 to 0.23 on 617 rows,
   disbelieve it. Real hyperparameter gains on tabular data are usually small;
   30% improvements from tuning alone are almost always noise harvesting.
2. **Re-validate the chosen config on fresh splits:** after tuning, re-run CV
   with seeds Optuna never saw. The score will drop; *that* number is your
   estimate.
3. **Nested CV** (the textbook answer): an outer loop measures, an inner loop
   tunes. Expensive but honest — the outer folds never touch tuning.
4. **Report uncertainty:** run repeated CV (e.g. 5 repeats × 10 folds) and
   report mean ± std across repeats. If your std is ±0.04, stop celebrating
   differences of 0.01.

---

## 3. Most of the transform machinery couldn't help: trees are invariant to monotone transforms

The most elaborate part of the notebook — probplot R² over six candidate
transforms per column, then a five-branch `ColumnTransformer` — feeds an
ensemble whose weights are **0.45 LGBM + 0.45 XGB + 0.1 SVC**.

**The concept.** A decision tree never looks at the *values* of a feature, only
at their *order*: every split is "is x ≤ t?", and the chosen split partitions
the data identically under any strictly monotone transform (log, sqrt, Box-Cox,
reciprocal*, standardization). So for 90% of the ensemble's weight, Box-Cox vs
log vs nothing changes (almost) nothing — the only consumers of all that work
were the SVC (weight 0.1) and the KNN imputer's distance computations.
(*Reciprocal is order-reversing, still fine. Caveat: LightGBM's histogram
binning means results aren't bit-identical under transforms, but the effect is
noise-level.)

Normality-targeting transforms matter for models that use **distances, dot
products, or coefficients**: SVM, KNN, logistic/linear regression, anything
with an L2 penalty, PCA. They are wasted on tree ensembles.

**The takeaway** is not "never transform" but **match preprocessing effort to
the model class**. Two clean options:

- Trees-only pipeline: raw features, median imputation (or let LightGBM handle
  NaN natively), label-encode `EJ`. Done.
- Hybrid ensemble: keep two preprocessing branches — raw → trees,
  transformed+scaled → SVC/logistic — rather than forcing one pipeline on all.

Related sub-issue — **the `Binarizer` on semi-constant columns destroys
information**: for columns where ≥50% of values sit at the floor, everything
above the median collapses to 1, discarding the magnitudes in the informative
tail. A tree would happily find the floor/not-floor split *by itself* while
also using the tail values. If you want to help non-tree models, keep the value
**and add** an `is_at_floor` indicator instead of replacing the value.

---

## 4. Undersampling and the metric: you were right for an unexamined reason

`RandomUnderSampler` to 50/50 before fitting actually *matches* this
competition's metric — but it's worth understanding why, because under plain
log loss the same trick would have hurt.

**The concept (probability calibration vs. class priors).** Balanced log loss
is ordinary log loss computed under a reweighted distribution where each class
has 50% mass. The Bayes-optimal prediction for it is not P(y=1|x) under the
true 17.5/82.5 prior, but the posterior under a **50/50 prior**. Training on a
balanced sample (or with balanced class weights) makes a probabilistic
classifier estimate exactly that quantity. So undersampling pointed your
probabilities at the right target — likely one real reason the notebook held up
in the shakeup.

The general tool is the **prior-shift correction** (Bayes' rule, worth
memorizing). If a model is calibrated under training prior π₁ = P(y=1) and you
need probabilities under a new prior π′₁:

```
p′(1|x) = [p(1|x)·π′₁/π₁] / [p(1|x)·π′₁/π₁ + p(0|x)·π′₀/π₀]
```

With π′ = 0.5 this converts full-data probabilities to balanced-metric
probabilities — no rows discarded.

**What to improve:** undersampling throws away ~70% of the majority class per
fit (195 of 555 rows kept). Your 20-bag averaging partially recovers this
(that's essentially the EasyEnsemble idea — good instinct), but the simpler,
data-efficient route is `class_weight="balanced"` (LightGBM/sklearn) or
`scale_pos_weight` (XGBoost): every fit sees all rows, with minority errors
weighted up so probabilities target the balanced prior directly. One line, no
discarded data, no bags needed for that purpose.

---

## 5. Reproducibility: the unseeded RNG

```python
seeds = np.random.randint(0, 20000, size=N_BAGS)
```

Unseeded — every notebook run produces different bag seeds, so the reported
0.228 and the submitted predictions cannot be reproduced. In a competition this
is operationally dangerous: you cannot tell whether a score change came from a
code change or from seed luck, which makes A/B-ing your own ideas impossible.

Fix is one line: `rng = np.random.default_rng(42); seeds = rng.integers(0, 2**31 - 1, size=N_BAGS)`.
Rule: **every** stochastic component gets an explicit seed (splits, samplers,
models, numpy), and changing the seed deliberately is itself a useful
experiment — the spread across seeds estimates your score's noise floor.

---

## 6. Fragile inference path: the `1e-9` hack and ±inf

```python
if np.all(np.isclose(test.select_dtypes("number").sum(), 0)):
    test[test_numeric_cols] += 1e-9
```

This exists because the public placeholder test rows are all zeros, and
`log(0) = -inf`, `boxcox` requires strictly positive input, `reciprocal(0) = inf`.
But it only triggers when *every* numeric column sums to ~0 — i.e. only on the
placeholder. On the real hidden test, a single legitimate zero (or negative) in
any log/Box-Cox/reciprocal column produces ±inf that flows silently into
predictions. In a code competition, the rerun either crashes (submission error)
or scores garbage rows you never see.

**Fixes:**
- Prefer transforms with full-real-line domains: `np.log1p` instead of `log`,
  Yeo-Johnson instead of Box-Cox (Yeo-Johnson *is* Box-Cox extended to zero and
  negatives — and your R² table shows it essentially tied with Box-Cox on every
  winner column, so you lose nothing).
- Assert your assumptions at inference: `assert np.isfinite(X_test.values).all()`
  after transforming. A loud crash during testing beats a silent bad score.

---

## 7. Smaller code-quality issues

- **Stale comment** in the submission cell claims each classifier contains
  preprocessing; they don't (preprocessing was applied separately). Stale
  comments cause real bugs later — e.g. double-transforming test data.
- `y_test = np.zeros_like(test_ids)` — `test_ids` is an object-dtype string
  Series, so this allocates an **object** array of zeros that happens to
  support `+=`. Intent-revealing version: `np.zeros(len(test), dtype=float)`.
- Dead imports: `SMOTE`, `CatBoostClassifier`, `LogisticRegression`,
  `StackingClassifier`, `RandomForestClassifier`, `statistics.mean`,
  `f_classif`. Dead code makes a notebook read as "experiments in progress";
  prune before publishing.
- `defaultdict(object)` is a plain `dict` with extra steps. Also, keeping all
  **200 fitted voting pipelines** (~600 boosted models) in memory just to
  predict one test set is heavy; predict inside the CV loop and keep only the
  accumulated test probabilities.
- `greeks.csv` loaded, never used. Its `Epsilon` column (measurement date) was
  the key to the strongest validation schemes in this competition: the hidden
  test was *later in time* than train, so a time-ordered split estimates
  generalization more honestly than random K-fold. General lesson: when
  train/test are split by time, validate by time.

---

## 8. What the notebook gets right (keep doing these)

- **Implementing the exact competition metric locally** (with clipping) and
  validating against it, not a proxy like accuracy or AUC.
- **Stratified** K-fold on an imbalanced target.
- **Out-of-fold evaluation** of the full ensemble rather than fold-mean of
  per-model scores.
- **Seed bagging** to reduce variance — on 617 rows, variance reduction is
  exactly the right lever (20 bags is past diminishing returns; 5 buys most
  of it — measure the marginal gain next time).
- **Soft voting across diverse model families**, and down-weighting the SVC
  rather than equal-weighting.
- Resisting the public-LB chase. The shakeup rewarded exactly that.

---

## The one-paragraph version

Wrap *all* data-dependent preprocessing in the per-fold pipeline (leakage),
never quote a CV number that hyperparameter search was allowed to optimize
(winner's curse), match preprocessing to model class (trees don't care about
monotone transforms), prefer reweighting to discarding data (class weights over
undersampling — while understanding that the balanced metric is why balancing
helped at all), seed everything, and make the inference path crash loudly
rather than fail silently. With those fixed, a *much* simpler model — one
LightGBM with shallow trees, class weights, and repeated CV reported as
mean ± std — would have been competitive, reproducible, and trustworthy.
See `icr_reference.py` for that pipeline in ~120 lines.
