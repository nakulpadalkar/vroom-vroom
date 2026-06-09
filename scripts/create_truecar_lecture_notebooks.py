import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUTPUT_DIR = Path("lecture_notes")


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def make_notebook(title, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "title": title,
    }
    return nb


COMMON_IMPORTS = """
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 120)

DATA_DIR = Path("../data")
CLEAN_DATA = DATA_DIR / "truecar_clean_combined.csv"
MODEL_DATA = DATA_DIR / "truecar_model_ready.csv"
"""


def notebook_01():
    cells = [
        md(
            """
            # TrueCar Used-Car Dataset: Data Import, Cleaning, and EDA

            These notes follow the same broad structure as the Ames Housing and Red Wine examples:
            import the data, inspect variables, separate numeric/categorical fields, handle missing
            values, create features, and prepare a modeling table.

            ## Learning Objectives

            - Import a cleaned observational dataset created from web-scraped car listings.
            - Distinguish raw listing fields from model-ready fields.
            - Explore the response variable for regression: `sales_price`.
            - Explore a classification label: `price_tier`.
            - Build a reproducible preprocessing object for later predictive models.
            """
        ),
        md(
            """
            ## Hypotheses and Modeling Questions

            The dataset supports both regression and classification questions.

            **Regression question:** Can listing features predict a used car's sales price?

            - Null idea: after accounting for sampling variation, vehicle attributes such as mileage,
              age, make, fuel type, and metro area do not explain meaningful variation in price.
            - Alternative idea: at least some of these features explain meaningful variation in price.

            **Classification question:** Can listing features classify a car as below, near, or above
            market relative to TrueCar's average market price?

            - Null idea: features do not classify `price_tier` better than a simple baseline.
            - Alternative idea: the model improves on the baseline classification rule.
            """
        ),
        code(COMMON_IMPORTS),
        md("## Import Data"),
        code(
            """
            if not CLEAN_DATA.exists():
                import sys
                sys.path.append("..")
                from truecar_data_pipeline import build_clean_dataset

                clean, report = build_clean_dataset(data_dir=DATA_DIR)
            else:
                clean = pd.read_csv(CLEAN_DATA)

            print(clean.shape)
            clean.head()
            """
        ),
        md("## Basic Overview"),
        code(
            """
            overview = pd.DataFrame({
                "dtype": clean.dtypes.astype(str),
                "missing": clean.isna().sum(),
                "missing_rate": clean.isna().mean().round(3),
                "unique_values": clean.nunique(dropna=True),
            }).sort_values(["missing_rate", "unique_values"], ascending=[False, False])

            overview.head(25)
            """
        ),
        code(
            """
            numeric_vars = clean.select_dtypes(include=np.number).columns.tolist()
            categorical_vars = clean.select_dtypes(exclude=np.number).columns.tolist()

            print(f"There are {len(numeric_vars)} numeric variables and {len(categorical_vars)} categorical/text variables.")
            print("Numeric variables:", numeric_vars)
            """
        ),
        md("## Response Variable: Sales Price"),
        code(
            """
            clean["log_sales_price"] = np.log(clean["sales_price"])

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.histplot(clean["sales_price"], bins=50, kde=True, ax=axes[0])
            axes[0].set_title("Sales Price Distribution")
            axes[0].set_xlabel("Sales Price")

            sns.histplot(clean["log_sales_price"], bins=50, kde=True, ax=axes[1])
            axes[1].set_title("Log Sales Price Distribution")
            axes[1].set_xlabel("Log Sales Price")
            plt.tight_layout()
            """
        ),
        md(
            """
            The raw price distribution is right-skewed, which is common for prices and incomes.
            A log transformation often makes linear modeling more stable and makes coefficient
            interpretation closer to percent changes.
            """
        ),
        md("## Market and Vehicle Coverage"),
        code(
            """
            metro_summary = (
                clean.groupby("source_metro")
                .agg(
                    rows=("url", "count"),
                    median_price=("sales_price", "median"),
                    median_miles=("odometer_miles", "median"),
                    unique_makes=("make", "nunique"),
                )
                .sort_values("rows", ascending=False)
            )
            metro_summary
            """
        ),
        code(
            """
            make_summary = (
                clean.groupby("make")
                .agg(rows=("url", "count"), median_price=("sales_price", "median"))
                .sort_values("rows", ascending=False)
                .head(20)
            )
            make_summary
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 5))
            top_makes = clean["make"].value_counts().head(15).index
            sns.boxplot(data=clean[clean["make"].isin(top_makes)], x="sales_price", y="make", ax=ax)
            ax.set_title("Sales Price by Common Makes")
            ax.set_xlabel("Sales Price")
            ax.set_ylabel("")
            plt.tight_layout()
            """
        ),
        md("## Correlation Analysis for Numeric Variables"),
        code(
            """
            corr_cols = [
                "sales_price",
                "log_sales_price",
                "odometer_miles",
                "vehicle_age",
                "average_market_price",
                "dealer_distance_miles",
                "feature_count",
                "standard_feature_count",
            ]
            corr_cols = [c for c in corr_cols if c in clean.columns]
            corr = clean[corr_cols].corr(numeric_only=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
            ax.set_title("Correlation among Numeric Features")
            plt.tight_layout()
            corr
            """
        ),
        md("## Classification Target Balance"),
        code(
            """
            price_tier_counts = clean["price_tier"].value_counts(dropna=False).rename_axis("price_tier").to_frame("rows")
            price_tier_counts["rate"] = (price_tier_counts["rows"] / len(clean)).round(3)
            price_tier_counts
            """
        ),
        md(
            """
            The target is imbalanced. This matters because a model can appear accurate by predicting
            the majority class. In the classification notebook, we will use stratified sampling,
            balanced class weights, and macro-averaged metrics.
            """
        ),
        md("## Feature Preparation for Later Models"),
        code(
            """
            modeling_columns = [
                "sales_price", "price_tier", "year", "vehicle_age", "make", "model", "trim",
                "odometer_miles", "fuel_type", "exterior", "interior", "dealer_state",
                "dealer_distance_miles", "source_metro", "feature_count", "standard_feature_count",
            ]

            model_df = clean[[c for c in modeling_columns if c in clean.columns]].copy()
            model_df = model_df.dropna(subset=["sales_price", "odometer_miles", "vehicle_age", "make", "model"])
            model_df.to_csv(MODEL_DATA, index=False)

            print(model_df.shape)
            model_df.head()
            """
        ),
        md(
            """
            ## Wrap-Up

            The modeling table now contains a continuous target (`sales_price`) and a categorical target
            (`price_tier`). The next notebooks use this table to build regression and classification
            models, while retaining the statistical-learning ideas from the Ames and Red Wine notes:
            hypotheses, train/test splits, likelihood, AIC/BIC, diagnostics, and predictive evaluation.
            """
        ),
    ]
    return make_notebook("TrueCar Data Import and EDA", cells)


def notebook_02():
    cells = [
        md(
            """
            # TrueCar Price Prediction: Linear Regression, MLE, AIC/BIC, and Final Model

            This notebook treats used-car price as a regression problem. We start with an interpretable
            linear model, connect ordinary least squares to maximum likelihood estimation, compare
            models using AIC/BIC, inspect diagnostics, and then fit a stronger predictive pipeline.
            """
        ),
        md(
            """
            ## Hypotheses

            Let `sales_price` be the observed listing price.

            **Overall regression hypothesis**

            - $H_0$: Mileage, age, make, fuel type, and metro area do not improve prediction beyond an intercept-only model.
            - $H_A$: At least one predictor has a non-zero association with price.

            **Mileage hypothesis**

            - $H_0: \\beta_{mileage}=0$
            - $H_A: \\beta_{mileage}<0$

            We expect higher mileage to be associated with lower price, holding other features constant.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            """
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            from scipy import stats

            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import KFold, cross_validate, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            """
        ),
        md("## Import Modeling Data"),
        code(
            """
            if MODEL_DATA.exists():
                cars = pd.read_csv(MODEL_DATA)
            else:
                cars = pd.read_csv(CLEAN_DATA)

            cars = cars.dropna(subset=["sales_price", "odometer_miles", "vehicle_age", "make", "model"]).copy()
            cars = cars[cars["sales_price"] > 0].copy()
            cars["log_sales_price"] = np.log(cars["sales_price"])
            cars["log_odometer"] = np.log1p(cars["odometer_miles"])

            print(cars.shape)
            cars.head()
            """
        ),
        md("## Train-Test Split"),
        code(
            """
            train, test = train_test_split(cars, test_size=0.25, random_state=571)
            print(train.shape, test.shape)
            """
        ),
        md("## Interpretable OLS Models"),
        code(
            """
            simple_model = smf.ols("log_sales_price ~ log_odometer + vehicle_age", data=train).fit()

            medium_model = smf.ols(
                "log_sales_price ~ log_odometer + vehicle_age + feature_count + C(fuel_type) + C(source_metro)",
                data=train,
            ).fit()

            top_makes = train["make"].value_counts().head(12).index
            train_ols = train.assign(make_lumped=np.where(train["make"].isin(top_makes), train["make"], "Other"))
            test_ols = test.assign(make_lumped=np.where(test["make"].isin(top_makes), test["make"], "Other"))

            full_model = smf.ols(
                "log_sales_price ~ log_odometer + vehicle_age + feature_count + standard_feature_count + "
                "C(fuel_type) + C(source_metro) + C(make_lumped)",
                data=train_ols,
            ).fit()

            comparison = pd.DataFrame({
                "model": ["simple", "medium", "full"],
                "adj_r2": [simple_model.rsquared_adj, medium_model.rsquared_adj, full_model.rsquared_adj],
                "log_likelihood": [simple_model.llf, medium_model.llf, full_model.llf],
                "AIC": [simple_model.aic, medium_model.aic, full_model.aic],
                "BIC": [simple_model.bic, medium_model.bic, full_model.bic],
            })
            comparison
            """
        ),
        md(
            """
            ## MLE Connection

            OLS can be viewed as maximum likelihood estimation when residuals are assumed to be
            independent and normally distributed:

            $$
            y_i = x_i^T\\beta + \\epsilon_i, \\quad \\epsilon_i \\sim N(0, \\sigma^2)
            $$

            Maximizing this likelihood is equivalent to minimizing the residual sum of squares. AIC and
            BIC use the maximized log-likelihood but penalize model complexity.
            """
        ),
        code(
            """
            resid = full_model.resid
            n = full_model.nobs
            rss = np.sum(resid ** 2)
            sigma2_mle = rss / n
            manual_loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)

            pd.DataFrame({
                "quantity": ["statsmodels logLik", "manual Gaussian logLik", "sigma2 MLE", "RSS"],
                "value": [full_model.llf, manual_loglik, sigma2_mle, rss],
            })
            """
        ),
        md("## Coefficient Table and Hypothesis Tests"),
        code(
            """
            coef_table = full_model.summary2().tables[1]
            coef_table.loc[coef_table.index.str.contains("log_odometer|vehicle_age|feature_count", regex=True)]
            """
        ),
        code(
            """
            mileage_coef = full_model.params["log_odometer"]
            mileage_p_two_sided = full_model.pvalues["log_odometer"]
            mileage_p_one_sided_negative = mileage_p_two_sided / 2 if mileage_coef < 0 else 1 - mileage_p_two_sided / 2

            print(f"log_odometer coefficient: {mileage_coef:.4f}")
            print(f"one-sided p-value for beta_mileage < 0: {mileage_p_one_sided_negative:.4g}")
            """
        ),
        md("## Regression Diagnostics"),
        code(
            """
            fitted = full_model.fittedvalues
            residuals = full_model.resid

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.scatterplot(x=fitted, y=residuals, alpha=0.35, ax=axes[0])
            axes[0].axhline(0, color="black", linewidth=1)
            axes[0].set_title("Residuals vs Fitted")
            axes[0].set_xlabel("Fitted log price")
            axes[0].set_ylabel("Residual")

            sm.qqplot(residuals, line="45", fit=True, ax=axes[1])
            axes[1].set_title("Normal Q-Q Plot")
            plt.tight_layout()
            """
        ),
        code(
            """
            influence = full_model.get_influence()
            cooks = influence.cooks_distance[0]
            used_index = full_model.model.data.row_labels
            influence_source = train_ols.loc[used_index]
            influential = pd.DataFrame({
                "cooks_distance": cooks,
                "sales_price": influence_source["sales_price"].values,
                "make": influence_source["make"].values,
                "model": influence_source["model"].values,
                "odometer_miles": influence_source["odometer_miles"].values,
            }).sort_values("cooks_distance", ascending=False)

            influential.head(10)
            """
        ),
        md("## Test-Set Evaluation for OLS"),
        code(
            """
            def regression_metrics(y_true, y_pred, label):
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                return {
                    "model": label,
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": rmse,
                    "R2": r2_score(y_true, y_pred),
                }

            pred_log = full_model.predict(test_ols)
            pred_price = np.exp(pred_log)

            ols_eval = test_ols[["sales_price"]].assign(pred_price=pred_price).dropna()
            ols_metrics = regression_metrics(ols_eval["sales_price"], ols_eval["pred_price"], "OLS log-price")
            pd.DataFrame([ols_metrics])
            """
        ),
        md("## Predictive Pipelines"),
        code(
            """
            numeric_features = ["vehicle_age", "odometer_miles", "dealer_distance_miles", "feature_count", "standard_feature_count"]
            categorical_features = ["make", "model", "fuel_type", "exterior", "interior", "dealer_state", "source_metro"]
            numeric_features = [c for c in numeric_features if c in cars.columns]
            categorical_features = [c for c in categorical_features if c in cars.columns]

            X = cars[numeric_features + categorical_features]
            y = cars["sales_price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=571)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical_features),
                ],
                remainder="drop",
                sparse_threshold=0.0,
            )

            models = {
                "Ridge": Ridge(alpha=10),
                "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=5, random_state=571, n_jobs=1),
                "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, random_state=571),
            }

            rows = []
            fitted_models = {}
            for name, estimator in models.items():
                pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
                pipe.fit(X_train, y_train)
                fitted_models[name] = pipe
                pred = pipe.predict(X_test)
                rows.append(regression_metrics(y_test, pred, name))

            results = pd.DataFrame(rows).sort_values("RMSE")
            results
            """
        ),
        md("## Cross-Validation"),
        code(
            """
            cv = KFold(n_splits=5, shuffle=True, random_state=571)
            scoring = {
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
                "R2": "r2",
            }

            cv_rows = []
            for name, estimator in models.items():
                pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
                cv_rows.append({
                    "model": name,
                    "CV_MAE": -scores["test_MAE"].mean(),
                    "CV_RMSE": -scores["test_RMSE"].mean(),
                    "CV_R2": scores["test_R2"].mean(),
                })

            pd.DataFrame(cv_rows).sort_values("CV_RMSE")
            """
        ),
        md("## Predicted vs Actual Plot"),
        code(
            """
            best_name = results.iloc[0]["model"]
            best_model = fitted_models[best_name]
            best_pred = best_model.predict(X_test)

            fig, ax = plt.subplots(figsize=(6, 6))
            sns.scatterplot(x=y_test, y=best_pred, alpha=0.35, ax=ax)
            low = min(y_test.min(), best_pred.min())
            high = max(y_test.max(), best_pred.max())
            ax.plot([low, high], [low, high], color="black", linewidth=1)
            ax.set_title(f"Predicted vs Actual: {best_name}")
            ax.set_xlabel("Actual sales price")
            ax.set_ylabel("Predicted sales price")
            plt.tight_layout()
            """
        ),
        md(
            """
            ## Conclusion

            The OLS model is valuable for interpretation, hypothesis testing, and AIC/BIC comparison.
            The machine-learning pipelines are more flexible and are usually better suited for final
            prediction. A careful report should include both: an interpretable statistical model and
            a predictive model evaluated on held-out data.
            """
        ),
    ]
    return make_notebook("TrueCar Regression", cells)


def notebook_03():
    cells = [
        md(
            """
            # TrueCar Price Tier Classification: Logistic Regression, AIC/BIC, and Predictive Models

            This notebook treats `price_tier` as a classification target. The goal is to predict whether
            a car is below, near, or above market based on listing attributes.
            """
        ),
        md(
            """
            ## Hypotheses

            **Baseline hypothesis**

            - $H_0$: Listing features do not improve classification beyond predicting the majority class.
            - $H_A$: Listing features improve classification performance beyond the majority-class baseline.

            **Logistic-regression hypothesis**

            For likelihood-based inference, we use a binary logistic model for `below_market`
            versus `not_below_market`. Each coefficient changes the log-odds of being below
            market. We compare nested models using log-likelihood, AIC, and BIC.

            For final prediction, we return to the full multiclass target: `below_market`,
            `near_market`, and `above_market`.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            """
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                f1_score,
            )
            from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            """
        ),
        md("## Import Classification Data"),
        code(
            """
            cars = pd.read_csv(MODEL_DATA if MODEL_DATA.exists() else CLEAN_DATA)
            cars = cars.dropna(subset=["price_tier", "sales_price", "odometer_miles", "vehicle_age", "make", "model"]).copy()
            cars["log_odometer"] = np.log1p(cars["odometer_miles"])

            print(cars.shape)
            cars["price_tier"].value_counts(normalize=True).to_frame("class_rate")
            """
        ),
        md("## Baseline Classifier"),
        code(
            """
            X_cols = [
                "vehicle_age", "odometer_miles", "dealer_distance_miles", "feature_count", "standard_feature_count",
                "make", "model", "fuel_type", "exterior", "interior", "dealer_state", "source_metro",
            ]
            X_cols = [c for c in X_cols if c in cars.columns]

            X = cars[X_cols]
            y = cars["price_tier"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=571, stratify=y
            )

            majority_class = y_train.mode()[0]
            baseline_pred = np.repeat(majority_class, len(y_test))

            baseline = {
                "model": "Majority class baseline",
                "accuracy": accuracy_score(y_test, baseline_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, baseline_pred),
                "macro_f1": f1_score(y_test, baseline_pred, average="macro"),
            }
            baseline
            """
        ),
        md("## Logistic Regression with AIC/BIC"),
        code(
            """
            # For an interpretable statsmodels model, use a binary target and lump high-cardinality makes.
            top_makes = cars["make"].value_counts().head(10).index
            sm_data = cars.assign(make_lumped=np.where(cars["make"].isin(top_makes), cars["make"], "Other"))
            sm_data["below_market"] = (sm_data["price_tier"] == "below_market").astype(int)

            base_formula = "below_market ~ log_odometer + vehicle_age"
            full_formula = "below_market ~ log_odometer + vehicle_age + feature_count + C(source_metro) + C(make_lumped)"

            logit_base = smf.logit(base_formula, data=sm_data).fit(method="lbfgs", maxiter=500, disp=False)
            logit_full = smf.logit(full_formula, data=sm_data).fit(method="lbfgs", maxiter=500, disp=False)

            logit_compare = pd.DataFrame({
                "model": ["base", "full"],
                "log_likelihood": [logit_base.llf, logit_full.llf],
                "AIC": [logit_base.aic, logit_full.aic],
                "BIC": [logit_base.bic, logit_full.bic],
            })
            logit_compare
            """
        ),
        md(
            """
            Lower AIC/BIC indicates a better tradeoff between fit and complexity. If the full model has
            much lower AIC/BIC than the base model, the additional vehicle and market variables are
            carrying useful classification information.
            """
        ),
        md("## Predictive Classification Pipelines"),
        code(
            """
            numeric_features = [c for c in ["vehicle_age", "odometer_miles", "dealer_distance_miles", "feature_count", "standard_feature_count"] if c in X.columns]
            categorical_features = [c for c in ["make", "model", "fuel_type", "exterior", "interior", "dealer_state", "source_metro"] if c in X.columns]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical_features),
                ],
                remainder="drop",
                sparse_threshold=0.0,
            )

            classifiers = {
                "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=571, n_jobs=1),
                "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=571),
            }

            rows = [baseline]
            fitted_classifiers = {}
            for name, estimator in classifiers.items():
                pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
                pipe.fit(X_train, y_train)
                fitted_classifiers[name] = pipe
                pred = pipe.predict(X_test)
                rows.append({
                    "model": name,
                    "accuracy": accuracy_score(y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro"),
                })

            class_results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
            class_results
            """
        ),
        md("## Cross-Validation"),
        code(
            """
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=571)
            scoring = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro",
            }

            cv_rows = []
            for name, estimator in classifiers.items():
                pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
                cv_rows.append({
                    "model": name,
                    "CV_accuracy": scores["test_accuracy"].mean(),
                    "CV_balanced_accuracy": scores["test_balanced_accuracy"].mean(),
                    "CV_macro_f1": scores["test_macro_f1"].mean(),
                })

            pd.DataFrame(cv_rows).sort_values("CV_macro_f1", ascending=False)
            """
        ),
        md("## Confusion Matrix and Class-Level Metrics"),
        code(
            """
            best_name = class_results.iloc[0]["model"]
            if best_name == "Majority class baseline":
                best_name = class_results.iloc[1]["model"]

            best_classifier = fitted_classifiers[best_name]
            best_pred = best_classifier.predict(X_test)

            print(best_name)
            print(classification_report(y_test, best_pred))

            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, best_pred, ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Confusion Matrix: {best_name}")
            plt.tight_layout()
            """
        ),
        md("## Probability Scores for Decision Support"),
        code(
            """
            if hasattr(best_classifier.named_steps["model"], "predict_proba"):
                proba = best_classifier.predict_proba(X_test)
                proba_df = pd.DataFrame(proba, columns=best_classifier.named_steps["model"].classes_, index=X_test.index)
                examples = cars.loc[X_test.index, ["make", "model", "sales_price", "price_tier"]].join(proba_df)
                examples.head(10)
            """
        ),
        md(
            """
            ## Conclusion

            Accuracy alone is not enough for this task because `price_tier` is imbalanced. Balanced
            accuracy, macro F1, and confusion matrices show whether the model is learning minority
            classes or simply predicting the dominant class. The statistical logistic model supports
            likelihood-based comparison, while the sklearn pipelines provide a stronger final
            predictive workflow.
            """
        ),
    ]
    return make_notebook("TrueCar Classification", cells)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    notebooks = {
        "01_truecar_data_import_eda.ipynb": notebook_01(),
        "02_truecar_price_regression.ipynb": notebook_02(),
        "03_truecar_price_tier_classification.ipynb": notebook_03(),
    }
    for name, nb in notebooks.items():
        path = OUTPUT_DIR / name
        nbf.write(nb, path)
        print(path)


if __name__ == "__main__":
    main()
