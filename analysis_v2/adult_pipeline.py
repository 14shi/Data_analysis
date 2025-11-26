"""Comprehensive Adult income analysis with enhanced preprocessing and models.

The script rebuilds the entire experimentation workflow:
1. 数据获取与描述性统计
2. 高级预处理与特征工程
3. 多模型训练、交叉验证与独立测试集评估
4. 结果可视化与特征重要性解释

All intermediate artefacts are saved under `analysis_v2/artifacts/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")

ARTIFACTS_DIR = Path("analysis_v2") / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 7
TEST_SIZE = 0.25
CV_SPLITS = 5


def sanitize_column(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def fetch_adult_dataframe() -> pd.DataFrame:
    adult = fetch_ucirepo(id=2)
    features = adult.data.features.copy()
    target = adult.data.targets.copy()
    df = pd.concat([features, target], axis=1)
    df.columns = [sanitize_column(col) for col in df.columns]
    return df


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.replace("?", np.nan, inplace=True)
    str_cols = cleaned.select_dtypes(include="object").columns
    cleaned[str_cols] = cleaned[str_cols].apply(lambda col: col.str.strip())
    cleaned.dropna(subset=["income"], inplace=True)
    cleaned["income_label"] = cleaned["income"].apply(lambda val: 1 if ">50K" in val else 0)
    cleaned = cleaned.drop(columns=["income"])
    cleaned = cleaned.drop_duplicates()
    cleaned["hours_per_week"] = cleaned["hours_per_week"].clip(lower=1, upper=80)
    cleaned["capital_gain"] = cleaned["capital_gain"].clip(lower=0, upper=99999)
    cleaned["capital_loss"] = cleaned["capital_loss"].clip(lower=0, upper=4356)
    return cleaned


def deduplicate(columns: List[str]) -> List[str]:
    return list(dict.fromkeys(columns))


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create non-linear signals and grouped categorical attributes."""

    def __init__(self, rare_threshold: float = 0.01):
        self.rare_threshold = rare_threshold
        self.group_columns = ["workclass", "occupation", "native_country"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AdultFeatureEngineer":
        self.rare_maps_: Dict[str, pd.Series] = {}
        for col in self.group_columns:
            freq = X[col].value_counts(normalize=True)
            self.rare_maps_[col] = freq[freq < self.rare_threshold].index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.group_columns:
            rare_levels = self.rare_maps_.get(col, [])
            df[col + "_grouped"] = df[col].apply(lambda val: "Other" if val in rare_levels else val)

        df["capital_net"] = df["capital_gain"] - df["capital_loss"]
        df["capital_gain_log"] = np.log1p(df["capital_gain"])
        df["capital_loss_log"] = np.log1p(df["capital_loss"])
        df["capital_intensity"] = (df["capital_gain"] + 1) / (df["hours_per_week"] + 1)
        df["education_hours_interaction"] = df["education_num"] * df["hours_per_week"]
        df["is_married"] = df["marital_status"].isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
        df["is_government_worker"] = df["workclass"].str.contains("gov", case=False, na=False).astype(int)
        df["is_self_employed"] = df["workclass"].str.contains("Self", na=False).astype(int)
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[16, 25, 35, 45, 55, 65, 100],
            labels=["16-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
        df["hours_bucket"] = pd.cut(
            df["hours_per_week"],
            bins=[0, 30, 40, 50, 60, 100],
            labels=["<30", "30-39", "40-49", "50-59", "60+"],
        )
        df["capital_positive_flag"] = (df["capital_gain"] > 0).astype(int)
        df["overtime_flag"] = (df["hours_per_week"] >= 45).astype(int)
        df["native_is_us"] = df["native_country"].eq("United-States").astype(int)
        df["household_role"] = df["relationship"].replace(
            {
                "Husband": "Primary earner",
                "Wife": "Secondary earner",
                "Own-child": "Dependent",
                "Unmarried": "Single",
            }
        )
        df["education_level_grouped"] = df["education"].replace(
            {
                "Preschool": "Dropout",
                "1st-4th": "Dropout",
                "5th-6th": "Dropout",
                "7th-8th": "Dropout",
                "9th": "Dropout",
                "10th": "Dropout",
                "11th": "Dropout",
                "12th": "Dropout",
                "HS-grad": "HighSchool",
                "Some-college": "SomeCollege",
                "Assoc-acdm": "Associate",
                "Assoc-voc": "Associate",
                "Bachelors": "Bachelors",
                "Masters": "Masters",
                "Prof-school": "Professional",
                "Doctorate": "Doctorate",
            }
        )
        return df


def generate_eda_artifacts(df: pd.DataFrame) -> None:
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "class_balance": df["income_label"].value_counts(normalize=True).round(4).to_dict(),
    }
    (ARTIFACTS_DIR / "data_overview.json").write_text(json.dumps(overview, indent=2))

    missing = (
        df.isna().sum().sort_values(ascending=False).reset_index()
    )
    missing.columns = ["column", "missing_count"]
    missing.to_csv(ARTIFACTS_DIR / "missing_values.csv", index=False)

    categorical_cols = df.select_dtypes(include="object").columns
    distribution_frames = []
    for col in categorical_cols:
        counts = df[col].value_counts(normalize=True).round(4).reset_index()
        counts.columns = [col, "ratio"]
        counts.insert(0, "feature", col)
        distribution_frames.append(counts)
    if distribution_frames:
        cat_distribution = pd.concat(distribution_frames, ignore_index=True)
        cat_distribution.to_csv(ARTIFACTS_DIR / "categorical_distributions.csv", index=False)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss", "fnlwgt"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, ax in zip(numeric_cols, axes.flatten()):
        sns.histplot(
            data=df,
            x=col,
            hue="income_label",
            kde=True,
            stat="density",
            common_norm=False,
            ax=ax,
            palette="tab10",
        )
        ax.set_title(f"{col} 分布")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "numeric_distributions.png", dpi=200)
    plt.close()


def plot_income_rate_by_category(df: pd.DataFrame, column: str, filename: str, top_n: int = 12) -> None:
    summary = (
        df.groupby(column)["income_label"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary, x="income_label", y=column, palette="viridis")
    plt.xlabel("收入>50K 概率")
    plt.ylabel(column)
    plt.title(f"{column} 与高收入概率 Top {top_n}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / filename, dpi=200)
    plt.close()


def plot_boxplots_by_income(df: pd.DataFrame) -> None:
    metrics = ["age", "education_num", "hours_per_week", "capital_gain_log"]
    melted = df[metrics + ["income_label"]].melt(id_vars="income_label", var_name="feature", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="income_label", palette="Set2")
    plt.xlabel("特征")
    plt.ylabel("取值分布")
    plt.title("不同收入标签的数值特征分布")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "boxplot_income.png", dpi=200)
    plt.close()


def plot_pairwise_relationships(df: pd.DataFrame) -> None:
    cols = ["age", "education_num", "hours_per_week", "capital_gain_log"]
    available_cols = [c for c in cols if c in df.columns]
    sample = df[available_cols + ["income_label"]].sample(
        n=min(2000, len(df)), random_state=RANDOM_STATE
    )
    pair_plot = sns.pairplot(
        sample,
        vars=available_cols,
        hue="income_label",
        diag_kind="kde",
        palette="tab10",
        plot_kws={"alpha": 0.6, "s": 20},
    )
    pair_plot.fig.suptitle("关键数值特征成对关系", y=1.02)
    pair_plot.fig.savefig(ARTIFACTS_DIR / "pairplot_income.png", dpi=200)
    plt.close()


def plot_pca_projection(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "income_label"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    comp_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    comp_df["income_label"] = df["income_label"].values
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=comp_df.sample(n=min(6000, len(comp_df)), random_state=RANDOM_STATE),
        x="PC1",
        y="PC2",
        hue="income_label",
        palette="tab10",
        alpha=0.6,
    )
    plt.title("PCA 投影（含工程特征）")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "pca_scatter.png", dpi=200)
    plt.close()


def create_additional_visuals(cleaned_df: pd.DataFrame, engineered_df: pd.DataFrame) -> None:
    plot_numeric_distributions(cleaned_df)
    plot_income_rate_by_category(engineered_df, "education_level_grouped", "income_by_education.png")
    plot_income_rate_by_category(engineered_df, "occupation_grouped", "income_by_occupation.png")
    if "sex" in engineered_df.columns:
        plot_income_rate_by_category(engineered_df, "sex", "income_by_gender.png", top_n=5)
    plot_pairwise_relationships(engineered_df)
    plot_pca_projection(engineered_df)
    plot_boxplots_by_income(engineered_df)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def build_models(
    numeric_cols: List[str],
    categorical_cols: List[str],
    pos_weight: float,
    use_feature_engineer: bool = True,
) -> Dict[str, Pipeline]:
    def build_pipeline(classifier: BaseEstimator) -> Pipeline:
        steps: List[tuple] = []
        if use_feature_engineer:
            steps.append(("feature_engineer", AdultFeatureEngineer()))
            target_numeric = numeric_cols
            target_categorical = categorical_cols
        else:
            target_numeric = deduplicate(
                [
                    col
                    for col in numeric_cols
                    if col
                    not in [
                        "capital_net",
                        "capital_gain_log",
                        "capital_loss_log",
                        "capital_intensity",
                        "education_hours_interaction",
                    ]
                ]
            )
            target_categorical = deduplicate(
                [
                    col
                    for col in categorical_cols
                    if col
                    not in [
                        "workclass_grouped",
                        "occupation_grouped",
                        "native_country_grouped",
                        "age_bucket",
                        "hours_bucket",
                        "household_role",
                        "education_level_grouped",
                        "sex",
                    ]
                ]
            )
        steps.append(
            (
                "preprocessor",
                build_preprocessor(target_numeric, target_categorical),
            )
        )
        steps.append(("classifier", classifier))
        return Pipeline(steps=steps)

    models = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": build_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_depth=8,
                max_iter=400,
                min_samples_leaf=25,
                l2_regularization=0.5,
                random_state=RANDOM_STATE,
            )
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
            )
        ),
    }
    return models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], str, Pipeline]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_rows = []
    test_metrics: Dict[str, Dict[str, float]] = {}
    best_model_name: str | None = None
    best_f1 = -np.inf
    best_pipeline: Pipeline | None = None

    for name, pipeline in models.items():
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_summary = {"model": name}
        for metric, scores in cv_result.items():
            if metric.startswith("test_"):
                clean_metric = metric.replace("test_", "")
                cv_summary[clean_metric] = float(np.mean(scores))
        cv_rows.append(cv_summary)

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(X_test)[:, 1]
        else:
            probas = pipeline.decision_function(X_test)
        test_metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probas),
        }
        if test_metrics[name]["f1"] > best_f1:
            best_f1 = test_metrics[name]["f1"]
            best_model_name = name
            best_pipeline = pipeline

    cv_table = pd.DataFrame(cv_rows)
    return cv_table, test_metrics, best_model_name or "", best_pipeline


def save_metrics(cv_table: pd.DataFrame, test_metrics: Dict[str, Dict[str, float]]) -> None:
    cv_table.to_csv(ARTIFACTS_DIR / "cv_metrics.csv", index=False)
    (ARTIFACTS_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


def plot_confusion_and_roc(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    preds = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_test)[:, 1]
    else:
        probas = pipeline.decision_function(X_test)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()


def plot_partial_dependence_curves(
    pipeline: Pipeline, X_test: pd.DataFrame, model_name: str
) -> None:
    candidate_features = ["age", "hours_per_week", "education_num", "capital_gain"]
    features = [feat for feat in candidate_features if feat in X_test.columns]
    if not features:
        return
    disp = PartialDependenceDisplay.from_estimator(
        pipeline,
        X_test,
        features=features,
        kind="average",
        grid_resolution=20,
        subsample=2000,
    )
    disp.figure_.suptitle(f"{model_name} - 局部依赖曲线", y=1.02)
    disp.figure_.tight_layout()
    disp.figure_.savefig(ARTIFACTS_DIR / f"{model_name}_partial_dependence.png", dpi=200)
    plt.close(disp.figure_)


def export_feature_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    importance = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_values = importance.importances_mean
    take = min(len(feature_names), len(importance_values))
    ranking = pd.DataFrame(
        {"feature": feature_names[:take], "importance": importance_values[:take]}
    ).sort_values(by="importance", ascending=False)
    ranking.to_csv(ARTIFACTS_DIR / f"{model_name}_feature_importance.csv", index=False)


def main() -> None:
    raw_df = fetch_adult_dataframe()
    cleaned_df = clean_raw_dataframe(raw_df)
    generate_eda_artifacts(cleaned_df)

    X = cleaned_df.drop(columns=["income_label"])
    y = cleaned_df["income_label"]

    feature_engineer = AdultFeatureEngineer()
    feature_engineer.fit(X, y)
    engineered_view = feature_engineer.transform(X.copy())
    engineered_view["income_label"] = y
    create_additional_visuals(cleaned_df, engineered_view)

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist() + [
        "capital_net",
        "capital_gain_log",
        "capital_loss_log",
        "capital_intensity",
        "education_hours_interaction",
    ]
    categorical_cols = X.select_dtypes(include="object").columns.tolist() + [
        "workclass_grouped",
        "occupation_grouped",
        "native_country_grouped",
        "age_bucket",
        "hours_bucket",
        "household_role",
        "education_level_grouped",
    ]
    numeric_cols = deduplicate(numeric_cols)
    categorical_cols = deduplicate(categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    models = build_models(numeric_cols, categorical_cols, pos_weight)
    enriched_models = build_models(numeric_cols, categorical_cols, pos_weight, use_feature_engineer=True)
    enriched_cv, enriched_test, best_model_name, best_pipeline = evaluate_models(
        enriched_models, X_train, X_test, y_train, y_test
    )
    save_metrics(enriched_cv, enriched_test)

    baseline_models = build_models(numeric_cols, categorical_cols, pos_weight, use_feature_engineer=False)
    baseline_cv, baseline_test, _, _ = evaluate_models(
        baseline_models, X_train, X_test, y_train, y_test
    )
    baseline_cv.to_csv(ARTIFACTS_DIR / "cv_metrics_baseline.csv", index=False)
    (ARTIFACTS_DIR / "test_metrics_baseline.json").write_text(json.dumps(baseline_test, indent=2))

    if best_pipeline:
        plot_confusion_and_roc(best_pipeline, X_test, y_test, best_model_name)
        export_feature_importance(best_pipeline, X_test, y_test, best_model_name)
        plot_partial_dependence_curves(best_pipeline, X_test, best_model_name)

    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "test_metrics": enriched_test.get(best_model_name, {}),
                "cv_metrics_file": "cv_metrics.csv",
                "feature_importance_file": f"{best_model_name}_feature_importance.csv",
                "baseline_metrics_file": "test_metrics_baseline.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

