"""Train the Aero Toolkit production model from a cleaned CSV file.

This script turns Notebook 2 into a repeatable command-line workflow. It can be
run locally from the repo root with:

    python src/train_model.py

Default inputs and outputs:
    input CSV:      data/processed/openfoam_phase1_cleaned.csv
    model artifact: models/notebook2_gradient_boosting.joblib
    metadata JSON:  models/notebook2_model_metadata.json
    predictions:    models/notebook2_gradient_boosting_holdout_predictions.csv

The default model filename intentionally matches the current Streamlit app and
`src/inference.py` expectation. The metadata JSON records which model family was
actually selected and tuned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.preprocess import TARGET_COLUMN, build_preprocessor, load_and_prepare_dataset
    from src.utils import (
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        PROJECT_ROOT,
        ensure_directory,
        export_holdout_predictions,
        print_section,
        regression_metrics,
        save_json,
        save_model,
        save_model_and_metadata,
        timestamp_utc,
    )
except ModuleNotFoundError:
    from preprocess import TARGET_COLUMN, build_preprocessor, load_and_prepare_dataset
    from utils import (
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        PROJECT_ROOT,
        ensure_directory,
        export_holdout_predictions,
        print_section,
        regression_metrics,
        save_json,
        save_model,
        save_model_and_metadata,
        timestamp_utc,
    )


RANDOM_STATE = 42


def build_model_pipelines(feature_columns: List[str], random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Create candidate sklearn pipelines for the tabular flow-separation dataset."""
    preprocessor = build_preprocessor(feature_columns)

    return {
        "Ridge": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge(random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=random_state,
                        min_samples_leaf=1,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=200,
                        random_state=random_state,
                        min_samples_leaf=1,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def choose_cv_strategy(n_samples: int, random_state: int = RANDOM_STATE) -> RepeatedKFold:
    """Choose a repeated K-fold strategy that is safe for smaller datasets."""
    if n_samples < 10:
        raise ValueError("At least 10 rows are needed for this training workflow.")

    n_splits = min(5, n_samples)
    n_repeats = 3 if n_samples >= 50 else 2
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def evaluate_candidates(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate all candidate model families with repeated cross-validation."""
    cv = choose_cv_strategy(len(X_train), random_state=random_state)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows = []
    for model_name, pipeline in pipelines.items():
        print(f"Evaluating {model_name}...")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": model_name,
                "cv_rmse_mean": float(-scores["test_rmse"].mean()),
                "cv_rmse_std": float(scores["test_rmse"].std()),
                "cv_mae_mean": float(-scores["test_mae"].mean()),
                "cv_mae_std": float(scores["test_mae"].std()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "cv_r2_std": float(scores["test_r2"].std()),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)
    return results


def parameter_grid_for(model_name: str) -> Dict[str, List[Any]]:
    """Return a compact GridSearchCV grid for the selected model family."""
    grids: Dict[str, Dict[str, List[Any]]] = {
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 1.0],
        },
        "Extra Trees": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 1.0],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
            "model__min_samples_leaf": [1, 2],
        },
    }

    if model_name not in grids:
        raise ValueError(f"No parameter grid defined for model family: {model_name}")
    return grids[model_name]


def tune_selected_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Tune the selected best model family with GridSearchCV."""
    cv = choose_cv_strategy(len(X_train), random_state=random_state)
    param_grid = parameter_grid_for(model_name)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def compute_permutation_importance_table(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute permutation importance on original input features."""
    result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=random_state,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df.reset_index(drop=True)


def relative_to_project(path: Path) -> str:
    """Convert a path to a repo-relative string when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def train_and_save(
    data_path: Path,
    model_output: Path,
    metadata_output: Path,
    predictions_output: Path,
    cv_results_output: Path,
    feature_importance_output: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Run the full training workflow and save the production artifacts."""
    print_section("Loading and validating dataset")
    X, y, feature_columns_used, validation_report = load_and_prepare_dataset(
        data_path,
        drop_constant_features=True,
    )
    print(f"Rows: {len(X)}")
    print(f"Features used: {feature_columns_used}")
    print(f"Dropped constant features: {validation_report.get('dropped_constant_features', [])}")

    print_section("Creating train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print_section("Repeated cross-validation model comparison")
    pipelines = build_model_pipelines(feature_columns_used, random_state=random_state)
    cv_results = evaluate_candidates(pipelines, X_train, y_train, random_state=random_state)
    ensure_directory(cv_results_output.parent)
    cv_results.to_csv(cv_results_output, index=False)
    print(cv_results)

    best_model_name = str(cv_results.iloc[0]["model"])
    print(f"Best model family before tuning: {best_model_name}")

    print_section("GridSearchCV tuning")
    search = tune_selected_model(
        best_model_name,
        pipelines[best_model_name],
        X_train,
        y_train,
        random_state=random_state,
    )
    best_pipeline = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV RMSE during tuning: {-search.best_score_:.6f}")

    print_section("Held-out test evaluation")
    y_pred = best_pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    holdout_metrics = regression_metrics(y_test, y_pred)
    print(json.dumps(holdout_metrics, indent=2))

    print_section("Permutation feature importance")
    importance_df = compute_permutation_importance_table(
        best_pipeline,
        X_test,
        y_test,
        random_state=random_state,
    )
    ensure_directory(feature_importance_output.parent)
    importance_df.to_csv(feature_importance_output, index=False)
    print(importance_df)

    print_section("Saving artifacts")
    saved_predictions_path = export_holdout_predictions(X_test, y_test, y_pred, predictions_output)

    metadata: Dict[str, Any] = {
        "project": "Aero Toolkit",
        "target_column": TARGET_COLUMN,
        "source_data": relative_to_project(data_path),
        "n_rows_total": int(len(X)),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "feature_columns_used": feature_columns_used,
        "dropped_constant_features": validation_report.get("dropped_constant_features", []),
        "best_model_family": best_model_name,
        "best_params": search.best_params_,
        "tuning_best_cv_rmse": float(-search.best_score_),
        "holdout_metrics": holdout_metrics,
        "cv_results_csv": relative_to_project(cv_results_output),
        "holdout_predictions_csv": relative_to_project(saved_predictions_path),
        "permutation_feature_importance_csv": relative_to_project(feature_importance_output),
        "validation_warnings": validation_report.get("warnings", []),
        "random_state": random_state,
        "test_size": test_size,
        "trained_at_utc": timestamp_utc(),
        "notes": (
            "This is a Phase 1 or Phase 2 screening model trained on simulated CFD-style data. "
            "It is intended for educational aerodynamic screening, not production CFD replacement."
        ),
    }

    saved_paths = save_model_and_metadata(
        best_pipeline,
        model_output,
        metadata=metadata,
        metadata_path=metadata_output,
    )

    print(f"Saved model: {saved_paths['model']}")
    print(f"Saved metadata: {saved_paths['metadata']}")
    print(f"Saved holdout predictions: {saved_predictions_path}")
    print(f"Saved CV results: {cv_results_output}")
    print(f"Saved feature importance: {feature_importance_output}")

    return {
        "model_path": str(saved_paths["model"]),
        "metadata_path": str(saved_paths["metadata"]),
        "predictions_path": str(saved_predictions_path),
        "cv_results_path": str(cv_results_output),
        "feature_importance_path": str(feature_importance_output),
        "best_model_family": best_model_name,
        "holdout_metrics": holdout_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_data_path = PROCESSED_DATA_DIR / "openfoam_phase1_cleaned.csv"
    default_model_output = MODELS_DIR / "notebook2_gradient_boosting.joblib"
    default_metadata_output = MODELS_DIR / "notebook2_model_metadata.json"
    default_predictions_output = MODELS_DIR / "notebook2_gradient_boosting_holdout_predictions.csv"
    default_cv_results_output = MODELS_DIR / "notebook2_cv_results.csv"
    default_feature_importance_output = MODELS_DIR / "notebook2_permutation_feature_importance.csv"

    parser = argparse.ArgumentParser(description="Train the Aero Toolkit flow-separation model.")
    parser.add_argument("--data", type=Path, default=default_data_path, help="Path to cleaned CSV data.")
    parser.add_argument("--model-output", type=Path, default=default_model_output, help="Where to save the joblib model.")
    parser.add_argument("--metadata-output", type=Path, default=default_metadata_output, help="Where to save model metadata JSON.")
    parser.add_argument("--predictions-output", type=Path, default=default_predictions_output, help="Where to save holdout predictions CSV.")
    parser.add_argument("--cv-results-output", type=Path, default=default_cv_results_output, help="Where to save CV comparison CSV.")
    parser.add_argument("--feature-importance-output", type=Path, default=default_feature_importance_output, help="Where to save permutation importance CSV.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction of rows held out for final testing.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    result = train_and_save(
        data_path=args.data,
        model_output=args.model_output,
        metadata_output=args.metadata_output,
        predictions_output=args.predictions_output,
        cv_results_output=args.cv_results_output,
        feature_importance_output=args.feature_importance_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print_section("Training complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
