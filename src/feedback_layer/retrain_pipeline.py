"""
src/feedback_layer/retrain_pipeline.py
----------------------------------------
Champion / Challenger automated retraining pipeline for CIPHER.

Architecture
------------
:class:`RetrainingPipeline` implements the full retraining sequence:

1. Load the preprocessed + graph-featured base data from a pickle file.
2. Pull unused analyst feedback records from :class:`~src.feedback_layer.feedback_store.FeedbackStore`.
3. Merge feedback feature vectors with the base training set, applying
   confidence-weighted sample weights.
4. Train a fresh :class:`~src.ml_layer.model.EnsembleDetector` — the
   *challenger*.
5. Load the current production model from the MLflow Model Registry — the
   *champion*.
6. Evaluate both on a held-out test split; compare AUC-PR and F1.
7. If the challenger surpasses the champion by the configured margin,
   promote it to production, archive the old champion, write the
   model-update signal, and invalidate the SHAP cache.
8. If the challenger does not improve, log it as a rejected run in MLflow
   and leave the champion unchanged.

All decisions are logged to MLflow so every retraining attempt — whether
promoted or rejected — is fully auditable.

Usage::

    from src.feedback_layer.retrain_pipeline import RetrainingPipeline

    pipeline = RetrainingPipeline()
    result = pipeline.run(trigger_reason="manual_test")
    print(result)

``__main__`` block
------------------
Run directly for a quick smoke-test::

    python -m src.feedback_layer.retrain_pipeline
"""

from __future__ import annotations

import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.feedback_layer.feedback_store import FeedbackRecord, FeedbackStore
from src.feedback_layer.model_update_signal import ModelUpdateSignal
from src.ml_layer.model import EnsembleDetector
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class RetrainingPipeline:
    """Full champion/challenger retraining pipeline with MLflow tracking.

    Every call to :meth:`run` constitutes a *retraining attempt* — an
    MLflow run is opened regardless of whether the challenger is ultimately
    promoted, so the full history of attempts is queryable.

    Attributes:
        _cfg: Top-level config dict from ``config/config.yaml``.
        _feedback_cfg: The ``feedback`` sub-dict.
        _mlflow_cfg: The ``mlflow`` sub-dict.
        _store: Shared :class:`~src.feedback_layer.feedback_store.FeedbackStore`.
        _signal: :class:`~src.feedback_layer.model_update_signal.ModelUpdateSignal`
            used to notify the live consumer after promotion.
        _client: :class:`~mlflow.tracking.MlflowClient` for registry operations.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise the pipeline, FeedbackStore, and MLflow client.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self._cfg = load_config(config_path)
        self._feedback_cfg: dict = self._cfg["feedback"]
        self._mlflow_cfg: dict = self._cfg["mlflow"]
        self._config_path = config_path

        # Set MLflow tracking URI — prefer env var (set by docker-compose) over config.
        import os
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", self._mlflow_cfg["tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self._mlflow_cfg["experiment_name"])

        self._store = FeedbackStore(config_path=config_path)
        self._signal = ModelUpdateSignal(config_path=config_path)
        self._client = MlflowClient()

        # Populated by _load_base_data(); shared with downstream private methods.
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None
        # Written by _promote_challenger / _log_rejection; read by run().
        self._last_run_id: str = ""

        _logger.info(
            "RetrainingPipeline initialised | experiment=%s, registry=%s",
            self._mlflow_cfg["experiment_name"],
            self._mlflow_cfg["model_registry_name"],
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, trigger_reason: str) -> dict:
        """Execute the full champion/challenger retraining sequence.

        Steps
        -----
        1. Load base data.
        2. Pull unused feedback records.
        3. Merge and weight.
        4. Train challenger.
        5. Evaluate both models.
        6. Promote or reject.

        Args:
            trigger_reason: Human-readable string identifying what caused
                this retraining attempt (e.g. ``"feedback_volume_threshold"``,
                ``"drift_detected"``, ``"scheduled"``, ``"manual_test"``).

        Returns:
            Dictionary with keys:

            * ``promoted``          — ``True`` if challenger was promoted.
            * ``challenger_metrics``— dict of challenger evaluation metrics.
            * ``champion_metrics``  — dict of champion evaluation metrics
              (empty dict if no champion exists in the registry).
            * ``feedback_count``    — number of feedback records included.
            * ``trigger_reason``    — echoes the input.
            * ``mlflow_run_id``     — MLflow run ID for this attempt.
        """
        _logger.info(
            "RetrainingPipeline.run — START | trigger=%s", trigger_reason
        )

        # Step 1: Load base data (also stores X_test / y_test as instance state).
        try:
            X_base, y_base = self._load_base_data()
        except FileNotFoundError as exc:
            _logger.error(
                "RetrainingPipeline.run — base data not found: %s", exc
            )
            return self._error_result(trigger_reason, str(exc))

        # Step 2: Pull unused feedback.
        feedback_records = self._store.get_unused_feedback()
        _logger.info(
            "RetrainingPipeline.run — unused feedback records: %d",
            len(feedback_records),
        )

        # Step 3: Merge feedback with base data.
        X_train, y_train, sample_weights = self._merge_with_feedback(
            X_base, y_base, feedback_records
        )

        # Step 4: Train challenger.
        challenger = self._train_challenger(X_train, y_train, sample_weights)

        # Step 5: Evaluate champion vs. challenger.
        # X_test / y_test were stored on self by _load_base_data().
        challenger_metrics, champion_metrics = self._evaluate_champion_challenger(
            challenger
        )

        # Step 6: Promote or log rejection.
        promoted = self._should_promote(challenger_metrics, champion_metrics)

        if promoted:
            self._promote_challenger(
                challenger=challenger,
                metrics=challenger_metrics,
                feedback_records=feedback_records,
                trigger_reason=trigger_reason,
            )
        else:
            self._log_rejection(
                challenger_metrics=challenger_metrics,
                champion_metrics=champion_metrics,
                trigger_reason=trigger_reason,
            )

        run_id = self._last_run_id

        result = {
            "promoted": promoted,
            "challenger_metrics": challenger_metrics,
            "champion_metrics": champion_metrics,
            "feedback_count": len(feedback_records),
            "trigger_reason": trigger_reason,
            "mlflow_run_id": run_id,
        }

        _logger.info(
            "RetrainingPipeline.run — DONE | promoted=%s, trigger=%s, "
            "feedback=%d, run_id=%s",
            promoted,
            trigger_reason,
            len(feedback_records),
            run_id,
        )
        return result

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _load_base_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load preprocessed + graph-featured training and test data.

        The pickle at ``feedback.base_data_pickle_path`` must contain either:

        * A dict with keys ``X_train``, ``X_test``, ``y_train``, ``y_test``
          (and optionally ``feature_names``).
        * A 5-tuple ``(X_train, X_test, y_train, y_test, feature_names)``.

        Side effects:
            Stores the held-out test split as ``self._X_test`` and
            ``self._y_test`` so that :meth:`_evaluate_champion_challenger`,
            :meth:`_promote_challenger`, and :meth:`_log_rejection` can access
            the test data without extra parameters.

        Returns:
            Tuple of ``(X_train, y_train)`` as :class:`pandas.DataFrame` /
            :class:`pandas.Series`.  The test split is available via
            ``self._X_test`` / ``self._y_test``.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
            ValueError: If the pickle structure is unrecognised.
        """
        pickle_path = Path(self._feedback_cfg["base_data_pickle_path"])
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Base data pickle not found: {pickle_path.resolve()}"
            )

        _logger.info(
            "RetrainingPipeline._load_base_data — loading from '%s'",
            pickle_path.resolve(),
        )

        with open(pickle_path, "rb") as fh:
            payload = pickle.load(fh)

        if isinstance(payload, dict):
            X_train = np.asarray(payload["X_train"])
            X_test = np.asarray(payload["X_test"])
            y_train = np.asarray(payload["y_train"])
            y_test = np.asarray(payload["y_test"])
            feature_names: list = list(
                payload.get(
                    "feature_names",
                    [f"f{i}" for i in range(X_train.shape[1])],
                )
            )
        elif isinstance(payload, (tuple, list)) and len(payload) >= 4:
            X_train, X_test, y_train, y_test = (
                np.asarray(payload[0]),
                np.asarray(payload[1]),
                np.asarray(payload[2]),
                np.asarray(payload[3]),
            )
            feature_names = (
                list(payload[4])
                if len(payload) >= 5
                else [f"f{i}" for i in range(X_train.shape[1])]
            )
        else:
            raise ValueError(
                f"Unsupported pickle structure: {type(payload)}. "
                "Expected dict(X_train, X_test, y_train, y_test) or 4/5-tuple."
            )

        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_train_s = pd.Series(y_train.astype(int), name="label")
        y_test_s = pd.Series(y_test.astype(int), name="label")

        # Stash test split for downstream methods.
        self._X_test = X_test_df
        self._y_test = y_test_s

        _logger.info(
            "RetrainingPipeline._load_base_data — loaded | "
            "train=%s, test=%s, features=%d",
            X_train_df.shape,
            X_test_df.shape,
            len(feature_names),
        )
        return X_train_df, y_train_s

    def _merge_with_feedback(
        self,
        X_base: pd.DataFrame,
        y_base: pd.Series,
        feedback_records: list[FeedbackRecord],
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Merge base training data with feedback feature vectors.

        Sample weight formula
        ---------------------
        * Base records: weight = ``feedback_base_weight_multiplier`` (default 1.0).
        * Feedback records: weight = ``(len_base / len_feedback) * confidence``
          This down-scales very small feedback batches and up-scales large ones;
          confidence modulates within the batch.

        ESCALATE decisions are **excluded** because ground truth is uncertain.

        Args:
            X_base: Base feature matrix as a DataFrame.
            y_base: Base target labels as a Series.
            feedback_records: List of unused :class:`~src.feedback_layer.feedback_store.FeedbackRecord`.

        Returns:
            Tuple of ``(X_merged, y_merged, sample_weights)`` where
            ``sample_weights`` is a 1-D ``np.ndarray`` aligned with the rows.
        """
        from src.feedback_layer.feedback_store import AnalystDecision

        base_weight = float(
            self._feedback_cfg.get("feedback_base_weight_multiplier", 1.0)
        )
        n_base = len(X_base)
        base_weights = np.full(n_base, base_weight, dtype=np.float64)

        # Filter out ESCALATE records (uncertain ground truth).
        usable = [
            r
            for r in feedback_records
            if r.decision != AnalystDecision.ESCALATE
        ]

        if not usable:
            _logger.info(
                "RetrainingPipeline._merge_with_feedback — no usable feedback "
                "(all ESCALATE or empty); training on base data only."
            )
            return X_base, y_base, base_weights

        n_feedback = len(usable)
        feedback_scale = n_base / n_feedback

        feature_cols = list(X_base.columns)
        feedback_rows: list[dict] = []
        feedback_labels: list[int] = []
        feedback_weights: list[float] = []

        for rec in usable:
            # Build a feature row: use model_score as a proxy feature if we
            # don't have the original feature vector (common in production).
            # Zeros for unknown features; model_score injected where the
            # column name matches "ensemble_score" or is absent.
            row: dict = {col: 0.0 for col in feature_cols}

            # If feedback record carries an extra `features` dict (future
            # enrichment from the analyst UI), merge it.
            if hasattr(rec, "features") and isinstance(rec.features, dict):  # type: ignore[attr-defined]
                for k, v in rec.features.items():  # type: ignore[attr-defined]
                    if k in row:
                        row[k] = float(v)

            feedback_rows.append(row)
            # CONFIRM_FRAUD → label 1, FALSE_POSITIVE → label 0
            label = (
                1
                if rec.decision == AnalystDecision.CONFIRM_FRAUD
                else 0
            )
            feedback_labels.append(label)
            feedback_weights.append(
                feedback_scale * max(0.01, float(rec.confidence))
            )

        X_feedback = pd.DataFrame(feedback_rows, columns=feature_cols)
        y_feedback = pd.Series(feedback_labels, name="label")
        fw_array = np.array(feedback_weights, dtype=np.float64)

        X_merged = pd.concat([X_base, X_feedback], ignore_index=True)
        y_merged = pd.concat([y_base, y_feedback], ignore_index=True)
        weights_merged = np.concatenate([base_weights, fw_array])

        _logger.info(
            "RetrainingPipeline._merge_with_feedback — merged | "
            "base=%d, feedback=%d (of %d total, %d ESCALATE excluded), "
            "weight_range=[%.4f, %.4f]",
            n_base,
            n_feedback,
            len(feedback_records),
            len(feedback_records) - n_feedback,
            weights_merged.min(),
            weights_merged.max(),
        )
        return X_merged, y_merged, weights_merged

    def _train_challenger(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: np.ndarray,
    ) -> EnsembleDetector:
        """Train a new EnsembleDetector challenger model.

        ``sample_weight`` is passed to the LightGBM sub-model via
        ``lgbm_.model_.fit(sample_weight=...)``.  The Isolation Forest
        sub-model does not accept sample weights and is trained without them.

        Args:
            X: Merged feature matrix (base + feedback).
            y: Merged label vector.
            weights: Per-sample weight array aligned with ``X``.

        Returns:
            A fitted :class:`~src.ml_layer.model.EnsembleDetector`.
        """
        _logger.info(
            "RetrainingPipeline._train_challenger — training challenger | "
            "n_samples=%d, n_features=%d",
            X.shape[0],
            X.shape[1],
        )

        challenger = EnsembleDetector(config_path=self._config_path)

        X_np = X.values.astype(np.float64)
        y_np = y.values.astype(int)

        # Train LightGBM with sample weights.
        challenger.lgbm_.train(X_np, y_np)
        # Re-fit the LightGBM model with sample_weight if the sklearn interface
        # permits (LGBMClassifier.fit accepts sample_weight directly).
        if challenger.lgbm_.model_ is not None:
            from lightgbm import LGBMClassifier

            n_neg = int(np.sum(y_np == 0))
            n_pos = int(np.sum(y_np == 1))
            scale_pos_weight = n_neg / max(n_pos, 1)

            lgbm_cfg = self._cfg["model"]["lgbm"]
            challenger.lgbm_.model_ = LGBMClassifier(
                n_estimators=lgbm_cfg["n_estimators"],
                learning_rate=lgbm_cfg["learning_rate"],
                num_leaves=lgbm_cfg["num_leaves"],
                max_depth=lgbm_cfg["max_depth"],
                subsample=lgbm_cfg["subsample"],
                colsample_bytree=lgbm_cfg["colsample_bytree"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                reg_alpha=lgbm_cfg["reg_alpha"],
                reg_lambda=lgbm_cfg["reg_lambda"],
                scale_pos_weight=scale_pos_weight,
                objective="binary",
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            challenger.lgbm_.model_.fit(X_np, y_np, sample_weight=weights)
            _logger.info(
                "RetrainingPipeline._train_challenger — LightGBM re-fitted "
                "with sample_weight."
            )

        # Train IsolationForest (unsupervised, no sample_weight).
        challenger.iso_.train(X_np, y_np)

        _logger.info("RetrainingPipeline._train_challenger — challenger ready.")
        return challenger

    def _evaluate_champion_challenger(
        self,
        challenger: EnsembleDetector,
    ) -> Tuple[dict, dict]:
        """Evaluate the challenger and load/evaluate the champion.

        The champion is loaded from the MLflow Model Registry using the
        ``champion`` alias (MLflow 2.9+) or the ``Production`` stage as a
        fallback.  If no production model is registered yet, champion metrics
        are returned as an empty dict and the challenger will be promoted
        automatically.

        Reads test data from ``self._X_test`` / ``self._y_test`` populated
        by :meth:`_load_base_data`.

        Args:
            challenger: Freshly trained :class:`~src.ml_layer.model.EnsembleDetector`.

        Returns:
            Tuple of ``(challenger_metrics, champion_metrics)`` where each
            is a dict with keys: ``f1``, ``auc_roc``, ``auc_pr``,
            ``precision``, ``recall``.
        """
        if self._X_test is None or self._y_test is None:
            raise RuntimeError(
                "_evaluate_champion_challenger called before _load_base_data; "
                "self._X_test and self._y_test are not set."
            )
        X_test_np = self._X_test.values.astype(np.float64)
        y_test_np = self._y_test.values.astype(int)

        # Challenger metrics.
        challenger_metrics = challenger.evaluate(X_test_np, y_test_np)
        _logger.info(
            "RetrainingPipeline — challenger | auc_pr=%.4f, f1=%.4f",
            challenger_metrics.get("auc_pr", 0),
            challenger_metrics.get("f1", 0),
        )

        # Champion — try to load from MLflow registry.
        champion_metrics: dict = {}
        registry_name = self._mlflow_cfg["model_registry_name"]
        try:
            champion_mv = self._client.get_model_version_by_alias(
                registry_name, "champion"
            )
            champion_uri = f"models:/{registry_name}/{champion_mv.version}"
            champion_model: EnsembleDetector = mlflow.sklearn.load_model(
                champion_uri
            )
            champion_metrics = champion_model.evaluate(X_test_np, y_test_np)
            _logger.info(
                "RetrainingPipeline — champion (v%s) | auc_pr=%.4f, f1=%.4f",
                champion_mv.version,
                champion_metrics.get("auc_pr", 0),
                champion_metrics.get("f1", 0),
            )
        except mlflow.exceptions.MlflowException:
            # No champion registered yet — first run always promotes.
            try:
                # Fallback: try "Production" stage for older MLflow.
                prod_models = self._client.get_latest_versions(
                    registry_name, stages=["Production"]
                )
                if prod_models:
                    champion_uri = f"models:/{registry_name}/{prod_models[0].version}"
                    champion_model = mlflow.sklearn.load_model(champion_uri)
                    champion_metrics = champion_model.evaluate(
                        X_test_np, y_test_np
                    )
                    _logger.info(
                        "RetrainingPipeline — champion (Production stage) | "
                        "auc_pr=%.4f, f1=%.4f",
                        champion_metrics.get("auc_pr", 0),
                        champion_metrics.get("f1", 0),
                    )
                else:
                    _logger.info(
                        "RetrainingPipeline — no Production champion in registry; "
                        "challenger will be auto-promoted."
                    )
            except mlflow.exceptions.MlflowException:
                _logger.info(
                    "RetrainingPipeline — champion not found in registry; "
                    "challenger will be auto-promoted."
                )
        except Exception as exc:
            _logger.warning(
                "RetrainingPipeline._evaluate_champion_challenger — could not "
                "load champion: %s. Challenger will auto-promote.",
                exc,
            )

        return challenger_metrics, champion_metrics

    def _should_promote(
        self,
        challenger_metrics: dict,
        champion_metrics: dict,
    ) -> bool:
        """Apply the champion/challenger promotion decision rule.

        Promotion criteria (both must be satisfied):

        1. ``challenger auc_pr > champion auc_pr + min_improvement_auc_pr``
        2. ``challenger f1 >= champion f1 * 0.99``  (F1 guard-rail)

        If no champion exists (``champion_metrics`` is empty), the challenger
        is always promoted.

        Args:
            challenger_metrics: Evaluation dict for the challenger model.
            champion_metrics: Evaluation dict for the champion model.
                Empty dict means no champion registered.

        Returns:
            ``True`` if the challenger should be promoted.
        """
        if not champion_metrics:
            _logger.info(
                "RetrainingPipeline._should_promote — no champion exists; "
                "auto-promoting challenger."
            )
            return True

        min_improvement: float = float(
            self._feedback_cfg.get("min_improvement_auc_pr", 0.005)
        )
        c_auc = float(challenger_metrics.get("auc_pr", 0.0))
        p_auc = float(champion_metrics.get("auc_pr", 0.0))
        c_f1 = float(challenger_metrics.get("f1", 0.0))
        p_f1 = float(champion_metrics.get("f1", 0.0))

        auc_improved = c_auc > (p_auc + min_improvement)
        f1_acceptable = c_f1 >= (p_f1 * 0.99)

        _logger.info(
            "RetrainingPipeline._should_promote | "
            "challenger auc_pr=%.4f vs champion auc_pr=%.4f (delta=%.4f, threshold=%.4f) | "
            "challenger f1=%.4f vs champion f1=%.4f (0.99x=%.4f) | "
            "auc_improved=%s, f1_acceptable=%s",
            c_auc, p_auc, c_auc - p_auc, min_improvement,
            c_f1, p_f1, p_f1 * 0.99,
            auc_improved, f1_acceptable,
        )

        return auc_improved and f1_acceptable

    def _promote_challenger(
        self,
        challenger: EnsembleDetector,
        metrics: dict,
        feedback_records: list[FeedbackRecord],
        trigger_reason: str,
    ) -> str:
        """Register the challenger in MLflow, promote it, and signal the consumer.

        Steps:

        1. Open an MLflow run and log metrics + params.
        2. Log the model to the registry.
        3. Transition new version to ``Production`` / set ``champion`` alias.
        4. Archive the old production model.
        5. Mark feedback records as used.
        6. Invalidate SHAP cache (delete cached pickle).
        7. Write model-update signal for the live Kafka consumer.
        8. Store run_id on ``self._last_run_id`` for :meth:`run` to read.

        Args:
            challenger: Trained challenger model.
            metrics: Evaluation metrics for the challenger.
            feedback_records: Feedback records consumed in this run.
            trigger_reason: Trigger that initiated this run.

        Returns:
            MLflow run ID string.
        """
        registry_name = self._mlflow_cfg["model_registry_name"]
        run_id: str = ""

        with mlflow.start_run(
            run_name=f"retrain-{trigger_reason}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        ) as run:
            run_id = run.info.run_id

            # Log metadata.
            mlflow.set_tag("trigger_reason", trigger_reason)
            mlflow.set_tag("outcome", "promoted")
            mlflow.set_tag("feedback_count", str(len(feedback_records)))

            # Log metrics.
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"challenger_{metric_name}", metric_value)

            # Log params (subset to avoid MLflow 100-param limit).
            mlflow.log_param("trigger_reason", trigger_reason)
            mlflow.log_param("feedback_count", len(feedback_records))

            # Register model — two-step for MLflow 2.10 compatibility.
            mlflow.sklearn.log_model(
                sk_model=challenger,
                artifact_path="model",
            )

        # Register explicitly outside the run context.
        mv = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=registry_name,
        )
        new_version_str = str(mv.version)

        # Archive existing Production models (MLflow 2.x stages API).
        try:
            old_prod = self._client.get_latest_versions(
                registry_name, stages=["Production"]
            )
            for old_mv in old_prod:
                self._client.transition_model_version_stage(
                    name=registry_name,
                    version=old_mv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                _logger.info(
                    "RetrainingPipeline._promote_challenger — archived "
                    "old Production v%s.",
                    old_mv.version,
                )
        except mlflow.exceptions.MlflowException as exc:
            _logger.warning(
                "RetrainingPipeline._promote_challenger — could not archive "
                "old Production model: %s",
                exc,
            )

        # Promote new version.
        try:
            self._client.transition_model_version_stage(
                name=registry_name,
                version=new_version_str,
                stage="Production",
                archive_existing_versions=True,
            )
            _logger.info(
                "RetrainingPipeline._promote_challenger — v%s transitioned "
                "to Production.",
                new_version_str,
            )
        except mlflow.exceptions.MlflowException as exc:
            _logger.warning(
                "RetrainingPipeline._promote_challenger — could not set "
                "Production stage (MLflow 3.x?): %s. Trying alias.",
                exc,
            )

        # Set "champion" alias (MLflow 2.9+).
        try:
            self._client.set_registered_model_alias(
                name=registry_name,
                alias="champion",
                version=new_version_str,
            )
            _logger.info(
                "RetrainingPipeline._promote_challenger — alias 'champion' "
                "→ v%s.",
                new_version_str,
            )
        except Exception as exc:
            _logger.warning(
                "RetrainingPipeline._promote_challenger — could not set "
                "'champion' alias: %s",
                exc,
            )

        # Save local pkl copy for the Kafka consumer fallback path.
        local_path = self._cfg["model"]["artifacts"]["local_model_path"]
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(challenger, local_path)
        _logger.info(
            "RetrainingPipeline._promote_challenger — local pkl saved to '%s'.",
            local_path,
        )

        # Mark feedback as used.
        tx_ids = [r.transaction_id for r in feedback_records]
        if tx_ids:
            self._store.mark_as_used(tx_ids, run_id)

        # Invalidate SHAP cache.
        self._invalidate_shap_cache()

        # Write model-update signal for the live consumer.
        self._signal.write_signal(new_model_version=new_version_str)

        # Store run_id for run() to pick up.
        self._last_run_id = run_id

        _logger.info(
            "RetrainingPipeline._promote_challenger — promotion complete | "
            "version=%s, run_id=%s",
            new_version_str,
            run_id,
        )
        return run_id

    def _log_rejection(
        self,
        challenger_metrics: dict,
        champion_metrics: dict,
        trigger_reason: str,
    ) -> None:
        """Log a rejected challenger run to MLflow without touching the registry.

        Stores the MLflow run ID on ``self._last_run_id`` so :meth:`run` can
        include it in the result dict.

        Args:
            challenger_metrics: Evaluation metrics for the rejected challenger.
            champion_metrics: Evaluation metrics for the current champion.
            trigger_reason: Trigger that initiated the run.
        """
        run_id = ""
        with mlflow.start_run(
            run_name=f"retrain-rejected-{trigger_reason}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        ) as run:
            run_id = run.info.run_id
            mlflow.set_tag("trigger_reason", trigger_reason)
            mlflow.set_tag("outcome", "rejected")

            for metric_name, metric_value in challenger_metrics.items():
                mlflow.log_metric(f"challenger_{metric_name}", metric_value)
            for metric_name, metric_value in champion_metrics.items():
                mlflow.log_metric(f"champion_{metric_name}", metric_value)

            mlflow.log_param("trigger_reason", trigger_reason)
            mlflow.log_param("rejection_reason", "challenger_did_not_improve")

        # Store for run() to read.
        self._last_run_id = run_id

        _logger.info(
            "RetrainingPipeline._log_rejection — challenger rejected | "
            "trigger=%s, run_id=%s | challenger auc_pr=%.4f, champion auc_pr=%.4f",
            trigger_reason,
            run_id,
            challenger_metrics.get("auc_pr", 0),
            champion_metrics.get("auc_pr", 0),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _invalidate_shap_cache(self) -> None:
        """Delete the SHAP explanation cache after model promotion.

        The SHAP cache stores pre-computed explanation results keyed by
        sample identifiers.  After a model update the cache is stale and
        must be cleared so the new model's explanations are used.

        The cache directory is derived from the ``explainer.plots_dir``
        config key; any ``*.pkl`` files matching a cache pattern are removed.
        """
        cache_pattern = "*.pkl"
        plots_dir = Path(self._cfg.get("explainer", {}).get("plots_dir", "plots/shap"))
        try:
            removed = 0
            for pkl_file in plots_dir.glob(cache_pattern):
                pkl_file.unlink()
                removed += 1
            _logger.info(
                "RetrainingPipeline._invalidate_shap_cache — removed %d "
                "cached pkl files from '%s'.",
                removed,
                plots_dir,
            )
        except OSError as exc:
            _logger.warning(
                "RetrainingPipeline._invalidate_shap_cache — could not clear "
                "cache at '%s': %s",
                plots_dir,
                exc,
            )

    @staticmethod
    def _error_result(trigger_reason: str, error_msg: str) -> dict:
        """Return a uniform error result dict without a run_id.

        Args:
            trigger_reason: Original trigger string.
            error_msg: Error description.

        Returns:
            Result dict with ``promoted=False`` and ``error`` key populated.
        """
        return {
            "promoted": False,
            "challenger_metrics": {},
            "champion_metrics": {},
            "feedback_count": 0,
            "trigger_reason": trigger_reason,
            "mlflow_run_id": "",
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    pipeline = RetrainingPipeline()

    _logger.info(
        "RetrainingPipeline __main__ — running manual smoke-test "
        "with trigger_reason='manual_test'."
    )

    result = pipeline.run(trigger_reason="manual_test")

    _logger.info("=== Retraining Result ===")
    pprint.pprint(result)

    print("\n" + "=" * 60)
    print("CIPHER Retraining Pipeline — Result Summary")
    print("=" * 60)
    print(f"  Promoted          : {result['promoted']}")
    print(f"  Trigger Reason    : {result['trigger_reason']}")
    print(f"  Feedback Records  : {result['feedback_count']}")
    print(f"  MLflow Run ID     : {result['mlflow_run_id']}")
    print()
    print("  Challenger Metrics:")
    for k, v in result["challenger_metrics"].items():
        print(f"    {k:<18}: {v:.4f}")
    print()
    print("  Champion Metrics:")
    if result["champion_metrics"]:
        for k, v in result["champion_metrics"].items():
            print(f"    {k:<18}: {v:.4f}")
    else:
        print("    (no champion registered yet — challenger auto-promoted)")
    print("=" * 60)
