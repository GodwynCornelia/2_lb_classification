"""Audit the full additional assignment without repeating neural network training."""
import base64
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import joblib
import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from notebook_quality import validate_notebook_text

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "additional" / "artifacts"


def measured(truth, proba, predictions=None):
    # Keras softmax is computed in float32 and stored as float64. Check its actual
    # rounding error before silencing sklearn's much stricter float64 sum warning.
    assert np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1, atol=2e-6, rtol=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The y_prob values do not sum to one.*", category=UserWarning)
        loss = log_loss(truth, proba, labels=[0,1,2,3])
    pred = proba.argmax(axis=1) if predictions is None else predictions
    return {"accuracy": accuracy_score(truth, pred), "loss": loss,
        "precision_macro": precision_score(truth, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(truth, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(truth, pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(label_binarize(truth, classes=[0,1,2,3]), proba,
                                          multi_class="ovr", average="macro")}


def close_metrics(actual, expected):
    for metric, value in actual.items():
        assert np.isclose(value, expected[metric], atol=1e-7, rtol=1e-7), (metric, value, expected[metric])


def main():
    report = json.loads((ARTIFACTS / "experiment_report.json").read_text(encoding="utf-8"))
    notebook = nbformat.read(ROOT / "Lab2_Additional_TensorFlow_CV.ipynb", as_version=4)
    nbformat.validate(notebook)
    analyses = json.loads((ARTIFACTS / "analyses.json").read_text(encoding="utf-8"))
    validate_notebook_text(notebook, expected_analyses=analyses)
    code_count, png_count = 0, 0
    for i, cell in enumerate(notebook.cells):
        assert cell.source.strip(), i
        if cell.cell_type != "code":
            continue
        code_count += 1
        assert cell.execution_count == code_count
        compile(cell.source, f"cell_{i}", "exec")
        following = notebook.cells[i+1]
        assert following.cell_type == "markdown" and len(following.source) > 200
        assert following.metadata["additional_analysis"] == cell.metadata["additional_step"]
        for output in cell.outputs:
            assert output.output_type != "error"
            if "image/png" in output.get("data", {}):
                with Image.open(BytesIO(base64.b64decode(output.data["image/png"]))) as im:
                    assert im.width > 300 and im.height > 200
                    im.verify()
                png_count += 1
    assert code_count == 14 and png_count == 6, (code_count, png_count)
    import generate_additional_notebook as generator
    assert [c.source for c in generator.cells if c.cell_type == "code"] == [
        c.source for c in notebook.cells if c.cell_type == "code"]

    raw = ROOT / "data" / "raw" / "train.csv"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == report["dataset_sha256"]
    data = pd.read_csv(raw)
    X, y = data.drop(columns="price_range"), data.price_range
    train, test, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    manifest = json.loads((ARTIFACTS / "split_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["outer_train_indices"]) == set(train.index)
    assert set(manifest["outer_test_indices"]) == set(test.index)
    manifests = {fold["fold"]: fold for fold in manifest["folds"]}
    all_validation = []
    for fold in manifests.values():
        fit_idx, val_idx = set(fold["train_indices"]), set(fold["validation_indices"])
        assert len(fit_idx) == 1280 and len(val_idx) == 320
        assert fit_idx.isdisjoint(val_idx) and fit_idx | val_idx == set(train.index)
        assert (fit_idx | val_idx).isdisjoint(test.index)
        assert y.loc[sorted(val_idx)].value_counts().sort_index().tolist() == [80]*4
        all_validation.extend(val_idx)
    assert len(all_validation) == len(set(all_validation)) == 1600

    results = pd.read_csv(ARTIFACTS / "cv_results.csv")
    folds = pd.read_csv(ARTIFACTS / "fold_metrics.csv")
    predictions = pd.read_csv(ARTIFACTS / "oof_predictions.csv")
    audit = json.loads((ARTIFACTS / "fit_audit.json").read_text(encoding="utf-8"))
    assert len(results) == 9 and len(folds) == 45 and len(audit) == 46
    assert len(predictions) == 9*1600
    assert set(results.learning_rate) == {0.0001, 0.001, 0.01}
    assert set(results.architecture) == {"shallow", "deep", "regularized"}
    assert not results.duplicated(["architecture", "learning_rate"]).any()
    assert results.selected.sum() == 1
    assert results.loc[results.selected, "config"].iloc[0] == report["best_config"]
    for cfg, group in predictions.groupby("config"):
        assert len(group) == 1600 and group.row_index.nunique() == 1600
        assert set(group.row_index) == set(train.index)
        for fold_id, rows in group.groupby("fold"):
            assert set(rows.row_index) == set(manifests[fold_id]["validation_indices"])
            assert np.array_equal(rows.truth.to_numpy(), y.loc[rows.row_index].to_numpy())
            proba = rows[[f"p{c}" for c in range(4)]].to_numpy()
            assert np.isfinite(proba).all() and (proba >= 0).all()
            assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
            assert np.array_equal(rows.prediction, proba.argmax(axis=1))
            expected = folds.loc[(folds.config == cfg) & (folds.fold == fold_id)].iloc[0]
            close_metrics(measured(rows.truth, proba), expected)
        metrics = folds[folds.config == cfg]
        row = results[results.config == cfg].iloc[0]
        for metric in ["accuracy", "loss"]:
            assert np.isclose(row[f"mean_{metric}"], metrics[metric].mean(), atol=1e-10)
            assert np.isclose(row[f"std_{metric}"], metrics[metric].std(ddof=0), atol=1e-10)
    ordered = results.sort_values(["mean_accuracy", "mean_loss", "std_accuracy"], ascending=[False,True,True])
    assert ordered.iloc[0].config == report["best_config"]

    with tempfile.TemporaryDirectory(prefix="lab2_additional_audit_") as temporary:
        for record in audit:
            fold_id = record["fold"]
            fit_indices = manifests[fold_id]["train_indices"] if fold_id else manifest["outer_train_indices"]
            preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            preprocessor.fit(X.loc[fit_indices])
            assert np.allclose(preprocessor["scaler"].mean_, record["preprocessor_mean"], atol=1e-12)
            assert np.allclose(preprocessor["scaler"].scale_, record["preprocessor_scale"], atol=1e-12)
            assert record["seed"] == 42+fold_id and record["epochs"] == report["epochs"] == 60
            run = f"fold_{fold_id}" if fold_id else "refit"
            history = pd.read_csv(ARTIFACTS / f"history_{record['config']}_{run}.csv")
            assert len(history) == 60
            staging = Path(temporary) / record["config"] / run
            shutil.copytree(ROOT / record["log_path"], staging)
            for partition, prefix in [("train", ""), ("validation", "val_")]:
                if not fold_id and partition == "validation":
                    continue
                events = EventAccumulator(str(staging / partition), size_guidance={"tensors":0})
                events.Reload()
                for metric in ["loss", "accuracy"]:
                    values = events.Tensors(f"epoch_{metric}")
                    assert len(values) == 60
                    assert [event.step for event in values] == list(range(60))
                    scalar = [tf.make_ndarray(event.tensor_proto).item() for event in values]
                    assert np.allclose(scalar, history[prefix+metric], atol=1e-7)
                if partition == "train":
                    assert any("histogram" in key for key in events.Tags()["tensors"])

    baseline = joblib.load(ARTIFACTS / "baseline_logistic.joblib")
    baseline_proba = baseline.predict_proba(test)
    close_metrics(measured(test_y, baseline_proba, baseline.predict(test)), report["baseline_test"])
    saved = joblib.load(ARTIFACTS / "neural_preprocessing.joblib")
    model = tf.keras.models.load_model(ARTIFACTS / "best_model.keras")
    features = saved["preprocessor"].transform(test[saved["features"]]).astype("float32")
    proba = model(tf.convert_to_tensor(features), training=False).numpy().astype("float64")
    close_metrics(measured(test_y, proba), report["neural_test"])
    saved_predictions = pd.read_csv(ARTIFACTS / "test_predictions.csv")
    assert np.array_equal(saved_predictions.row_index, test.index)
    assert np.allclose(saved_predictions[[f"neural_p{c}" for c in range(4)]], proba, atol=1e-7)
    assert np.isclose(100*(report["neural_test"]["accuracy"]-report["baseline_test"]["accuracy"]),
                      report["test_accuracy_delta_pp"])
    print(f"OK: {code_count} executed code cells, {png_count} figures; "
          "45 OOF fold evaluations, 46 preprocessing audits and TensorBoard histories verified; "
          "all 12 final metrics reproduced from saved models.")


if __name__ == "__main__":
    main()
