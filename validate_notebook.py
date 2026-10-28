"""Validate the delivered notebook, event logs, and persisted inference artifacts."""
import base64
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import joblib
import kagglehub
import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="Validate bundled TensorBoard logs")
    args = parser.parse_args()
    nb = nbformat.read(ROOT / "Lab2_Classification_NeuralNetworks.ipynb", as_version=4)
    nbformat.validate(nb)
    code_cells = 0
    images = 0
    for i, cell in enumerate(nb.cells):
        assert cell.source.strip(), f"Empty cell {i}"
        if cell.cell_type != "code":
            continue
        code_cells += 1
        compile(cell.source, f"cell_{i}", "exec")
        assert cell.execution_count == code_cells, f"Execution order at {i}"
        assert nb.cells[i+1].cell_type == "markdown", f"Missing interpretation at {i}"
        assert nb.cells[i+1].metadata["lab2_analysis"] == cell.metadata["lab2_step"]
        assert len(nb.cells[i+1].source) > 200
        for output in cell.outputs:
            assert output.output_type != "error", output
            if "image/png" in output.get("data", {}):
                with Image.open(BytesIO(base64.b64decode(output.data["image/png"]))) as im:
                    assert im.width > 300 and im.height > 200
                    im.verify()
                images += 1
    assert code_cells == 26, code_cells
    assert images >= 16, images
    report = json.loads((ROOT / "artifacts/experiment_report.json").read_text(encoding="utf-8"))
    assert len(report["final"]) == 6
    assert len(report["best_params"]) == 5
    assert len(report["nn_logs"]) == 5
    with tempfile.TemporaryDirectory(prefix="lab2_validate_") as staging:
        for run, log_dir in report["nn_logs"].items():
            source_events = Path(log_dir)
            if args.snapshot or not source_events.is_dir():
                source_events = ROOT / "artifacts" / "tensorboard" / run
            event_path = Path(staging) / run
            shutil.copytree(source_events, event_path)
            train = EventAccumulator(str(event_path / "train"), size_guidance={"tensors": 0})
            val = EventAccumulator(str(event_path / "validation"), size_guidance={"tensors": 0})
            train.Reload()
            val.Reload()
            assert "epoch_loss" in train.Tags()["tensors"], run
            assert "epoch_accuracy" in val.Tags()["tensors"], run
            assert any("histogram" in key for key in train.Tags()["tensors"]), train.Tags()

    data_path = ROOT / "data" / "raw"
    if not (data_path / "train.csv").is_file():
        data_path = Path(kagglehub.dataset_download("iabhishekofficial/mobile-price-classification"))
    import hashlib
    assert hashlib.sha256((data_path / "train.csv").read_bytes()).hexdigest() == report["dataset_sha256"]
    data = pd.read_csv(data_path / "train.csv")
    preprocess = joblib.load(ROOT / "artifacts/neural_preprocessing.joblib")
    X = data[preprocess["features"]]
    y = data.price_range
    _, test, _, truth = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    classifiers = joblib.load(ROOT / "artifacts/classical_pipelines.joblib")
    all_predictions = {n: (m.predict(test), m.predict_proba(test)) for n,m in classifiers.items()}
    neural = tf.keras.models.load_model(ROOT / "artifacts/best_neural_network.keras")
    features = preprocess["scaler"].transform(preprocess["imputer"].transform(test))
    probabilities = neural(tf.convert_to_tensor(features, dtype=tf.float32), training=False).numpy()
    all_predictions["TensorFlow MLP"] = (probabilities.argmax(axis=1), probabilities)
    for name, (pred, proba) in all_predictions.items():
        computed = {
            "Accuracy": accuracy_score(truth, pred),
            "Precision": precision_score(truth, pred, average="macro", zero_division=0),
            "Recall": recall_score(truth, pred, average="macro", zero_division=0),
            "F1-macro": f1_score(truth, pred, average="macro", zero_division=0),
            "ROC-AUC": roc_auc_score(label_binarize(truth, classes=[0,1,2,3]), proba, average="macro"),
        }
        for metric, value in computed.items():
            assert np.isclose(value, report["final"][name][metric], atol=1e-7), (name, metric, value)
    print(f"OK: {code_cells} executed code cells, {images} PNG outputs, 5 TensorBoard runs; "
          "all 30 final metrics reproduced from saved models.")


if __name__ == "__main__":
    main()
