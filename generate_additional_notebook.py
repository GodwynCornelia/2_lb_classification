"""Execute the additional TensorFlow assignment: 3 architectures x 3 rates x 5 folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Lab2_Additional_TensorFlow_CV.ipynb"
MIME = "application/vnd.lab2.additional-analysis+json"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def step(key, code):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(code).strip(), metadata={"additional_step": key}))
    cells.append(nbf.v4.new_markdown_cell("", metadata={"additional_analysis": key}))


md(r"""
# Дополнительное задание. TensorFlow: 5-fold CV, Grid Search и TensorBoard

**Цель:** воспроизводимо сравнить девять конфигураций нейросети для четырёхклассовой
классификации и проверить результат относительно фиксированной baseline-модели.
Используется Mobile Price Classification из лабораторной работы 2.

План: первичный анализ → baseline → стратифицированная 5-fold CV → полный перебор
3 архитектур × 3 learning rate → TensorBoard и статические кривые → сравнение
качества, loss, времени и устойчивости → refit победителя и тестовая оценка.

Все конфигурации обучаются ровно **60 эпох**, batch size **64**. Валидационные метрики
записываются, но EarlyStopping, подбор эпох и восстановление весов по проверочному
фолду не применяются. Оценка каждого фолда соответствует последней эпохе.
Выбор конфигурации использует только CV; после выбора GridSearchCV выполняет один
refit на всей обучающей части. Baseline — LogisticRegression с фиксированными параметрами.

Сохраняется внешнее разбиение 80/20 с seed 42 из основной работы. Его тест уже
анализировался в предыдущей работе: это повторное учебное сравнение на известном
наборе, а не новая внешняя проверка. В дополнительной работе тест не участвует
в поиске параметров, построении фолдов и остановке обучения.

Текстовые выводы сформированы из реальных вычислений. Команда
`python generate_additional_notebook.py` повторяет расчёт и обновляет Markdown.
""")

step("setup", r'''
RANDOM_STATE = 42
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import random
import warnings
warnings.filterwarnings("ignore")
import gc
import hashlib
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sklearn
import kagglehub
import joblib
from IPython.display import display
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import (accuracy_score, log_loss, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve, auc,
                            confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
tf.get_logger().setLevel("ERROR")
thread_limit = threadpool_limits(limits=1)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
pd.set_option("display.max_columns", 20)
EPOCHS, BATCH_SIZE = 60, 64
CLASSES = np.array([0, 1, 2, 3])
ARTIFACTS = Path("additional") / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
TB_WORKDIR = Path.cwd()
if os.name == "nt" and not str(TB_WORKDIR).isascii():
    TB_WORKDIR = Path(tempfile.mkdtemp(prefix="lab2_additional_tb_"))
    if not str(TB_WORKDIR).isascii():
        raise RuntimeError("TEMP должен иметь ASCII-путь для файлового backend TensorFlow.")
TB_ROOT = TB_WORKDIR / "logs" / "additional" / RUN_ID
ANALYSIS, FIT_RECORDS, HISTORIES, OOF_PREDICTIONS = {}, [], {}, []
versions = {"Python": platform.python_version(), "NumPy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "TensorFlow": tf.__version__, "Keras": tf.keras.__version__}
display(pd.Series(versions, name="Версия").to_frame())

def show(name):
    plt.tight_layout()
    plt.savefig(ARTIFACTS / f"{name}.png", bbox_inches="tight")
    plt.show()

def dump_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+"\n", encoding="utf-8")

def metric_row(truth, pred, proba):
    assert proba.shape == (len(truth), 4) and np.isfinite(proba).all()
    assert np.all(proba >= 0) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    return {"accuracy": accuracy_score(truth, pred), "loss": log_loss(truth, proba, labels=CLASSES),
        "precision_macro": precision_score(truth, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(truth, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(truth, pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(label_binarize(truth, classes=CLASSES), proba,
                                          multi_class="ovr", average="macro")}

ANALYSIS["setup"] = (
    f"Среда: Python {versions['Python']}, TensorFlow {versions['TensorFlow']}, "
    f"Keras {versions['Keras']}, scikit-learn {versions['scikit-learn']}. "
    f"Базовый seed={RANDOM_STATE}, бюджет={EPOCHS} эпох, batch_size={BATCH_SIZE}. "
    "Для одинакового фолда все конфигурации получают одинаковый seed, для финального "
    "refit — 42. Включены детерминированные операции и один вычислительный поток. "
    "Результаты воспроизводимы в зафиксированной среде; между платформами побитовая "
    "идентичность вычислений не гарантируется.")
''')

md("## 1. Загрузка, качество данных и баланс классов")
step("data", r'''
csv_path = Path("data") / "raw" / "train.csv"
if not csv_path.is_file():
    csv_path = Path(kagglehub.dataset_download("iabhishekofficial/mobile-price-classification")) / "train.csv"
df = pd.read_csv(csv_path)
FEATURES = ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height", "px_width",
    "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"]
assert df.shape == (2000, 21) and set(df.columns) == set(FEATURES+["price_range"])
assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
assert not np.isinf(df.to_numpy()).any()
assert df.price_range.notna().all() and set(df.price_range) == set(CLASSES)
dataset_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
display(df.head())
df.info()
display(df.describe().T)
quality = pd.DataFrame({"dtype": df.dtypes.astype(str), "NaN": df.isnull().sum()})
duplicates = int(df.duplicated().sum())
display(quality)
print("Полных дубликатов:", duplicates)
assert duplicates == 0, "Перед разбиением требуется обработать изменившийся набор дубликатов."
ANALYSIS["data"] = (
    f"Загружено {len(df)} строк и {len(FEATURES)} входных признаков. "
    f"Пропусков во всех 21 столбце: {int(df.isnull().sum().sum())}; дубликатов: {duplicates}. "
    f"Типов float: {sum(pd.api.types.is_float_dtype(t) for t in df.dtypes)}, "
    f"целочисленных столбцов: {sum(pd.api.types.is_integer_dtype(t) for t in df.dtypes)}. "
    "Все входы числовые; бинарные признаки уже представлены кодами 0/1, "
    "OneHotEncoder и кодирование строковых категорий не требуются. Медианный импутер "
    "оставлен как защита, но искусственно создавать пропуски не нужно. "
    f"SHA-256: `{dataset_sha256}`.")
''')

step("eda", r'''
counts = df.price_range.value_counts().sort_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts.plot.bar(ax=axes[0], rot=0, color=sns.color_palette("muted", 4))
axes[0].set(title="Распределение ценовых классов", xlabel="Класс", ylabel="Число объектов")
for i, count in enumerate(counts):
    axes[0].text(i, count+5, str(count), ha="center")
sns.boxplot(data=df, x="price_range", y="ram", hue="price_range", legend=False, ax=axes[1])
axes[1].set(title="RAM по ценовым классам", xlabel="Класс", ylabel="RAM, МБ")
show("data_overview")
ram_corr = float(df.corr().loc["ram", "price_range"])
ANALYSIS["eda"] = (
    f"Численности классов 0–3 равны {counts.tolist()}, "
    f"доля каждого класса {counts.iloc[0]/len(df):.2%}. "
    f"Точность константного классификатора составляет {counts.max()/len(df):.4f}. "
    "Весовые поправки классов и ресемплинг не требуются. "
    f"Корреляция Пирсона RAM с номером ценового сегмента r={ram_corr:.4f}; "
    "она описывает порядковую ассоциацию и не доказывает причинность. "
    f"RAM меняется от {df.ram.min()} до {df.ram.max()}, тогда как бинарные признаки "
    "лежат в {0,1}: стандартизация устраняет несопоставимость численных масштабов. "
    "Все 20 признаков сохранены; EDA не используется для отбора входов или изменения сетки.")
''')

step("split", r'''
X, y = df[FEATURES], df.price_range.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
assert set(X_train.index).isdisjoint(X_test.index)
display(pd.DataFrame({"train": y_train.value_counts().sort_index(),
                      "test": y_test.value_counts().sort_index()}))
ANALYSIS["split"] = (
    f"Внешний train содержит {len(X_train)} объектов, test — {len(X_test)}. "
    f"По классам в train: {y_train.value_counts().sort_index().tolist()}, "
    f"в test: {y_test.value_counts().sort_index().tolist()}. "
    "Сначала выполнено разбиение сырых данных, затем каждый обучающий процесс "
    "самостоятельно оценивает медианы, средние и стандартные отклонения. "
    "Ни скейлер, ни импутер не обучаются на всей таблице до CV. Внешнее разбиение "
    "совпадает с основной лабораторной; его результаты уже были известны ранее. "
    "В текущем поиске доступ к тесту не используется для выбора гиперпараметров.")
''')

md("## 2. Baseline: логистическая регрессия")
step("baseline", r'''
baseline_model = Pipeline([("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE))])
baseline_start = time.perf_counter()
baseline_model.fit(X_train, y_train)
baseline_seconds = time.perf_counter()-baseline_start
baseline_proba = baseline_model.predict_proba(X_test)
baseline_pred = baseline_model.predict(X_test)
baseline_metrics = metric_row(y_test, baseline_pred, baseline_proba)
display(pd.DataFrame([baseline_metrics], index=["LogisticRegression"]).round(5))
joblib.dump(baseline_model, ARTIFACTS / "baseline_logistic.joblib")
ANALYSIS["baseline"] = (
    f"Фиксированная LogisticRegression: Accuracy={baseline_metrics['accuracy']:.4f}, "
    f"loss={baseline_metrics['loss']:.4f}, F1-macro={baseline_metrics['f1_macro']:.4f}, "
    f"ROC-AUC={baseline_metrics['roc_auc_ovr_macro']:.4f}; обучение заняло {baseline_seconds:.3f} с. "
    "Модель задаёт линейные границы в стандартизованном пространстве и служит "
    "практической точкой отсчёта для более гибкой MLP. C=1 и max_iter=2000 "
    "заданы заранее и не подбирались по тесту. Для нейросети требуется показать "
    "фактический эффект настройки; превосходство над линейным baseline не предполагается.")
''')

md("## 3. Пять стратифицированных фолдов")
step("cv", r'''
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
CV_SPLITS = list(cv.split(X_train, y_train))
FOLD_MONITORS, split_manifest, fold_rows = {}, [], []
for fold, (fit_pos, val_pos) in enumerate(CV_SPLITS, start=1):
    fit_index = X_train.iloc[fit_pos].index
    val_index = X_train.iloc[val_pos].index
    assert set(fit_index).isdisjoint(val_index)
    assert set(fit_index).isdisjoint(X_test.index)
    assert set(val_index).isdisjoint(X_test.index)
    # Registry identifies a fold from its training indices. Validation rows are used
    # only by Keras' validation pass for monitoring; no callback changes the weights.
    FOLD_MONITORS[frozenset(fit_index)] = (fold, X_train.iloc[val_pos], y_train.iloc[val_pos])
    split_manifest.append({"fold": fold, "train_indices": fit_index.tolist(),
        "validation_indices": val_index.tolist()})
    fold_rows.append({"fold": fold, "train": len(fit_pos), "validation": len(val_pos),
        **{f"validation_class_{c}": int((y_train.iloc[val_pos] == c).sum()) for c in CLASSES}})
display(pd.DataFrame(fold_rows).set_index("fold"))
dump_json(ARTIFACTS / "split_manifest.json", {"random_state": RANDOM_STATE,
    "outer_train_indices": X_train.index.tolist(), "outer_test_indices": X_test.index.tolist(),
    "folds": split_manifest})
ANALYSIS["cv"] = (
    f"Организованы {len(CV_SPLITS)} общих фолдов: каждый содержит 1280 train-объектов "
    "и 320 validation-объектов, по 80 наблюдений каждого проверочного класса. "
    "Для каждой конфигурации объект внешнего train ровно один раз попадает в validation. "
    "Одинаковые фолды и seed по номеру фолда делают сравнение конфигураций парным. "
    "Проверены непересечение train/validation и отсутствие всех 400 test-объектов в CV. "
    "Реестр FOLD_MONITORS передаёт проверочные данные только для отображения кривых; "
    "обучение длится фиксированные 60 эпох, градиенты по validation не вычисляются.")
''')

md("## 4. TensorFlow-модель и полный Grid Search")
step("network", r'''
ARCHITECTURES = {
    "shallow": {"units": [32], "regularized": False},
    "deep": {"units": [128, 64, 32], "regularized": False},
    "regularized": {"units": [128, 64, 32], "regularized": True}}
LEARNING_RATES = [0.0001, 0.001, 0.01]

def config_id(architecture, learning_rate):
    return f"{architecture}_lr_{learning_rate:g}"

def make_dataset(features, labels, training=False, seed=RANDOM_STATE):
    ds = tf.data.Dataset.from_tensor_slices((np.asarray(features, dtype="float32"),
                                           np.asarray(labels, dtype="int32")))
    if training:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    return ds.batch(BATCH_SIZE).with_options(options).prefetch(1)

def build_network(architecture, learning_rate, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    spec = ARCHITECTURES[architecture]
    layers = [tf.keras.layers.Input(shape=(len(FEATURES),))]
    for units in spec["units"]:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        if spec["regularized"]:
            layers.extend([tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.3)])
    layers.append(tf.keras.layers.Dense(4, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class TensorFlowClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable sklearn estimator; all preprocessing is fitted inside fit()."""
    def __init__(self, architecture="shallow", learning_rate=0.001, epochs=EPOCHS,
                 random_state=RANDOM_STATE):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        assert np.array_equal(self.classes_, CLASSES)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        monitor = FOLD_MONITORS.get(frozenset(X.index))
        fold = monitor[0] if monitor else 0
        if monitor is None:
            assert set(X.index) == set(X_train.index), "Unknown training partition"
        seed = self.random_state + fold
        self.preprocessor_ = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())])
        fit_features = self.preprocessor_.fit_transform(X).astype("float32")
        validation_data = None
        if monitor is not None:
            _, val_X, val_y = monitor
            validation_data = make_dataset(self.preprocessor_.transform(val_X), val_y)
        self.model_ = build_network(self.architecture, self.learning_rate, seed)
        cfg = config_id(self.architecture, self.learning_rate)
        run = f"fold_{fold}" if fold else "refit"
        log_dir = TB_ROOT / cfg / run
        callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir.as_posix(), histogram_freq=1)
        start = time.perf_counter()
        history = self.model_.fit(make_dataset(fit_features, y, training=True, seed=seed),
            validation_data=validation_data, epochs=self.epochs, callbacks=[callback], verbose=0)
        seconds = time.perf_counter()-start
        self.history_ = history.history
        self.training_seconds_ = seconds
        archive = ARTIFACTS / "tensorboard" / cfg / run
        shutil.copytree(log_dir, archive, dirs_exist_ok=True)
        HISTORIES[f"{cfg}/{run}"] = history.history
        pd.DataFrame(history.history).rename_axis("epoch_zero_based").to_csv(
            ARTIFACTS / f"history_{cfg}_{run}.csv")
        record = {"config": cfg, "architecture": self.architecture, "learning_rate": self.learning_rate,
            "fold": fold, "seed": seed, "epochs": self.epochs, "n_train": len(X),
            "training_seconds": seconds, "log_path": archive.as_posix(),
            "train_loss_last": history.history["loss"][-1],
            "train_accuracy_last": history.history["accuracy"][-1],
            "preprocessor_mean": self.preprocessor_["scaler"].mean_.tolist(),
            "preprocessor_scale": self.preprocessor_["scaler"].scale_.tolist()}
        if monitor is not None:
            proba = self.predict_proba(val_X)
            pred = proba.argmax(axis=1)
            record.update(metric_row(val_y, pred, proba))
            record["val_loss_last"] = history.history["val_loss"][-1]
            record["val_accuracy_last"] = history.history["val_accuracy"][-1]
            OOF_PREDICTIONS.append(pd.DataFrame({"config": cfg, "fold": fold,
                "row_index": val_X.index.to_numpy(), "truth": np.asarray(val_y), "prediction": pred,
                **{f"p{c}": proba[:, c] for c in CLASSES}}))
        FIT_RECORDS.append(record)
        dump_json(ARTIFACTS / "progress.json", {"completed_cv_fits": sum(r["fold"] > 0 for r in FIT_RECORDS),
            "last_run": f"{cfg}/{run}", "last_accuracy": record.get("accuracy"),
            "total_training_seconds": sum(r["training_seconds"] for r in FIT_RECORDS)})
        print(f"{cfg}/{run}: {self.epochs} epochs, {seconds:.1f}s" +
              (f", validation accuracy={record['accuracy']:.4f}" if monitor else ""))
        gc.collect()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        values = self.preprocessor_.transform(X).astype("float32")
        return self.model_(tf.convert_to_tensor(values), training=False).numpy().astype("float64")

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

architecture_table = []
for name, spec in ARCHITECTURES.items():
    preview = build_network(name, 0.001, RANDOM_STATE)
    architecture_table.append({"architecture": name, "units": str(spec["units"]),
        "BatchNorm / Dropout(0.3)": spec["regularized"], "parameters": preview.count_params()})
display(pd.DataFrame(architecture_table).set_index("architecture"))
display(pd.DataFrame(list(ParameterGrid({"architecture": list(ARCHITECTURES),
                                        "learning_rate": LEARNING_RATES}))))
ANALYSIS["network"] = (
    "Полная сетка содержит 3 архитектуры × 3 learning rate = 9 конфигураций; "
    "5 фолдов дают 45 отдельных CV-обучений. Shallow: 32 ReLU; deep: 128→64→32 ReLU; "
    "regularized добавляет BatchNormalization и Dropout(0.3) к той же глубокой сети. "
    "Выход всех сетей — 4 softmax-вероятности, loss — sparse categorical crossentropy. "
    "Скорости Adam: 0.0001, 0.001 и 0.01. Обёртка совместима с clone/GridSearchCV "
    "и обучает предобработку внутри fit на сырых данных своего фолда. "
    "Валидационные данные передаются только для мониторинга. Отсутствуют callback-и, "
    "изменяющие обучение по validation; сравниваются финальные веса после 60 эпох.")
''')

step("search", r'''
def choose_candidate(results):
    # Primary objective: mean accuracy. Exact ties: lower loss, then lower accuracy std,
    # then stable candidate order. Runtime is reported but never breaks random ties.
    return min(range(len(results["params"])), key=lambda i: (
        -results["mean_test_accuracy"][i], -results["mean_test_neg_log_loss"][i],
        results["std_test_accuracy"][i], i))

grid = {"architecture": list(ARCHITECTURES), "learning_rate": LEARNING_RATES}
search = GridSearchCV(TensorFlowClassifier(), grid, cv=CV_SPLITS,
    scoring={"accuracy": "accuracy", "neg_log_loss": "neg_log_loss"}, refit=choose_candidate,
    n_jobs=1, return_train_score=True, error_score="raise")
search_start = time.perf_counter()
search.fit(X_train, y_train)
search_seconds = time.perf_counter()-search_start
assert len(search.cv_results_["params"]) == 9
assert sum(r["fold"] > 0 for r in FIT_RECORDS) == 45
assert len(FIT_RECORDS) == 46
cv_rows = []
for i, params in enumerate(search.cv_results_["params"]):
    cfg = config_id(**params)
    runs = [r for r in FIT_RECORDS if r["config"] == cfg and r["fold"] > 0]
    cv_rows.append({"config": cfg, **params,
        "mean_accuracy": search.cv_results_["mean_test_accuracy"][i],
        "std_accuracy": search.cv_results_["std_test_accuracy"][i],
        "mean_loss": -search.cv_results_["mean_test_neg_log_loss"][i],
        "std_loss": search.cv_results_["std_test_neg_log_loss"][i],
        "mean_train_accuracy": search.cv_results_["mean_train_accuracy"][i],
        "mean_train_loss": -search.cv_results_["mean_train_neg_log_loss"][i],
        "mean_training_seconds": np.mean([r["training_seconds"] for r in runs]),
        "mean_fit_seconds": search.cv_results_["mean_fit_time"][i],
        "selected": i == search.best_index_})
cv_table = pd.DataFrame(cv_rows).sort_values(
    ["mean_accuracy", "mean_loss", "std_accuracy"], ascending=[False, True, True]).reset_index(drop=True)
best_config = config_id(**search.best_params_)
display(cv_table.round(5))
print("Выбрано:", search.best_params_)
fold_metrics = pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v, list)}
    for r in FIT_RECORDS if r["fold"] > 0])
cv_table.to_csv(ARTIFACTS / "cv_results.csv", index=False)
fold_metrics.to_csv(ARTIFACTS / "fold_metrics.csv", index=False)
pd.concat(OOF_PREDICTIONS, ignore_index=True).to_csv(ARTIFACTS / "oof_predictions.csv", index=False)
dump_json(ARTIFACTS / "fit_audit.json", FIT_RECORDS)
winner = cv_table.loc[cv_table.selected].iloc[0]
ANALYSIS["search"] = (
    f"Завершены 45 CV-обучений и один refit; общее время поиска {search_seconds:.1f} с. "
    f"Выбрана {best_config}: mean accuracy={winner.mean_accuracy:.4f}, "
    f"std={winner.std_accuracy:.4f}, mean loss={winner.mean_loss:.4f}; "
    f"среднее время обучения на фолд {winner.mean_training_seconds:.2f} с. "
    "Первичный критерий — максимальная средняя accuracy. При точном равенстве "
    "используются меньший средний loss, затем меньший разброс accuracy; "
    "время приведено отдельно и не влияет на выбор. Стандартное отклонение считается "
    "с ddof=0, как в GridSearchCV; это описание разброса пяти фолдов, не доверительный интервал. "
    "При refit с callable поле best_score_ не создаётся, поэтому итоговые оценки "
    "берутся из cv_results_ по best_index_.")
''')

md("## 5. Мониторинг и сравнение поведения конфигураций")
step("heatmaps", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, value, title, fmt, cmap in [
    (axes[0], "mean_accuracy", "CV accuracy (mean)", ".4f", "YlGn"),
    (axes[1], "mean_loss", "CV loss (mean)", ".3f", "YlOrRd"),
    (axes[2], "std_accuracy", "Разброс CV accuracy", ".4f", "Blues")]:
    table = cv_table.pivot(index="architecture", columns="learning_rate", values=value).reindex(ARCHITECTURES)
    sns.heatmap(table, annot=True, fmt=fmt, cmap=cmap, ax=ax)
    ax.set(title=title, xlabel="Learning rate", ylabel="Архитектура")
show("cv_heatmaps")
lr_means = cv_table.groupby("learning_rate").mean_accuracy.mean()
arch_means = cv_table.groupby("architecture").mean_accuracy.mean()
lr_effect = float(lr_means.max()-lr_means.min())
arch_effect = float(arch_means.max()-arch_means.min())
stronger = "learning rate" if lr_effect > arch_effect else "архитектура"
ANALYSIS["heatmaps"] = (
    "Средняя accuracy по трём архитектурам при каждом lr: "
    + "; ".join(f"{lr:g} → {v:.4f}" for lr,v in lr_means.items()) + ". "
    "Средняя accuracy по трём lr для каждой архитектуры: "
    + "; ".join(f"{a} → {v:.4f}" for a,v in arch_means.items()) + ". "
    f"Размах маргинальных средних: lr={100*lr_effect:.2f} п.п., "
    f"архитектура={100*arch_effect:.2f} п.п.; больший размах даёт {stronger}. "
    "Это описательная чувствительность в заданной сетке и при бюджете 60 эпох, "
    "а не причинная оценка или универсальный рейтинг важности параметров. "
    "Взаимодействия архитектуры и шага видны по отдельным ячейкам; "
    "BatchNormalization и Dropout здесь исследуются как единая архитектурная конфигурация.")
''')

step("curves", r'''
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for r, architecture in enumerate(ARCHITECTURES):
    for c, lr in enumerate(LEARNING_RATES):
        cfg = config_id(architecture, lr)
        h = [HISTORIES[f"{cfg}/fold_{fold}"] for fold in range(1, 6)]
        ax = axes[r, c]
        epochs = np.arange(1, EPOCHS+1)
        for metric, label, style in [("loss", "train", "--"), ("val_loss", "validation", "-")]:
            values = np.asarray([run[metric] for run in h])
            mean, std = values.mean(axis=0), values.std(axis=0)
            line, = ax.plot(epochs, mean, style, label=label)
            ax.fill_between(epochs, np.maximum(mean-std, 0), mean+std, alpha=0.15, color=line.get_color())
        ax.set(title=f"{architecture}, lr={lr:g}", xlabel="Эпоха", ylabel="Cross-entropy")
        ax.legend(fontsize=8)
show("all_loss_curves")

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
curve_diagnostics = []
for r, architecture in enumerate(ARCHITECTURES):
    for c, lr in enumerate(LEARNING_RATES):
        cfg = config_id(architecture, lr)
        h = [HISTORIES[f"{cfg}/fold_{fold}"] for fold in range(1, 6)]
        ax = axes[r, c]
        for metric, label, style in [("accuracy", "train", "--"), ("val_accuracy", "validation", "-")]:
            values = np.asarray([run[metric] for run in h])
            mean, std = values.mean(axis=0), values.std(axis=0)
            line, = ax.plot(np.arange(1, EPOCHS+1), mean, style, label=label)
            ax.fill_between(np.arange(1, EPOCHS+1), np.clip(mean-std,0,1),
                            np.clip(mean+std,0,1), alpha=0.15, color=line.get_color())
        ax.set(title=f"{architecture}, lr={lr:g}", xlabel="Эпоха", ylabel="Accuracy", ylim=(0,1.03))
        ax.legend(fontsize=8)
        losses = np.asarray([run["val_loss"] for run in h]).mean(axis=0)
        train_losses = np.asarray([run["loss"] for run in h]).mean(axis=0)
        curve_diagnostics.append({"config": cfg, "min_mean_val_loss_epoch": int(losses.argmin())+1,
            "min_mean_val_loss": float(losses.min()), "last_mean_val_loss": float(losses[-1]),
            "last_mean_train_loss": float(train_losses[-1]),
            "late_train_loss_change": float(train_losses[-5:].mean()-train_losses[-10:-5].mean()),
            "late_val_loss_change": float(losses[-5:].mean()-losses[-10:-5].mean())})
show("all_accuracy_curves")
diagnostics = pd.DataFrame(curve_diagnostics).set_index("config")
display(diagnostics.round(5))
diagnostics.to_csv(ARTIFACTS / "curve_diagnostics.csv")
deep_high = diagnostics.loc["deep_lr_0.01"]
selected_curve = diagnostics.loc[best_config]
ANALYSIS["curves"] = (
    "Каждая панель показывает среднее и ±1 std пяти фолдов; каждый из 45 запусков "
    "также доступен отдельно в TensorBoard. Для deep, lr=0.01 минимум среднего "
    f"validation loss={deep_high.min_mean_val_loss:.4f} на эпохе "
    f"{int(deep_high.min_mean_val_loss_epoch)}, последнее значение={deep_high.last_mean_val_loss:.4f}, "
    f"train loss={deep_high.last_mean_train_loss:.4f}. "
    f"Для выбранной конфигурации минимум среднего val_loss на эпохе "
    f"{int(selected_curve.min_mean_val_loss_epoch)}, конечный train/validation loss="
    f"{selected_curve.last_mean_train_loss:.4f}/{selected_curve.last_mean_val_loss:.4f}. "
    "Минимумы приведены только как диагностика: веса этих эпох не выбирались. "
    "Рост validation loss при продолжающемся снижении train loss указывает на "
    "переобучение; одновременное высокое значение обеих ошибок и продолжающееся "
    "снижение совместимы с недообучением. У Dropout/BatchNorm train и validation "
    "рассчитываются в разных режимах, поэтому отрицательный разрыв возможен.")
''')

step("stability", r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = cv_table.config.tolist()
sns.boxplot(data=fold_metrics, x="config", y="accuracy", order=order, color="lightblue", ax=axes[0])
sns.stripplot(data=fold_metrics, x="config", y="accuracy", order=order, color="black", size=4, ax=axes[0])
axes[0].set(title="Accuracy: распределение пяти фолдов", xlabel="Конфигурация", ylabel="Accuracy")
cv_table.set_index("config").mean_training_seconds.plot.bar(ax=axes[1], color="steelblue")
axes[1].set(title="Среднее время обучения одного фолда", xlabel="Конфигурация", ylabel="Секунды")
for ax in axes:
    ax.tick_params(axis="x", rotation=55)
show("stability_and_time")
best_folds = fold_metrics[fold_metrics.config == best_config]
most_stable = cv_table.loc[cv_table.std_accuracy.idxmin()]
fastest = cv_table.loc[cv_table.mean_training_seconds.idxmin()]
ANALYSIS["stability"] = (
    f"Выбранная {best_config}: accuracy по фолдам "
    f"{best_folds.sort_values('fold').accuracy.round(4).tolist()}, "
    f"диапазон {best_folds.accuracy.min():.4f}–{best_folds.accuracy.max():.4f}. "
    f"Минимальный std accuracy у {most_stable.config}: {most_stable.std_accuracy:.4f}; "
    f"наименьшее среднее время у {fastest.config}: {fastest.mean_training_seconds:.2f} с. "
    "Малый разброс при низкой средней accuracy может означать устойчивое недообучение. "
    "Выбор следует первичному критерию средней accuracy, а loss, std и время объясняют "
    "компромиссы между качеством и затратами. Замеры включают TensorBoard histogram_freq=1 "
    "и относятся к этой машине; они не являются универсальным бенчмарком архитектур.")
''')

step("tensorboard", r'''
original_cwd = Path.cwd()
try:
    # ASCII working directory avoids TensorFlow's Unicode Windows path issue.
    os.chdir(TB_WORKDIR)
    get_ipython().run_line_magic("load_ext", "tensorboard")  # %load_ext tensorboard
    get_ipython().run_line_magic("tensorboard", f"--logdir logs/additional/{RUN_ID} --port 6007")
except Exception as exc:
    print("Встраивание TensorBoard недоступно:", type(exc).__name__)
    print("Используйте: python view_tensorboard.py --additional")
finally:
    os.chdir(original_cwd)
print("Снимок логов:", (ARTIFACTS / "tensorboard").as_posix())
ANALYSIS["tensorboard"] = (
    "Сохранены 45 пар train/validation-журналов для CV и один train-журнал refit. "
    "TensorBoard пишет epoch_accuracy, epoch_loss и гистограммы весов на каждой эпохе. "
    "Run names содержат архитектуру, learning rate и номер фолда; refit выделен отдельно. "
    "Вызваны %load_ext tensorboard и %tensorboard с портом 6007. "
    "Все события скопированы в additional/artifacts/tensorboard, поэтому доступны "
    "после клонирования ветки. При кириллице в пути Windows запись и встраивание "
    "используют временный ASCII-каталог. Статические сетки loss и accuracy "
    "сохранены в самом ноутбуке и PNG независимо от доступности виджета.")
''')

md("## 6. Финальная оценка и сравнение с baseline")
step("final", r'''
final_model = search.best_estimator_
final_proba = final_model.predict_proba(X_test)
final_pred = final_model.predict(X_test)
final_metrics = metric_row(y_test, final_pred, final_proba)
final_table = pd.DataFrame([baseline_metrics, final_metrics], index=["LogisticRegression", best_config])
final_table["training_seconds"] = [baseline_seconds, final_model.training_seconds_]
display(final_table.round(5))
final_table.to_csv(ARTIFACTS / "final_metrics.csv", index_label="model")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
final_table[["accuracy", "f1_macro", "roc_auc_ovr_macro"]].plot.bar(ax=axes[0], rot=15)
axes[0].set(title="Baseline и лучшая MLP", ylabel="Метрика", ylim=(0,1.05))
axes[0].legend(fontsize=7)
ConfusionMatrixDisplay(confusion_matrix(y_test, final_pred, labels=CLASSES),
    display_labels=CLASSES).plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Матрица ошибок MLP")
y_test_bin = label_binarize(y_test, classes=CLASSES)
for c in CLASSES:
    fpr, tpr, _ = roc_curve(y_test_bin[:, c], final_proba[:, c])
    axes[2].plot(fpr, tpr, label=f"Класс {c}: {auc(fpr,tpr):.3f}")
axes[2].plot([0,1],[0,1],"k:")
axes[2].set(title="MLP: One-vs-Rest ROC", xlabel="FPR", ylabel="TPR")
axes[2].legend(fontsize=8)
show("final_comparison")
final_model.model_.save(ARTIFACTS / "best_model.keras")
joblib.dump({"preprocessor": final_model.preprocessor_, "features": FEATURES,
             "classes": CLASSES}, ARTIFACTS / "neural_preprocessing.joblib")
pd.DataFrame({"row_index": X_test.index, "truth": y_test.to_numpy(),
    "baseline_prediction": baseline_pred, "neural_prediction": final_pred,
    **{f"baseline_p{c}": baseline_proba[:, c] for c in CLASSES},
    **{f"neural_p{c}": final_proba[:, c] for c in CLASSES}}).to_csv(
        ARTIFACTS / "test_predictions.csv", index=False)
delta = final_metrics["accuracy"]-baseline_metrics["accuracy"]
ANALYSIS["final"] = (
    f"GridSearchCV обучил выбранную {best_config} с нуля на всех 1600 train-объектах "
    f"в течение {EPOCHS} эпох. Baseline Accuracy={baseline_metrics['accuracy']:.4f}, "
    f"MLP Accuracy={final_metrics['accuracy']:.4f}; изменение {100*delta:+.2f} п.п. "
    f"({100*delta/baseline_metrics['accuracy']:+.2f}% относительно baseline). "
    f"Loss: {baseline_metrics['loss']:.4f} → {final_metrics['loss']:.4f}; "
    f"F1-macro MLP={final_metrics['f1_macro']:.4f}, ROC-AUC={final_metrics['roc_auc_ovr_macro']:.4f}. "
    f"Ошибок MLP: {int((final_pred != y_test).sum())}/400; "
    f"refit занял {final_model.training_seconds_:.2f} с. "
    "Baseline и победитель используют один и тот же тест. Отрицательная дельта "
    "сохраняется как результат: настройка MLP не обязана превосходить сильную линейную модель.")
''')

step("conclusion", r'''
reference_config = "shallow_lr_0.001"
reference_row = cv_table.set_index("config").loc[reference_config]
cv_delta = float(winner.mean_accuracy-reference_row.mean_accuracy)
summary = pd.DataFrame({"Значение": [9, 45, EPOCHS, winner.mean_accuracy,
    winner.std_accuracy, baseline_metrics["accuracy"], final_metrics["accuracy"], 100*delta]},
    index=["Конфигураций", "CV-обучений", "Эпох в каждом запуске", "Лучшая CV accuracy",
           "Std CV accuracy", "Baseline test accuracy", "MLP test accuracy", "Изменение, п.п."])
display(summary)
winning_gap = float(winner.mean_train_accuracy-winner.mean_accuracy)
ANALYSIS["conclusion"] = (
    f"Точка отсчёта внутри нейросетевой сетки {reference_config}: "
    f"CV Accuracy={reference_row.mean_accuracy:.4f}; выбранная {best_config}: "
    f"{winner.mean_accuracy:.4f}; изменение {100*cv_delta:+.2f} п.п. "
    f"На тесте изменение относительно LogisticRegression равно {100*delta:+.2f} п.п. "
    "Эти сравнения отвечают разным вопросам: первое описывает настройку MLP на общих "
    "фолдах, второе — конкуренцию финальной сети с линейным baseline. "
    f"Для победителя средние inference train/validation accuracy="
    f"{winner.mean_train_accuracy:.4f}/{winner.mean_accuracy:.4f}, разрыв "
    f"{100*winning_gap:.2f} п.п. Описательная чувствительность к lr: "
    f"{100*lr_effect:.2f} п.п., к архитектуре: {100*arch_effect:.2f} п.п. "
    "Высокий train-score при росте val_loss указывает на переобучение; медленная "
    "сходимость при малом lr зависит от фиксированного бюджета 60 эпох. "
    "Средние CV-оценки победителя могут быть оптимистичны вследствие выбора среди "
    "9 кандидатов. Фолды пересекаются обучающими частями, их std не является "
    "доверительным интервалом. Единственный тест из 400 объектов и его использование "
    "в основной лабораторной ограничивают обобщение выводов; одна ошибка меняет "
    "Accuracy на 0.25 п.п. Для независимой последующей оценки нужны новые данные "
    "или заранее спланированная повторная вложенная CV.")
report = {"versions": versions, "random_state": RANDOM_STATE, "epochs": EPOCHS,
    "batch_size": BATCH_SIZE, "dataset_sha256": dataset_sha256, "n_configs": 9,
    "n_cv_fits": 45, "n_total_neural_fits": 46, "best_config": best_config,
    "best_params": search.best_params_, "best_cv_accuracy": float(winner.mean_accuracy),
    "best_cv_std_accuracy": float(winner.std_accuracy), "best_cv_loss": float(winner.mean_loss),
    "baseline_test": baseline_metrics, "neural_test": final_metrics,
    "test_accuracy_delta_pp": 100*delta, "cv_accuracy_delta_vs_shallow_pp": 100*cv_delta,
    "lr_marginal_range_pp": 100*lr_effect, "architecture_marginal_range_pp": 100*arch_effect,
    "baseline_training_seconds": baseline_seconds, "refit_training_seconds": final_model.training_seconds_,
    "total_search_seconds": search_seconds, "selection": "mean_accuracy desc, mean_loss asc, std_accuracy asc, stable order",
    "fold_seeds": [RANDOM_STATE+i for i in range(1,6)], "refit_seed": RANDOM_STATE,
    "reused_lab2_holdout": True}
dump_json(ARTIFACTS / "experiment_report.json", report)
dump_json(ARTIFACTS / "analyses.json", ANALYSIS)
assert len(HISTORIES) == 46 and all(len(h["loss"]) == EPOCHS for h in HISTORIES.values())
display({"application/vnd.lab2.additional-analysis+json": ANALYSIS}, raw=True)
''')

md(r"""
### Воспроизведение и источники

- [Исходный набор](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).
- [GridSearchCV: многокритериальное оценивание и callable refit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [Контракт sklearn-совместимого estimator](https://scikit-learn.org/stable/developers/develop.html).
- [TensorBoard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard).

Установка зависимостей: `python -m pip install -r requirements-lock.txt`.
Полный расчёт: `python generate_additional_notebook.py`.
Проверка сохранённых результатов: `python validate_additional_notebook.py`.
Мониторинг: `python view_tensorboard.py --additional` (порт 6007).
Артефакты дополнительного задания находятся в `additional/artifacts/`.
Результаты основной лабораторной хранятся отдельно в прежних файлах.
""")


def enrich_interpretations(analyses, report):
    """Add explicit diagnoses from measured histories, without choosing new weights."""
    import pandas as pd
    result = dict(analyses)
    diagnostics = []
    for architecture in ["shallow", "deep", "regularized"]:
        for rate in [0.0001, 0.001, 0.01]:
            cfg = f"{architecture}_lr_{rate:g}"
            histories = [pd.read_csv(
                ROOT / "additional/artifacts" / f"history_{cfg}_fold_{fold}.csv") for fold in range(1,6)]
            mean = sum(histories[1:], histories[0].copy()) / 5
            best_epoch = int(mean.val_loss.idxmin())
            overfit = (mean.val_loss.iloc[-1] > mean.val_loss.iloc[best_epoch]+0.02
                       and mean.loss.iloc[-1] < mean.loss.iloc[best_epoch])
            underfit = (mean.accuracy.iloc[-1] < 0.85 and mean.val_accuracy.iloc[-1] < 0.85
                        and mean.val_loss.iloc[-5:].mean() < mean.val_loss.iloc[-10:-5].mean()-0.01)
            if overfit:
                diagnostics.append(f"У {cfg} есть признаки переобучения: после минимума среднего "
                    f"val_loss на эпохе {best_epoch+1} он вырос с {mean.val_loss.iloc[best_epoch]:.4f} "
                    f"до {mean.val_loss.iloc[-1]:.4f}, тогда как train loss снизился с "
                    f"{mean.loss.iloc[best_epoch]:.4f} до {mean.loss.iloc[-1]:.4f}.")
            if underfit:
                diagnostics.append(f"У {cfg} есть признаки недообучения при заданном бюджете: "
                    f"финальные train/validation accuracy={mean.accuracy.iloc[-1]:.4f}/"
                    f"{mean.val_accuracy.iloc[-1]:.4f}, а средний val_loss последних пяти эпох "
                    f"ещё снизился относительно предыдущих пяти на "
                    f"{mean.val_loss.iloc[-10:-5].mean()-mean.val_loss.iloc[-5:].mean():.4f}.")
    diagnosis = "\n\n".join(diagnostics) if diagnostics else (
        "По объявленным диагностическим порогам устойчивые признаки переобучения или "
        "недообучения не выявлены; это не доказывает их отсутствие.")
    diagnosis += ("\n\nДиагностические пороги заданы явно: рост val_loss более 0.02 при снижении "
        "train loss после его validation-минимума; для недообучения — обе accuracy ниже 0.85 "
        "и снижение среднего val_loss ещё более 0.01 за последние два окна по пять эпох. "
        "Эти признаки служат интерпретации кривых и не меняют выбор конфигурации или эпох.")
    result["curves"] = result["curves"].split("\n\n**Диагностика по измеренным историям.**")[0]
    result["curves"] += "\n\n**Диагностика по измеренным историям.**\n\n"+diagnosis
    effect = "повысила" if report["test_accuracy_delta_pp"] > 0 else "не повысила"
    result["conclusion"] = result["conclusion"].split("\n\n**Фактический итог сравнения.**")[0]
    result["conclusion"] += (f"\n\n**Фактический итог сравнения.** Настроенная MLP {effect} тестовую "
        f"accuracy относительно фиксированной LogisticRegression: изменение "
        f"{report['test_accuracy_delta_pp']:+.2f} п.п. Эффект настройки внутри нейросетевой сетки "
        f"относительно shallow/lr=0.001 составляет {report['cv_accuracy_delta_vs_shallow_pp']:+.2f} "
        "п.п. средней CV accuracy. Оба результата сохранены без предположения о преимуществе MLP.")
    return result


def write_readme(report):
    text = f"""# Дополнительное задание: TensorFlow, 5-fold CV и Grid Search

Ветка `ДОП`, тот же Mobile Price Classification: 2000 смартфонов, 20 признаков, 4 класса.
Главный артефакт — [выполненный ноутбук](../Lab2_Additional_TensorFlow_CV.ipynb).

## Результат полного расчёта

| Показатель | Значение |
| :--- | ---: |
| Конфигураций: 3 архитектуры × 3 learning rate | 9 |
| Обучений на пяти фолдах | 45 |
| Эпох в каждом обучении, включая refit | {report['epochs']} |
| Победитель | `{report['best_config']}` |
| CV accuracy, mean ± std | {report['best_cv_accuracy']:.4f} ± {report['best_cv_std_accuracy']:.4f} |
| CV loss | {report['best_cv_loss']:.4f} |
| Baseline test accuracy (LogisticRegression) | {report['baseline_test']['accuracy']:.4f} |
| Итоговая MLP test accuracy | {report['neural_test']['accuracy']:.4f} |
| Изменение относительно baseline, п.п. | {report['test_accuracy_delta_pp']:+.2f} |

![CV: accuracy, loss и устойчивость](artifacts/cv_heatmaps.png)

## Протокол

- Внешнее стратифицированное разбиение 1600/400, seed 42.
- Пять общих стратифицированных фолдов 1280/320; скейлер и импутер обучаются внутри fit.
- Архитектуры: 32 ReLU; 128→64→32 ReLU; такая же глубокая сеть с BatchNorm и Dropout(0.3).
- Learning rate Adam: 0.0001, 0.001, 0.01; batch size 64.
- Ровно 60 эпох: проверочные кривые не управляют EarlyStopping или выбором весов.
- GridSearchCV максимизирует среднюю accuracy; точные равенства разрешаются по loss и std.
- После выбора выполнен один refit на всех 1600 train-объектах.
- Сохранены TensorBoard-логи всех 45 фолдов и refit, истории, OOF-вероятности,
  индексы разбиений, статистики предобработки, baseline и итоговая Keras-модель.

Это повторное учебное сравнение на том же тесте, который уже анализировался в основной
лабораторной. В текущем поиске тест не участвует в выборе параметров. Среднее качество
выбранного CV-кандидата подвержено смещению выбора, std по пяти зависимым фолдам
не является доверительным интервалом. Результат не гарантирует улучшения относительно
линейного baseline; отрицательные дельты сохранены.

## Запуск из корня репозитория

```powershell
git switch ДОП
.\\.venv\\Scripts\\python.exe generate_additional_notebook.py
.\\.venv\\Scripts\\python.exe validate_additional_notebook.py
.\\.venv\\Scripts\\python.exe view_tensorboard.py --additional
```

Зависимости общие с основной работой: `requirements-lock.txt`. На Linux/macOS
используйте `.venv/bin/python`. Для просмотра результатов повторное обучение не нужно.
После изменения кода или данных генератор обновляет измеренные Markdown-выводы.

## Файлы результатов

- [cv_results.csv](artifacts/cv_results.csv) — девять конфигураций и агрегаты CV.
- [fold_metrics.csv](artifacts/fold_metrics.csv) — метрики 45 проверочных фолдов.
- [oof_predictions.csv](artifacts/oof_predictions.csv) — прогнозы каждого объекта в CV.
- [split_manifest.json](artifacts/split_manifest.json), [fit_audit.json](artifacts/fit_audit.json)
  — проверка разбиений и статистик предобработки.
- [final_metrics.csv](artifacts/final_metrics.csv), [experiment_report.json](artifacts/experiment_report.json)
  — финальные оценки и параметры.
- `best_model.keras`, `neural_preprocessing.joblib`, `baseline_logistic.joblib` — модели.
- `history_*.csv`, `tensorboard/` — 46 историй и логов обучения.

![Финальная оценка](artifacts/final_comparison.png)
"""
    (ROOT / "additional" / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum seconds per cell")
    args = parser.parse_args()
    os.chdir(ROOT)
    # Preserve the previous event snapshot outside Git before a fresh execution.
    # Merging tfevents from two runs would duplicate epochs in TensorBoard and audits.
    snapshot = ROOT / "additional" / "artifacts" / "tensorboard"
    if snapshot.exists():
        preserved = ROOT / "logs" / "additional_snapshots" / str(time.time_ns())
        assert snapshot.resolve().is_relative_to(ROOT.resolve())
        assert preserved.resolve().is_relative_to(ROOT.resolve())
        preserved.parent.mkdir(parents=True, exist_ok=True)
        snapshot.rename(preserved)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    started = time.perf_counter()
    def on_cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"[{time.perf_counter()-started:7.1f}s] {cell.metadata.get('additional_step',cell_index)}", flush=True)
    client = NotebookClient(notebook, km=manager, timeout=args.timeout,
        resources={"metadata": {"path": str(ROOT)}}, on_cell_start=on_cell_start)
    try:
        client.execute(cleanup_kc=True)
    except BaseException:
        nbf.write(notebook, OUTPUT.with_suffix(".failed.ipynb"))
        raise
    analyses = None
    for cell in notebook.cells:
        if cell.cell_type == "code":
            clean = []
            for out in cell.outputs:
                if MIME in out.get("data", {}):
                    analyses = out.data[MIME]
                else:
                    clean.append(out)
            cell.outputs = clean
    if not analyses:
        raise RuntimeError("Measured analyses were not produced")
    report = json.loads((ROOT / "additional/artifacts/experiment_report.json").read_text(encoding="utf-8"))
    analyses = enrich_interpretations(analyses, report)
    for cell in notebook.cells:
        if key := cell.metadata.get("additional_analysis"):
            cell.source = "### Интерпретация результатов\n\n"+analyses[key]
    nbf.validate(notebook)
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
    nbf.write(notebook, OUTPUT)
    (ROOT / "additional/artifacts/analyses.json").write_text(
        json.dumps(analyses, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_readme(report)
    print(f"Saved {OUTPUT.name}: {len(notebook.cells)} cells, {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
