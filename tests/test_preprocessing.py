"""Contratos que impedem vazamento e quebra na entrada de dados novos."""

import numpy as np
import pandas as pd
import pytest
from dataset import CATEGORICAL, ENGAGEMENT, FEATURES, RATES, profile_groups, split_data
from preprocessing import make_preprocessor
from sklearn.base import clone


@pytest.fixture
def training_frame():
    frame = pd.DataFrame({col: [0.0, 1.0, 4.0, np.nan] for col in ENGAGEMENT})
    for col in RATES:
        frame[col] = [0.0, 0.1, 0.2, np.nan]
    for col in CATEGORICAL:
        frame[col] = ["a", "b", "a", "b"]
    return frame


def test_statistics_are_learned_from_training_only_and_transform_is_read_only(training_frame):
    pipeline = make_preprocessor().fit(training_frame)
    fitted = pipeline.named_steps["columns"].named_transformers_
    np.testing.assert_allclose(fitted["engagement"].named_steps["impute"].statistics_, 1.0)
    expected_log_mean = np.log1p([0.0, 1.0, 4.0, 1.0]).mean()
    np.testing.assert_allclose(fitted["engagement"].named_steps["scale"].mean_, expected_log_mean)
    np.testing.assert_allclose(fitted["rates"].named_steps["impute"].statistics_, 0.1)
    np.testing.assert_allclose(fitted["rates"].named_steps["scale"].mean_, 0.1)
    state_before = {
        block: fitted[block].named_steps["scale"].mean_.copy() for block in ["engagement", "rates"]
    }
    categories_before = [a.copy() for a in fitted["category"].named_steps["onehot"].categories_]
    test = training_frame.copy()
    test[ENGAGEMENT] = 1e9
    test[RATES] = 0.95
    test[CATEGORICAL] = "inédito"
    pipeline.transform(test)
    for block, state in state_before.items():
        np.testing.assert_array_equal(state, fitted[block].named_steps["scale"].mean_)
    for before, after in zip(
        categories_before, fitted["category"].named_steps["onehot"].categories_, strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_missing_and_unseen_category_keep_finite_output_and_shape(training_frame):
    pipeline = make_preprocessor().fit(training_frame)
    baseline = pipeline.transform(training_frame)
    test = training_frame.iloc[:2].copy()
    test["Month"] = "Nunca_observado"
    test.loc[test.index[0], ENGAGEMENT + RATES] = np.nan
    test.loc[test.index[1], "VisitorType"] = np.nan
    output = pipeline.transform(test)
    names = pipeline.get_feature_names_out()
    month_mask = np.array([name.startswith("category__Month_") for name in names])
    assert output.shape == (2, baseline.shape[1])
    assert np.isfinite(output).all()
    assert not output[:, month_mask].any()
    assert np.count_nonzero(baseline[:, month_mask], axis=1).tolist() == [1, 1, 1, 1]


def test_target_and_pagevalues_do_not_affect_transformation(training_frame):
    frame = training_frame.assign(PageValues=[0, 0, 999999, 0], Revenue=[0, 0, 1, 0])
    with_extra = make_preprocessor().fit_transform(frame)
    without_extra = make_preprocessor().fit_transform(training_frame)
    np.testing.assert_allclose(with_extra, without_extra)
    pipeline = make_preprocessor().fit(frame)
    assert all(
        "Revenue" not in name and "PageValues" not in name
        for name in pipeline.get_feature_names_out()
    )
    assert clone(pipeline).get_params()["columns__remainder"] == "drop"


def test_entirely_missing_numeric_column_still_has_stable_output(training_frame):
    frame = training_frame.copy()
    frame["Administrative"] = np.nan
    pipeline = make_preprocessor().fit(frame)
    output = pipeline.transform(frame)
    assert "engagement__Administrative" in pipeline.get_feature_names_out()
    assert np.isfinite(output).all()


def test_conflicting_labels_and_different_discarded_values_stay_in_same_partition():
    frame = pd.DataFrame({col: np.arange(60, dtype=float) for col in ENGAGEMENT})
    for col in RATES:
        frame[col] = 0.1
    for col in CATEGORICAL:
        frame[col] = "categoria"
    frame["Revenue"] = np.arange(60) % 2
    frame["PageValues"] = np.arange(60, dtype=float)
    duplicate_profile = frame.iloc[[0]].assign(Revenue=1, PageValues=100000)
    frame = pd.concat([frame, frame.iloc[[1]], duplicate_profile], ignore_index=True)
    clean, train, test = split_data(frame)
    assert len(clean) == 61  # Só a cópia exata foi removida; conflito do alvo foi preservado.
    assert not set(profile_groups(train)) & set(profile_groups(test))
    assert len(train) + len(test) == len(clean)
    second_run = split_data(frame)
    assert train.index.tolist() == second_run[1].index.tolist()
    assert test.index.tolist() == second_run[2].index.tolist()
    assert set(FEATURES).isdisjoint({"Revenue", "PageValues"})
