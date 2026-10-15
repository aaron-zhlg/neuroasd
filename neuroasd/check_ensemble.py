"""No-data checks for neuroasd.ensemble (synthetic FC).

    uv run python -m neuroasd.check_ensemble
"""

from __future__ import annotations

import numpy as np

from neuroasd import ensemble as en


class FakeSplit:
    def __init__(self, fcs, age, sex, groups, y=None):
        self._fcs = fcs
        self.age = age
        self.sex = sex
        self.groups = groups
        self.y = y

    def __len__(self):
        return len(self.groups)

    def fc(self, atlas: str) -> np.ndarray:
        return self._fcs[atlas]


def _synthetic(n_sites: int = 4, per_site: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_sites * per_site
    y = np.tile(np.arange(2), n // 2)
    groups = np.repeat([f"S{k}" for k in range(n_sites)], per_site)
    age = rng.uniform(7, 30, n)
    sex = rng.integers(1, 3, n)
    fcs = {}
    for atlas, p in (("gordon", 8), ("hcp", 9), ("power", 6), ("markov", 7)):
        mats = []
        for label in y:
            ts = rng.standard_normal((40, p))
            ts[:, 1] += 0.7 * label * ts[:, 0]
            mats.append(np.corrcoef(ts, rowvar=False))
        fcs[atlas] = np.stack(mats).astype(np.float32)
    return FakeSplit(fcs, age, sex, groups, y)


def _split_at(source: FakeSplit, hold: str):
    mask = source.groups == hold
    train = FakeSplit(
        {k: v[~mask] for k, v in source._fcs.items()},
        source.age[~mask],
        source.sex[~mask],
        source.groups[~mask],
        source.y[~mask],
    )
    test = FakeSplit(
        {k: v[mask] for k, v in source._fcs.items()},
        source.age[mask],
        source.sex[mask],
        source.groups[mask],
        None,
    )
    return train, test, source.y[mask]


def check_tangent_ignores_test_rows() -> None:
    source = _synthetic()
    train, test, _ = _split_at(source, "S0")
    base = en.feature_block(train, test, "gordon", "tangent")
    test._fcs = {
        k: np.broadcast_to(np.eye(v.shape[-1]), v.shape).copy().astype(np.float32)
        for k, v in test._fcs.items()
    }
    again = en.feature_block(train, test, "gordon", "tangent")
    n_train = len(train.y)
    assert np.allclose(base[:n_train], again[:n_train]), "test FC changed the tangent reference"


def check_concat_is_not_stacking() -> None:
    source = _synthetic()
    train, test, y_test = _split_at(source, "S0")
    n_train = len(train.y)
    gordon = en.standardize(en.feature_block(train, test, "gordon"), n_train)
    hcp = en.standardize(en.feature_block(train, test, "hcp"), n_train)
    concat = en.concat_blocks(gordon, hcp)
    clf = en.make_estimator("logreg").fit(concat[:n_train], train.y, train.groups)
    concat_scores = clf.decision_function(concat[n_train:])

    oof_g, te_g = en.member_scores(train, test, "gordon")
    oof_h, te_h = en.member_scores(train, test, "hcp")
    _, stacked = en.stack_scores(
        np.column_stack([oof_g, oof_h]), train.y, np.column_stack([te_g, te_h])
    )
    averaged = en.average_scores(np.column_stack([te_g, te_h]))
    assert concat_scores.shape == stacked.shape == averaged.shape == (len(y_test),)
    assert not np.allclose(concat_scores, stacked, atol=1e-5)
    assert not np.allclose(concat_scores, averaged, atol=1e-5)
    assert not np.allclose(stacked, averaged, atol=1e-5)


def check_stack_fit_ignores_test_members() -> None:
    """The meta-learner is fit on OOF only; test columns must not change train meta."""
    source = _synthetic()
    train, test, _ = _split_at(source, "S0")
    oof_g, te_g = en.member_scores(train, test, "gordon")
    oof_h, te_h = en.member_scores(train, test, "hcp")
    oof = np.column_stack([oof_g, oof_h])
    test_a = np.column_stack([te_g, te_h])
    train_a, out_a = en.stack_scores(oof, train.y, test_a)
    train_b, out_b = en.stack_scores(oof, train.y, test_a + 3.0)
    assert np.allclose(train_a, train_b, equal_nan=True)
    assert not np.allclose(out_a, out_b)


def check_average_is_label_free() -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((30, 3))
    assert np.allclose(en.average_scores(scores), scores.mean(axis=1))


def check_gnn_member_finite_and_deterministic() -> None:
    source = _synthetic(per_site=12)
    train, test, y_test = _split_at(source, "S0")
    a = en.GCNMember(epochs=2, hidden=8, batch=16, seed=0)
    b = en.GCNMember(epochs=2, hidden=8, batch=16, seed=0)
    a.fit(train.fc("power"), train.y, train.groups)
    b.fit(train.fc("power"), train.y, train.groups)
    sa = a.decision_function(test.fc("power"))
    sb = b.decision_function(test.fc("power"))
    assert sa.shape == (len(y_test),) and np.isfinite(sa).all()
    assert np.allclose(sa, sb, atol=1e-5)


def check_incumbent_fit_predict_still_imports() -> None:
    from autoresearch.trial import smoke_check

    metrics = smoke_check()
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def main() -> None:
    checks = [
        check_tangent_ignores_test_rows,
        check_concat_is_not_stacking,
        check_stack_fit_ignores_test_members,
        check_average_is_label_free,
        check_gnn_member_finite_and_deterministic,
        check_incumbent_fit_predict_still_imports,
    ]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")
    print(f"{len(checks)} passed")


if __name__ == "__main__":
    main()
