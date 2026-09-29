import pandas as pd

from peyeutils.utils import clusterutils as cu


def test_unique_clusters_groups_rows_sharing_a_value():
    df = pd.DataFrame({"a": [1, 2, 1, 3], "b": ["x", "y", "z", "w"]})
    out = cu.unique_clusters(df)
    # rows 0 and 2 share a==1, so must end up in the same cluster
    assert out.loc[0, "cluster"] == out.loc[2, "cluster"]
    # rows 1 and 3 are singletons, distinct from each other and from 0/2
    assert out.loc[1, "cluster"] != out.loc[0, "cluster"]
    assert out.loc[3, "cluster"] != out.loc[0, "cluster"]
    assert out.loc[1, "cluster"] != out.loc[3, "cluster"]


def test_unique_clusters_groups_rows_sharing_any_column_value():
    # rows connected transitively: row0-row1 share 'a', row1-row2 share 'b'
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "y"]})
    out = cu.unique_clusters(df)
    assert out.loc[0, "cluster"] == out.loc[1, "cluster"] == out.loc[2, "cluster"]


def test_unique_clusters_works_with_non_contiguous_index():
    # Regression test: unique_clusters used to build a plain 0..N-1 list and
    # index into it using actual index LABELS (from df.iterrows()), which
    # raises IndexError (or silently misassigns) for any df whose index
    # isn't a default contiguous RangeIndex starting at 0.
    df = pd.DataFrame(
        {"a": [1, 2, 1, 3], "b": ["x", "y", "z", "w"]}, index=[10, 20, 30, 40]
    )
    out = cu.unique_clusters(df)
    assert list(out.index) == [10, 20, 30, 40]
    assert out.loc[10, "cluster"] == out.loc[30, "cluster"]
    assert out.loc[20, "cluster"] != out.loc[10, "cluster"]
