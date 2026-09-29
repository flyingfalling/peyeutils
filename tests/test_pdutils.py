import numpy as np
import pandas as pd

import peyeutils as pu


def test_safe_df_concat_basic_union_of_columns():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})
    out = pu.utils.safe_df_concat([df1, df2])
    assert len(out.index) == 4
    assert set(out.columns) == {"a", "b", "c"}
    # rows from df2 have no 'b' -> NaN
    assert out["b"].isna().sum() == 2
    assert out["c"].isna().sum() == 2


def test_safe_df_concat_skips_empty_frames():
    df1 = pd.DataFrame({"a": [1, 2]})
    out = pu.utils.safe_df_concat([df1, pd.DataFrame()])
    assert len(out.index) == 2


def test_safe_df_concat_all_empty_returns_empty_df():
    out = pu.utils.safe_df_concat([pd.DataFrame(), pd.DataFrame()])
    assert out.empty


def test_safe_df_concat_readds_column_that_was_all_nan_in_every_frame():
    df1 = pd.DataFrame({"a": [1, 2], "b": [np.nan, np.nan]})
    df2 = pd.DataFrame({"a": [3, 4]})
    out = pu.utils.safe_df_concat([df1, df2])
    assert "b" in out.columns
    assert out["b"].isna().all()
