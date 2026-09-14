import pandas as pd
import pytest

from peyeutils.eyemovements import isi


def test_compute_isis_from_events_fills_gaps_between_events():
    ev = pd.DataFrame({
        "label": ["SACC", "SACC", "BLNK"],
        "stsec": [1.0, 3.0, 6.0],
        "ensec": [1.1, 3.2, 6.5],
        "eye": ["L", "L", "L"],
    })
    isis = isi.compute_ISIs_from_events(ev, zerotime=0.0)
    isis = isis.sort_values("stsec").reset_index(drop=True)

    assert (isis["label"] == "ISI").all()
    assert len(isis.index) == 3

    # first ISI runs from zerotime to the first event's start
    assert isis.loc[0, "stsec"] == pytest.approx(0.0)
    assert isis.loc[0, "ensec"] == pytest.approx(1.0)
    assert isis.loc[0, "dursec"] == pytest.approx(1.0)

    # subsequent ISIs run from the end of one event to the start of the next
    assert isis.loc[1, "stsec"] == pytest.approx(1.1)
    assert isis.loc[1, "ensec"] == pytest.approx(3.0)
    assert isis.loc[1, "dursec"] == pytest.approx(1.9)

    assert isis.loc[2, "stsec"] == pytest.approx(3.2)
    assert isis.loc[2, "ensec"] == pytest.approx(6.0)
    assert isis.loc[2, "dursec"] == pytest.approx(2.8)


def test_compute_isis_from_events_skips_eyes_with_too_few_events():
    ev = pd.DataFrame({
        "label": ["SACC"],
        "stsec": [1.0],
        "ensec": [1.1],
        "eye": ["L"],
    })
    isis = isi.compute_ISIs_from_events(ev, zerotime=0.0)
    assert isis.empty


def test_compute_isis_from_events_filters_by_eventstouse():
    ev = pd.DataFrame({
        "label": ["SACC", "FIXA", "SACC"],
        "stsec": [1.0, 2.0, 4.0],
        "ensec": [1.1, 3.5, 4.2],
        "eye": ["L", "L", "L"],
    })
    isis = isi.compute_ISIs_from_events(ev, zerotime=0.0, eventstouse=["SACC"])
    # only the two SACC rows count; the FIXA in between is ignored entirely
    assert len(isis.index) == 2
