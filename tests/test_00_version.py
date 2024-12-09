import pyic


def test_version() -> None:
    assert pyic.__version__ != "999"
