import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_annotation() -> dict:
    return json.loads((FIXTURES_DIR / "sample_funsd_annotation.json").read_text(encoding="utf-8"))
