from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app import inference

client = TestClient(app)
DATA_DIR = Path(__file__).parent / "data"


def test_missing_columns_returns_400():
    csv_bytes = b"total_items\n1\n"
    response = client.post(
        "/predict",
        files={"file": ("invalid.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]


def test_predictions_add_column(monkeypatch):
    class DummyModel:
        def predict(self, data):
            return [999] * len(data)

    if hasattr(inference.load_model, "cache_clear"):
        inference.load_model.cache_clear()
    monkeypatch.setattr(inference, "load_model", lambda: DummyModel())

    sample_bytes = DATA_DIR.joinpath("sample.csv").read_bytes()
    response = client.post(
        "/predict",
        files={"file": ("sample.csv", sample_bytes, "text/csv")},
    )

    assert response.status_code == 200
    decoded = response.content.decode("utf-8-sig")
    header = decoded.splitlines()[0]
    assert "box_id_pred" in header
    assert "999" in decoded
