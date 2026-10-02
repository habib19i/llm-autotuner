"""My models library: interrupted downloads, cleanup, folder opening."""
import pytest
from fastapi.testclient import TestClient

from backend import downloader
from backend.app import app
from backend.utils import MODELS_DIR


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _pending(rel, files, model_id="someone/Thing-GGUF", quant="Q4_K_M", total=100):
    downloader._update_manifest(rel, {"model_id": model_id, "quant": quant, "format": "gguf",
                                      "pending": True, "started": "2026-10-03 10:00",
                                      "total_bytes": total, "files": files})


def test_interrupted_download_is_listed_with_progress(client):
    folder = MODELS_DIR / "Thing-GGUF"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Thing-Q4_K_M.gguf.tmp").write_bytes(b"x" * 40)
    _pending("Thing-GGUF/Thing-Q4_K_M.gguf", ["Thing-GGUF/Thing-Q4_K_M.gguf"])
    item = next(x for x in client.get("/api/installed").json() if x["filename"] == "Thing-GGUF/Thing-Q4_K_M.gguf")
    assert item["partial"] and not item["complete"]
    assert item["size_bytes"] == 40 and item["total_bytes"] == 100
    assert item["model_id"] == "someone/Thing-GGUF" and item["quant"] == "Q4_K_M"

    # Deleting it removes the partial file and the now-empty folder
    assert client.delete("/api/installed/Thing-GGUF/Thing-Q4_K_M.gguf").status_code == 200
    assert not folder.exists()
    assert not any(x["filename"].startswith("Thing-GGUF/") for x in client.get("/api/installed").json())


def test_partial_multi_shard_delete_keeps_other_models(client):
    folder = MODELS_DIR / "Big-GGUF"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Big-Q4_K_M-00001-of-00002.gguf").write_bytes(b"a" * 10)
    (folder / "Big-Q4_K_M-00002-of-00002.gguf.tmp").write_bytes(b"b" * 5)
    (folder / "Big-Q8_0.gguf").write_bytes(b"c")  # another, complete model in the same folder
    _pending("Big-GGUF/Big-Q4_K_M-00001-of-00002.gguf",
             ["Big-GGUF/Big-Q4_K_M-00001-of-00002.gguf", "Big-GGUF/Big-Q4_K_M-00002-of-00002.gguf"])
    items = {x["filename"]: x for x in client.get("/api/installed").json()}
    part = items["Big-GGUF/Big-Q4_K_M-00001-of-00002.gguf"]
    assert part["partial"] and not part["complete"]
    assert client.delete("/api/installed/Big-GGUF/Big-Q4_K_M-00001-of-00002.gguf").status_code == 200
    assert (folder / "Big-Q8_0.gguf").exists() and not (folder / "Big-Q4_K_M-00001-of-00002.gguf").exists()
    client.delete("/api/installed/Big-GGUF/Big-Q8_0.gguf")


def test_completed_download_is_not_partial(client):
    folder = MODELS_DIR / "Done-GGUF"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Done-Q4_K_M.gguf").write_bytes(b"x" * 10)
    downloader._update_manifest("Done-GGUF/Done-Q4_K_M.gguf", {"model_id": "a/Done-GGUF", "quant": "Q4_K_M",
                                                               "total_bytes": 10, "format": "gguf",
                                                               "downloaded": "2026-10-03 09:00"})
    item = next(x for x in client.get("/api/installed").json() if x["filename"] == "Done-GGUF/Done-Q4_K_M.gguf")
    assert item["complete"] and not item["partial"] and item["downloaded"] == "2026-10-03 09:00"
    client.delete("/api/installed/Done-GGUF/Done-Q4_K_M.gguf")


def test_open_folder_rejects_bad_paths(client):
    assert client.post("/api/open-folder", json={"filename": "../../etc"}).status_code == 400
    assert client.post("/api/open-folder", json={"filename": "Nope-GGUF/x.gguf"}).status_code == 404
