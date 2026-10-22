import io
import json
from urllib.error import HTTPError

import pytest

from bnhmm.catalog import CatalogClient, CatalogError, metadata_from_visual_manifest


def test_visual_metadata_excludes_all_local_storage_locations():
    manifest = {
        "status": "complete", "errors": {}, "image_count": 3,
        "dataset_id": "test-v1", "source": "G:/private/market.duckdb",
        "config": {"start": "2026-09-01", "end": "2026-10-01", "output_root": "G:/private/images"},
        "tensor": {"width": 841, "height": 21}, "encoding": {"x": "log(price / previous_close)"},
        "protocol": "visual-v1", "symbol_count": 1, "generated_at": "2026-10-22T00:00:00Z",
    }
    payload = metadata_from_visual_manifest(manifest, category="test", name="test", description="test dataset")
    assert payload["sample_count"] == 3
    assert payload["feature_schema"]["end_date_exclusive"] is True
    assert "G:/private" not in json.dumps(payload)
    for changes in ({"status": "complete_with_errors"}, {"image_count": 0}, {"errors": {"000001": "missing"}}):
        with pytest.raises(ValueError):
            metadata_from_visual_manifest({**manifest, **changes}, category="x", name="x", description="x")


def test_catalog_uses_authenticated_metadata_api_without_token_in_url():
    client = CatalogClient("https://quant.example", "test-secret")
    captured = []

    def open_request(request, timeout):
        captured.append(request)
        return io.BytesIO(b'{"datasets": [], "total": 0}')

    client._opener.open = open_request
    assert client.list("visual training")["total"] == 0
    client.publish({"dataset_id": "test-v1", "name": "test"})
    assert captured[0].full_url.endswith("/catalog?category=visual+training")
    assert captured[1].get_method() == "PUT"
    assert captured[1].full_url.endswith("/admin/catalog/test-v1")
    assert all(request.get_header("Authorization") == "Bearer test-secret" for request in captured)
    assert all("test-secret" not in request.full_url for request in captured)


@pytest.mark.parametrize("server", ["http://remote.example", "https://user:secret@example.com", "https://example.com/path"])
def test_catalog_rejects_insecure_or_credential_containing_origins(server):
    with pytest.raises(ValueError):
        CatalogClient(server, "token")


@pytest.mark.parametrize("status", [401, 403])
def test_catalog_reports_login_and_permission_failures(status):
    client = CatalogClient("https://quant.example", "test-secret")

    def denied(request, timeout):
        raise HTTPError(request.full_url, status, "denied", {}, None)

    client._opener.open = denied
    with pytest.raises(CatalogError) as error:
        client.list()
    assert "test-secret" not in str(error.value)
