import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_scoring_batch_endpoint(mocker):
    async_mock = mocker.AsyncMock()
    async_mock.return_value = {
        "company": "Test Company",
        "overview_comment": "This is a test."
    }
    mocker.patch("app.routers.scoring.process_single_file", async_mock)

    pdf_file = ("dummy.pdf", b"dummycontent", "application/pdf")
    response = client.post(
        "/scoring/batch",
        files={"files": pdf_file},
        data={"company_names": "Test Company", "website_urls": "http://example.com"},
    )

    assert response.status_code == 200
    response_json = response.json()
    assert isinstance(response_json, list)
    assert len(response_json) == 1
    assert response_json[0]["company"] == "Test Company"
    assert response_json[0]["overview_comment"] == "This is a test."
