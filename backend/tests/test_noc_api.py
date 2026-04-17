"""
Backend API tests for NOC Zoom-Call Prediction System.

Tests cover:
- Service health and model status
- KPIs endpoint
- Model metadata endpoint
- Incidents CRUD with filters (severity, zoom, search, pagination)
- Prediction endpoints (single and batch)
- Seed and train endpoints
- OCI client health and pull (mock mode)
- Architecture docs endpoint
"""
import os
import pytest
import requests

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')


class TestServiceHealth:
    """Test service banner and health endpoints"""
    
    def test_root_returns_service_banner(self):
        """GET /api/ returns service banner with model_loaded=true"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "noc-zoom-predictor"
        assert data["version"] == "1.0.0"
        assert data["model_loaded"] is True
        assert "docs" in data
        print(f"SUCCESS: Root endpoint returns banner with model_loaded={data['model_loaded']}")
    
    def test_health_returns_ok_with_counts(self):
        """GET /api/health returns status ok with mongo counts and model loaded"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        
        # Status check
        assert data["status"] == "ok"
        
        # Mongo counts
        assert "mongo" in data
        assert data["mongo"]["incidents"] > 0, "Expected incidents > 0"
        assert data["mongo"]["predictions"] > 0, "Expected predictions > 0"
        
        # Model status
        assert "model" in data
        assert data["model"]["loaded"] is True
        
        print(f"SUCCESS: Health check - incidents={data['mongo']['incidents']}, predictions={data['mongo']['predictions']}, model_loaded={data['model']['loaded']}")


class TestKPIs:
    """Test KPI endpoint"""
    
    def test_kpis_returns_numeric_fields(self):
        """GET /api/kpis returns all required numeric KPI fields"""
        response = requests.get(f"{BASE_URL}/api/kpis")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "total_open_events", "total_open_nocs", "impacted_regions",
            "multi_region_nocs", "impacted_customers", "zoom_calls_predicted"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing KPI field: {field}"
            assert isinstance(data[field], (int, float)), f"KPI {field} should be numeric"
        
        print(f"SUCCESS: KPIs - open_events={data['total_open_events']}, zoom_predicted={data['zoom_calls_predicted']}")


class TestModelEndpoint:
    """Test model metadata endpoint"""
    
    def test_model_returns_loaded_with_metrics(self):
        """GET /api/model returns loaded=true with metrics"""
        response = requests.get(f"{BASE_URL}/api/model")
        assert response.status_code == 200
        data = response.json()
        
        assert data["loaded"] is True
        assert "metadata" in data
        
        meta = data["metadata"]
        assert "metrics" in meta
        metrics = meta["metrics"]
        
        # Check all metrics are floats
        for metric in ["pr_auc", "roc_auc", "f1", "brier"]:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], float), f"Metric {metric} should be float"
        
        # Check feature count
        assert meta["feature_count"] == 56, f"Expected 56 features, got {meta['feature_count']}"
        
        # Check threshold
        assert 0 <= meta["threshold"] <= 1, f"Threshold should be 0-1, got {meta['threshold']}"
        
        print(f"SUCCESS: Model - pr_auc={metrics['pr_auc']:.4f}, roc_auc={metrics['roc_auc']:.4f}, f1={metrics['f1']:.4f}, features={meta['feature_count']}")


class TestIncidentsEndpoint:
    """Test incidents listing with various filters"""
    
    def test_incidents_default_returns_3000_total(self):
        """GET /api/incidents with defaults returns total==3000"""
        response = requests.get(f"{BASE_URL}/api/incidents")
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 3000, f"Expected 3000 total incidents, got {data['total']}"
        assert "rows" in data
        assert len(data["rows"]) > 0
        
        # Check prediction object in rows
        row = data["rows"][0]
        assert "prediction" in row
        pred = row["prediction"]
        assert pred["decision"] in ["Yes", "No", "Review"]
        assert 0 <= pred["probability"] <= 1
        assert 0 <= pred["confidence"] <= 1
        
        print(f"SUCCESS: Incidents default - total={data['total']}, rows_returned={len(data['rows'])}")
    
    def test_incidents_severity_filter_sev1(self):
        """GET /api/incidents?severity=SEV1 returns only SEV1 rows with decision=Yes"""
        response = requests.get(f"{BASE_URL}/api/incidents?severity=SEV1")
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["rows"]) > 0, "Expected some SEV1 incidents"
        
        for row in data["rows"]:
            assert row["severity"] == "SEV1", f"Expected SEV1, got {row['severity']}"
            # SEV1 hard rule: decision must be Yes
            assert row["prediction"]["decision"] == "Yes", f"SEV1 should have decision=Yes, got {row['prediction']['decision']}"
        
        print(f"SUCCESS: SEV1 filter - {len(data['rows'])} rows, all have decision=Yes")
    
    def test_incidents_zoom_filter_yes(self):
        """GET /api/incidents?zoom=Yes returns only rows with decision==Yes"""
        response = requests.get(f"{BASE_URL}/api/incidents?zoom=Yes")
        assert response.status_code == 200
        data = response.json()
        
        for row in data["rows"]:
            assert row["prediction"]["decision"] == "Yes", f"Expected decision=Yes, got {row['prediction']['decision']}"
        
        print(f"SUCCESS: Zoom=Yes filter - {len(data['rows'])} rows, all have decision=Yes")
    
    def test_incidents_search_filter(self):
        """GET /api/incidents?search=NOC-500 returns matching rows"""
        response = requests.get(f"{BASE_URL}/api/incidents?search=NOC-500")
        assert response.status_code == 200
        data = response.json()
        
        # Should find rows containing NOC-500 in alias/jira_id/title
        for row in data["rows"]:
            match_found = (
                "NOC-500" in row.get("alias", "").upper() or
                "NOC-500" in row.get("jira_id", "").upper() or
                "NOC-500" in row.get("title", "").upper()
            )
            assert match_found, f"Row should contain NOC-500: {row['alias']}"
        
        print(f"SUCCESS: Search filter - {len(data['rows'])} rows matching NOC-500")
    
    def test_incidents_pagination(self):
        """GET /api/incidents?page=2&page_size=10 returns exactly 10 rows and page==2"""
        response = requests.get(f"{BASE_URL}/api/incidents?page=2&page_size=10")
        assert response.status_code == 200
        data = response.json()
        
        assert data["page"] == 2
        assert data["page_size"] == 10
        assert len(data["rows"]) == 10, f"Expected 10 rows, got {len(data['rows'])}"
        
        print(f"SUCCESS: Pagination - page={data['page']}, page_size={data['page_size']}, rows={len(data['rows'])}")
    
    def test_incident_by_alias_found(self):
        """GET /api/incidents/{alias} for known alias returns incident with prediction"""
        # First get a known alias
        list_response = requests.get(f"{BASE_URL}/api/incidents?page_size=1")
        alias = list_response.json()["rows"][0]["alias"]
        
        response = requests.get(f"{BASE_URL}/api/incidents/{alias}")
        assert response.status_code == 200
        data = response.json()
        
        assert data["alias"] == alias
        assert "prediction" in data
        
        print(f"SUCCESS: Get incident by alias - {alias}")
    
    def test_incident_by_alias_not_found(self):
        """GET /api/incidents/NOC-999999999 returns 404"""
        response = requests.get(f"{BASE_URL}/api/incidents/NOC-999999999")
        assert response.status_code == 404
        
        print("SUCCESS: Non-existent alias returns 404")


class TestPredictEndpoints:
    """Test prediction endpoints"""
    
    def test_predict_sev1_returns_yes(self):
        """POST /api/predict with SEV1 payload returns decision=Yes"""
        payload = {
            "alias": "TEST-SEV1-001",
            "severity": "SEV1",
            "status": "INVESTIGATING",
            "region": "us-ashburn-1",
            "title": "Critical outage in production",
            "multi_region": 1,
            "has_customer_impact": 1,
            "has_outage": 1
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["decision"] == "Yes", f"SEV1 should return Yes, got {data['decision']}"
        assert data["reason"] == "sev1_hard_rule"
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        
        print(f"SUCCESS: SEV1 prediction - decision={data['decision']}, reason={data['reason']}")
    
    def test_predict_sev4_low_features(self):
        """POST /api/predict with SEV4 low-feature payload returns valid schema"""
        payload = {
            "alias": "TEST-SEV4-001",
            "severity": "SEV4",
            "status": "RESOLVED",
            "region": "us-ashburn-1",
            "title": "Minor issue resolved",
            "multi_region": 0,
            "has_customer_impact": 0,
            "has_outage": 0,
            "workstream_count": 1,
            "attachment_count": 0
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Check schema
        assert "decision" in data
        assert data["decision"] in ["Yes", "No", "Review"]
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        
        print(f"SUCCESS: SEV4 prediction - decision={data['decision']}, probability={data['probability']:.4f}")
    
    def test_predict_batch(self):
        """POST /api/predict/batch with 3 incidents returns count==3"""
        payload = {
            "incidents": [
                {"alias": "BATCH-001", "severity": "SEV1", "region": "us-ashburn-1"},
                {"alias": "BATCH-002", "severity": "SEV2", "region": "eu-frankfurt-1"},
                {"alias": "BATCH-003", "severity": "SEV4", "region": "ap-tokyo-1"}
            ]
        }
        response = requests.post(f"{BASE_URL}/api/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 3
        assert len(data["predictions"]) == 3
        
        # First should be Yes (SEV1)
        assert data["predictions"][0]["decision"] == "Yes"
        
        print(f"SUCCESS: Batch prediction - count={data['count']}")


class TestSeedAndTrain:
    """Test seed and train endpoints"""
    
    def test_seed_with_retrain(self):
        """POST /api/seed {n:500, retrain:true} returns retrained=true with metrics"""
        payload = {"n": 500, "retrain": True, "seed": 123}
        response = requests.post(f"{BASE_URL}/api/seed", json=payload, timeout=120)
        assert response.status_code == 200
        data = response.json()
        
        assert data["seeded"] == 500
        assert data["retrained"] is True
        assert "metrics" in data
        assert isinstance(data["metrics"]["pr_auc"], float)
        
        print(f"SUCCESS: Seed with retrain - seeded={data['seeded']}, pr_auc={data['metrics']['pr_auc']:.4f}")
    
    def test_train_endpoint(self):
        """POST /api/train {epochs:10, n_synthetic:500} returns metrics"""
        payload = {"epochs": 10, "n_synthetic": 500, "seed": 456}
        response = requests.post(f"{BASE_URL}/api/train", json=payload, timeout=120)
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert isinstance(data["metrics"]["pr_auc"], float)
        assert data["predictions_written"] > 0
        
        print(f"SUCCESS: Train - pr_auc={data['metrics']['pr_auc']:.4f}, predictions_written={data['predictions_written']}")


class TestOCIClient:
    """Test OCI client endpoints (mock mode)"""
    
    def test_oci_health_mock_mode(self):
        """GET /api/oci/health returns mode='mock' and excel_fallback_exists=true"""
        response = requests.get(f"{BASE_URL}/api/oci/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "mock", f"Expected mock mode, got {data['mode']}"
        assert data["excel_fallback_exists"] is True
        
        print(f"SUCCESS: OCI health - mode={data['mode']}, excel_fallback={data['excel_fallback_exists']}")
    
    def test_oci_pull_mock_mode(self):
        """POST /api/oci/pull?limit=5 in mock mode returns count>=0 and mode='mock'"""
        response = requests.post(f"{BASE_URL}/api/oci/pull?limit=5")
        assert response.status_code == 200
        data = response.json()
        
        assert data["mode"] == "mock"
        assert data["count"] >= 0
        
        print(f"SUCCESS: OCI pull - mode={data['mode']}, count={data['count']}")


class TestDocsEndpoint:
    """Test architecture docs endpoint"""
    
    def test_docs_architecture_returns_markdown(self):
        """GET /api/docs/architecture returns non-empty text starting with '# NOC'"""
        response = requests.get(f"{BASE_URL}/api/docs/architecture")
        assert response.status_code == 200
        
        text = response.text
        assert len(text) > 100, "Architecture doc should be non-empty"
        assert text.startswith("# NOC Zoom-Call Prediction"), f"Doc should start with expected header, got: {text[:50]}"
        
        print(f"SUCCESS: Architecture doc - {len(text)} characters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
