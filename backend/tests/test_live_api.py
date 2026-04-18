"""
Test suite for Live 4-API Pull Loop endpoints.

Tests:
- GET /api/live/status - scheduler status
- POST /api/live/tick - manual tick trigger
- POST /api/live/start - start scheduler
- POST /api/live/stop - stop scheduler
- GET /api/live/ticks - tick history
- GET /api/live/folders/{alias} - per-NOC folder files
- GET /api/incidents/{alias} - live-pulled incident with zoom_link, prediction, explanation
- Regression tests for core endpoints
"""
import os
import pytest
import requests

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')


class TestLiveStatus:
    """Tests for GET /api/live/status endpoint"""
    
    def test_live_status_returns_200(self):
        """Status endpoint should return 200"""
        response = requests.get(f"{BASE_URL}/api/live/status")
        assert response.status_code == 200
        print(f"PASSED: /api/live/status returns 200")
    
    def test_live_status_has_required_fields(self):
        """Status should have running, current_var, interval_seconds, retrain_every_n_ticks"""
        response = requests.get(f"{BASE_URL}/api/live/status")
        data = response.json()
        
        assert "running" in data, "Missing 'running' field"
        assert isinstance(data["running"], bool), "running should be boolean"
        
        assert "current_var" in data, "Missing 'current_var' field"
        assert isinstance(data["current_var"], int), "current_var should be int"
        
        assert "interval_seconds" in data, "Missing 'interval_seconds' field"
        assert data["interval_seconds"] == 60, f"interval_seconds should be 60, got {data['interval_seconds']}"
        
        assert "retrain_every_n_ticks" in data, "Missing 'retrain_every_n_ticks' field"
        assert data["retrain_every_n_ticks"] == 10, f"retrain_every_n_ticks should be 10, got {data['retrain_every_n_ticks']}"
        
        assert "last_tick_ts" in data, "Missing 'last_tick_ts' field"
        assert "last_tick" in data, "Missing 'last_tick' field"
        
        print(f"PASSED: /api/live/status has all required fields")
        print(f"  running={data['running']}, current_var={data['current_var']}")


class TestLiveTick:
    """Tests for POST /api/live/tick endpoint"""
    
    def test_live_tick_returns_200(self):
        """Manual tick should return 200"""
        response = requests.post(f"{BASE_URL}/api/live/tick")
        assert response.status_code == 200
        print(f"PASSED: POST /api/live/tick returns 200")
    
    def test_live_tick_returns_expected_schema(self):
        """Tick response should have var, pulled, items"""
        response = requests.post(f"{BASE_URL}/api/live/tick")
        data = response.json()
        
        assert "var" in data, "Missing 'var' field"
        assert isinstance(data["var"], int), "var should be int"
        
        assert "pulled" in data, "Missing 'pulled' field"
        assert isinstance(data["pulled"], int), "pulled should be int"
        
        assert "items" in data, "Missing 'items' field"
        assert isinstance(data["items"], list), "items should be list"
        
        if data["items"]:
            item = data["items"][0]
            assert "alias" in item, "Item missing 'alias'"
            assert "decision" in item, "Item missing 'decision'"
            assert "probability" in item, "Item missing 'probability'"
            assert "zoom_link" in item, "Item missing 'zoom_link'"
            assert "source" in item, "Item missing 'source'"
            
            # Verify alias format
            assert item["alias"].startswith("NOC-160"), f"Alias should start with NOC-160, got {item['alias']}"
            
            # Verify zoom_link format
            assert item["zoom_link"].startswith("https://oracle.zoom.us/j/"), \
                f"zoom_link should start with https://oracle.zoom.us/j/, got {item['zoom_link']}"
        
        print(f"PASSED: POST /api/live/tick returns expected schema")
        print(f"  var={data['var']}, pulled={data['pulled']}, items={len(data['items'])}")


class TestLiveStartStop:
    """Tests for POST /api/live/start and POST /api/live/stop"""
    
    def test_live_stop_then_status_shows_not_running(self):
        """After stop, status should show running=false"""
        # Stop
        stop_resp = requests.post(f"{BASE_URL}/api/live/stop")
        assert stop_resp.status_code == 200
        
        # Check status
        status_resp = requests.get(f"{BASE_URL}/api/live/status")
        data = status_resp.json()
        assert data["running"] == False, f"Expected running=false after stop, got {data['running']}"
        
        print(f"PASSED: POST /api/live/stop sets running=false")
    
    def test_live_start_then_status_shows_running(self):
        """After start, status should show running=true"""
        # Start
        start_resp = requests.post(f"{BASE_URL}/api/live/start")
        assert start_resp.status_code == 200
        
        # Check status
        status_resp = requests.get(f"{BASE_URL}/api/live/status")
        data = status_resp.json()
        assert data["running"] == True, f"Expected running=true after start, got {data['running']}"
        
        print(f"PASSED: POST /api/live/start sets running=true")


class TestLiveTicks:
    """Tests for GET /api/live/ticks endpoint"""
    
    def test_live_ticks_returns_200(self):
        """Ticks history endpoint should return 200"""
        response = requests.get(f"{BASE_URL}/api/live/ticks?limit=5")
        assert response.status_code == 200
        print(f"PASSED: GET /api/live/ticks returns 200")
    
    def test_live_ticks_returns_expected_schema(self):
        """Ticks response should have count and ticks array"""
        response = requests.get(f"{BASE_URL}/api/live/ticks?limit=5")
        data = response.json()
        
        assert "count" in data, "Missing 'count' field"
        assert "ticks" in data, "Missing 'ticks' field"
        assert isinstance(data["ticks"], list), "ticks should be list"
        
        if data["ticks"]:
            tick = data["ticks"][0]
            assert "var" in tick, "Tick missing 'var'"
            assert "pulled" in tick, "Tick missing 'pulled'"
            assert "ts" in tick, "Tick missing 'ts'"
            assert "sample" in tick, "Tick missing 'sample'"
        
        print(f"PASSED: GET /api/live/ticks returns expected schema")
        print(f"  count={data['count']}, ticks returned={len(data['ticks'])}")


class TestLiveFolders:
    """Tests for GET /api/live/folders/{alias} endpoint"""
    
    def test_live_folders_returns_files_for_valid_alias(self):
        """Folder endpoint should return files for a live-pulled alias"""
        # First get a valid alias from ticks
        ticks_resp = requests.get(f"{BASE_URL}/api/live/ticks?limit=1")
        ticks_data = ticks_resp.json()
        
        if not ticks_data["ticks"]:
            pytest.skip("No ticks available to test folders")
        
        alias = ticks_data["ticks"][0]["sample"][0]["alias"]
        
        # Get folder
        response = requests.get(f"{BASE_URL}/api/live/folders/{alias}")
        assert response.status_code == 200, f"Expected 200 for {alias}, got {response.status_code}"
        
        data = response.json()
        assert "alias" in data, "Missing 'alias' field"
        assert "folder" in data, "Missing 'folder' field"
        assert "files" in data, "Missing 'files' field"
        
        # Verify expected files
        expected_files = [
            "api1_incident.json",
            "api2_attachments.json",
            "api3_content.csv",
            "api3_content.json",
            "api4_channels.json"
        ]
        for f in expected_files:
            assert f in data["files"], f"Missing expected file: {f}"
        
        print(f"PASSED: GET /api/live/folders/{alias} returns expected files")
        print(f"  files={data['files']}")
    
    def test_live_folders_returns_404_for_unknown_alias(self):
        """Folder endpoint should return 404 for unknown alias"""
        response = requests.get(f"{BASE_URL}/api/live/folders/NOC-UNKNOWN-ALIAS")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        
        print(f"PASSED: GET /api/live/folders/NOC-UNKNOWN-ALIAS returns 404")


class TestLiveIncidentDetails:
    """Tests for GET /api/incidents/{alias} for live-pulled incidents"""
    
    def test_live_incident_has_zoom_link(self):
        """Live-pulled incident should have zoom_link field"""
        # Get a live alias
        ticks_resp = requests.get(f"{BASE_URL}/api/live/ticks?limit=1")
        ticks_data = ticks_resp.json()
        
        if not ticks_data["ticks"]:
            pytest.skip("No ticks available")
        
        alias = ticks_data["ticks"][0]["sample"][0]["alias"]
        
        response = requests.get(f"{BASE_URL}/api/incidents/{alias}")
        assert response.status_code == 200
        
        data = response.json()
        assert "zoom_link" in data, "Missing 'zoom_link' field"
        assert isinstance(data["zoom_link"], str), "zoom_link should be string"
        assert data["zoom_link"].startswith("https://oracle.zoom.us/j/"), \
            f"zoom_link should start with https://oracle.zoom.us/j/, got {data['zoom_link']}"
        
        print(f"PASSED: GET /api/incidents/{alias} has valid zoom_link")
        print(f"  zoom_link={data['zoom_link'][:60]}...")
    
    def test_live_incident_has_prediction(self):
        """Live-pulled incident should have prediction with decision/probability/confidence"""
        ticks_resp = requests.get(f"{BASE_URL}/api/live/ticks?limit=1")
        ticks_data = ticks_resp.json()
        
        if not ticks_data["ticks"]:
            pytest.skip("No ticks available")
        
        alias = ticks_data["ticks"][0]["sample"][0]["alias"]
        
        response = requests.get(f"{BASE_URL}/api/incidents/{alias}")
        data = response.json()
        
        assert "prediction" in data, "Missing 'prediction' field"
        pred = data["prediction"]
        
        assert "decision" in pred, "Prediction missing 'decision'"
        assert pred["decision"] in ["Yes", "No", "Review"], f"Invalid decision: {pred['decision']}"
        
        assert "probability" in pred, "Prediction missing 'probability'"
        assert 0 <= pred["probability"] <= 1, f"Probability out of range: {pred['probability']}"
        
        assert "confidence" in pred, "Prediction missing 'confidence'"
        
        print(f"PASSED: GET /api/incidents/{alias} has valid prediction")
        print(f"  decision={pred['decision']}, probability={pred['probability']:.4f}")
    
    def test_live_incident_has_explanation(self):
        """Live-pulled incident should have explanation with text and model"""
        ticks_resp = requests.get(f"{BASE_URL}/api/live/ticks?limit=1")
        ticks_data = ticks_resp.json()
        
        if not ticks_data["ticks"]:
            pytest.skip("No ticks available")
        
        alias = ticks_data["ticks"][0]["sample"][0]["alias"]
        
        response = requests.get(f"{BASE_URL}/api/incidents/{alias}")
        data = response.json()
        
        assert "explanation" in data, "Missing 'explanation' field"
        exp = data["explanation"]
        
        if exp is not None:  # Explanation may be null if LLM failed
            assert "text" in exp, "Explanation missing 'text'"
            assert len(exp["text"]) > 0, "Explanation text is empty"
            
            assert "model" in exp, "Explanation missing 'model'"
            # Should be claude-haiku fallback since Ollama is not running
            assert "claude" in exp["model"].lower() or "haiku" in exp["model"].lower(), \
                f"Expected Claude Haiku fallback, got {exp['model']}"
            
            print(f"PASSED: GET /api/incidents/{alias} has valid explanation")
            print(f"  model={exp['model']}, text_len={len(exp['text'])}")
        else:
            print(f"PASSED: GET /api/incidents/{alias} has explanation=null (LLM may have failed)")


class TestRegressionCoreEndpoints:
    """Regression tests for core endpoints that should still work"""
    
    def test_health_still_works(self):
        """GET /api/health should return 200 with status=ok"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print(f"PASSED: GET /api/health returns status=ok")
    
    def test_kpis_still_works(self):
        """GET /api/kpis should return 200 with expected fields"""
        response = requests.get(f"{BASE_URL}/api/kpis")
        assert response.status_code == 200
        data = response.json()
        assert "total_open_events" in data
        assert "zoom_calls_predicted" in data
        print(f"PASSED: GET /api/kpis returns expected fields")
    
    def test_model_still_works(self):
        """GET /api/model should return 200 with loaded=true"""
        response = requests.get(f"{BASE_URL}/api/model")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] == True
        print(f"PASSED: GET /api/model returns loaded=true")
    
    def test_incidents_still_works(self):
        """GET /api/incidents should return 200 with rows"""
        response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "total" in data
        print(f"PASSED: GET /api/incidents returns {len(data['rows'])} rows, total={data['total']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
