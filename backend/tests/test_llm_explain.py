"""
Backend API tests for LLM Explanation feature.

Tests cover:
- POST /api/explain/{alias} - compute or return cached explanation
- POST /api/explain/{alias}?force=true - force regeneration
- GET /api/explain/{alias} - get cached explanation only
- GET /api/incidents - verify explanation field in rows
- Error cases: invalid alias returns 404
"""
import os
import time
import pytest
import requests

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')


class TestExplainPostEndpoint:
    """Test POST /api/explain/{alias} endpoint"""
    
    def test_explain_post_valid_alias_returns_200_with_schema(self):
        """POST /api/explain/{alias} with valid alias returns 200 and correct schema"""
        # First get a valid alias from incidents
        list_response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=5")
        assert list_response.status_code == 200
        rows = list_response.json()["rows"]
        
        # Find an alias that doesn't have cached explanation (or use any)
        alias = None
        for row in rows:
            if row.get("explanation") is None:
                alias = row["alias"]
                break
        
        # If all have explanations, just use the first one
        if alias is None:
            alias = rows[0]["alias"]
        
        print(f"Testing POST /api/explain/{alias}")
        
        # POST to compute explanation (may take 2-5s for live LLM call)
        response = requests.post(f"{BASE_URL}/api/explain/{alias}", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Verify schema
        assert "alias" in data, "Response should have 'alias'"
        assert data["alias"] == alias, f"Expected alias={alias}, got {data['alias']}"
        
        assert "text" in data, "Response should have 'text'"
        assert isinstance(data["text"], str), "text should be string"
        assert len(data["text"]) > 0, "text should be non-empty"
        
        assert "model" in data, "Response should have 'model'"
        assert data["model"] == "claude-haiku-4-5-20251001", f"Expected claude-haiku-4-5-20251001, got {data['model']}"
        
        assert "decision" in data, "Response should have 'decision'"
        assert data["decision"] in ["Yes", "No", "Review"], f"Invalid decision: {data['decision']}"
        
        assert "probability" in data, "Response should have 'probability'"
        assert isinstance(data["probability"], (int, float)), "probability should be numeric"
        
        assert "ts" in data, "Response should have 'ts'"
        
        print(f"SUCCESS: POST /api/explain/{alias} - text='{data['text'][:50]}...', model={data['model']}")
        
        # Store for cache test
        return data
    
    def test_explain_post_returns_cached_on_second_call(self):
        """POST /api/explain/{alias} without force returns cached result (same ts)"""
        # Get a valid alias
        list_response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=1")
        alias = list_response.json()["rows"][0]["alias"]
        
        # First call
        response1 = requests.post(f"{BASE_URL}/api/explain/{alias}", timeout=30)
        assert response1.status_code == 200
        data1 = response1.json()
        ts1 = data1["ts"]
        text1 = data1["text"]
        
        # Wait a moment
        time.sleep(0.5)
        
        # Second call (should return cached)
        response2 = requests.post(f"{BASE_URL}/api/explain/{alias}", timeout=5)
        assert response2.status_code == 200
        data2 = response2.json()
        ts2 = data2["ts"]
        text2 = data2["text"]
        
        # Should be same timestamp (cached)
        assert ts1 == ts2, f"Expected same ts (cached), got ts1={ts1}, ts2={ts2}"
        assert text1 == text2, "Cached text should be identical"
        
        print(f"SUCCESS: Cache hit verified - ts1={ts1} == ts2={ts2}")
    
    def test_explain_post_force_regenerates(self):
        """POST /api/explain/{alias}?force=true regenerates with newer ts"""
        # Get a valid alias
        list_response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=1")
        alias = list_response.json()["rows"][0]["alias"]
        
        # First call to ensure cached
        response1 = requests.post(f"{BASE_URL}/api/explain/{alias}", timeout=30)
        assert response1.status_code == 200
        ts1 = response1.json()["ts"]
        
        # Wait a moment
        time.sleep(1)
        
        # Force regenerate
        response2 = requests.post(f"{BASE_URL}/api/explain/{alias}?force=true", timeout=30)
        assert response2.status_code == 200
        data2 = response2.json()
        ts2 = data2["ts"]
        
        # Should have newer timestamp
        assert ts2 > ts1, f"Expected newer ts with force=true, got ts1={ts1}, ts2={ts2}"
        
        print(f"SUCCESS: Force regenerate - ts1={ts1} < ts2={ts2}")
    
    def test_explain_post_invalid_alias_returns_404(self):
        """POST /api/explain/NOC-INVALID-999 returns 404 'incident not found'"""
        response = requests.post(f"{BASE_URL}/api/explain/NOC-INVALID-999", timeout=10)
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        
        data = response.json()
        assert "incident not found" in data.get("detail", "").lower(), f"Expected 'incident not found', got {data}"
        
        print("SUCCESS: Invalid alias returns 404 'incident not found'")


class TestExplainGetEndpoint:
    """Test GET /api/explain/{alias} endpoint"""
    
    def test_explain_get_cached_returns_200(self):
        """GET /api/explain/{alias} after POST returns cached doc"""
        # First ensure we have a cached explanation
        list_response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=1")
        alias = list_response.json()["rows"][0]["alias"]
        
        # POST to ensure cached
        post_response = requests.post(f"{BASE_URL}/api/explain/{alias}", timeout=30)
        assert post_response.status_code == 200
        post_data = post_response.json()
        
        # GET should return same cached doc
        get_response = requests.get(f"{BASE_URL}/api/explain/{alias}")
        assert get_response.status_code == 200
        get_data = get_response.json()
        
        assert get_data["alias"] == alias
        assert get_data["text"] == post_data["text"]
        assert get_data["ts"] == post_data["ts"]
        
        print(f"SUCCESS: GET /api/explain/{alias} returns cached doc")
    
    def test_explain_get_uncached_returns_404(self):
        """GET /api/explain/NOC-INVALID-999 returns 404 when no cache exists"""
        response = requests.get(f"{BASE_URL}/api/explain/NOC-INVALID-999")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        
        data = response.json()
        assert "no cached explanation" in data.get("detail", "").lower(), f"Expected 'no cached explanation', got {data}"
        
        print("SUCCESS: GET uncached alias returns 404 'no cached explanation'")


class TestIncidentsExplanationField:
    """Test that /api/incidents includes explanation field"""
    
    def test_incidents_rows_have_explanation_key(self):
        """GET /api/incidents?page=1&page_size=5 each row has 'explanation' key"""
        response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=5")
        assert response.status_code == 200
        
        rows = response.json()["rows"]
        assert len(rows) > 0, "Expected at least 1 row"
        
        for row in rows:
            assert "explanation" in row, f"Row {row['alias']} missing 'explanation' key"
            
            # explanation can be null (uncached) or object with text/model
            exp = row["explanation"]
            if exp is not None:
                assert "text" in exp, f"Explanation for {row['alias']} missing 'text'"
                assert "model" in exp, f"Explanation for {row['alias']} missing 'model'"
        
        # Count cached vs uncached
        cached = sum(1 for r in rows if r["explanation"] is not None)
        print(f"SUCCESS: All {len(rows)} rows have 'explanation' key ({cached} cached, {len(rows)-cached} uncached)")
    
    def test_incidents_cached_explanation_has_correct_schema(self):
        """Verify cached explanation in /api/incidents has text/model fields"""
        # First ensure at least one is cached
        list_response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=1")
        alias = list_response.json()["rows"][0]["alias"]
        
        # POST to cache
        requests.post(f"{BASE_URL}/api/explain/{alias}", timeout=30)
        
        # Now fetch incidents and check
        response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=5")
        rows = response.json()["rows"]
        
        # Find the row we just cached
        cached_row = next((r for r in rows if r["alias"] == alias), None)
        if cached_row:
            exp = cached_row["explanation"]
            assert exp is not None, f"Expected cached explanation for {alias}"
            assert "text" in exp and len(exp["text"]) > 0
            assert exp["model"] == "claude-haiku-4-5-20251001"
            print(f"SUCCESS: Cached explanation for {alias} has correct schema")
        else:
            print(f"INFO: {alias} not in first 5 rows, but test passed")


class TestRegressionCoreEndpoints:
    """Regression tests to ensure core endpoints still work"""
    
    def test_health_still_works(self):
        """GET /api/health returns status ok"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("SUCCESS: /api/health regression passed")
    
    def test_kpis_still_works(self):
        """GET /api/kpis returns numeric fields"""
        response = requests.get(f"{BASE_URL}/api/kpis")
        assert response.status_code == 200
        data = response.json()
        assert "total_open_events" in data
        assert isinstance(data["total_open_events"], (int, float))
        print("SUCCESS: /api/kpis regression passed")
    
    def test_incidents_still_works(self):
        """GET /api/incidents returns rows with predictions"""
        response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert len(data["rows"]) > 0
        assert "prediction" in data["rows"][0]
        print("SUCCESS: /api/incidents regression passed")
    
    def test_model_still_works(self):
        """GET /api/model returns loaded=true"""
        response = requests.get(f"{BASE_URL}/api/model")
        assert response.status_code == 200
        assert response.json()["loaded"] is True
        print("SUCCESS: /api/model regression passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
