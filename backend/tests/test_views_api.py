"""
Backend API tests for NOC Views Endpoints (Iteration 2).

Tests cover the 6 NEW view endpoints:
- GET /api/views/customer - Customer/reporter aggregation
- GET /api/views/instance - Region/instance aggregation  
- GET /api/views/alarm-lens - Severity x Region heatmap matrix
- GET /api/views/cluster-events - Multi-region clustered events
- GET /api/views/service-requests - SEV3/SEV4 service request tickets
- GET /api/views/blackouts - Scheduled maintenance windows

Also includes regression tests for core endpoints.
"""
import os
import pytest
import requests

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')


class TestViewsCustomer:
    """Test /api/views/customer endpoint"""
    
    def test_customer_view_returns_total_and_rows(self):
        """GET /api/views/customer returns total>=1 with rows"""
        response = requests.get(f"{BASE_URL}/api/views/customer")
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert data["total"] >= 1, f"Expected total>=1, got {data['total']}"
        assert "rows" in data
        assert len(data["rows"]) >= 1
        
        print(f"SUCCESS: Customer view - total={data['total']}, rows={len(data['rows'])}")
    
    def test_customer_view_row_schema(self):
        """Each row has required keys: reporter, queue, total, open, sev1..sev4, regions, zoom_yes"""
        response = requests.get(f"{BASE_URL}/api/views/customer")
        assert response.status_code == 200
        data = response.json()
        
        required_keys = ["reporter", "queue", "total", "open", "sev1", "sev2", "sev3", "sev4", "regions", "zoom_yes"]
        
        for row in data["rows"][:5]:  # Check first 5 rows
            for key in required_keys:
                assert key in row, f"Missing key '{key}' in customer row: {row}"
            
            # Type checks
            assert isinstance(row["total"], int)
            assert isinstance(row["open"], int)
            assert isinstance(row["sev1"], int)
            assert isinstance(row["sev2"], int)
            assert isinstance(row["sev3"], int)
            assert isinstance(row["sev4"], int)
            assert isinstance(row["regions"], int)
            assert isinstance(row["zoom_yes"], int)
        
        print(f"SUCCESS: Customer view row schema validated for {min(5, len(data['rows']))} rows")


class TestViewsInstance:
    """Test /api/views/instance endpoint"""
    
    def test_instance_view_returns_15_regions(self):
        """GET /api/views/instance returns total==15 (one per region)"""
        response = requests.get(f"{BASE_URL}/api/views/instance")
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert data["total"] == 15, f"Expected 15 regions, got {data['total']}"
        assert len(data["rows"]) == 15
        
        print(f"SUCCESS: Instance view - total={data['total']} regions")
    
    def test_instance_view_row_schema(self):
        """Each row has: region, total, open, sev1..sev4, multi_region, avg_age_min, zoom_yes"""
        response = requests.get(f"{BASE_URL}/api/views/instance")
        assert response.status_code == 200
        data = response.json()
        
        required_keys = ["region", "total", "open", "sev1", "sev2", "sev3", "sev4", "multi_region", "avg_age_min", "zoom_yes"]
        
        for row in data["rows"]:
            for key in required_keys:
                assert key in row, f"Missing key '{key}' in instance row: {row}"
            
            # Type checks
            assert isinstance(row["region"], str)
            assert isinstance(row["total"], int)
            assert isinstance(row["open"], int)
            assert isinstance(row["multi_region"], int)
            assert isinstance(row["avg_age_min"], (int, float))
            assert isinstance(row["zoom_yes"], int)
        
        print(f"SUCCESS: Instance view row schema validated for all {len(data['rows'])} rows")


class TestViewsAlarmLens:
    """Test /api/views/alarm-lens endpoint"""
    
    def test_alarm_lens_returns_matrix_structure(self):
        """GET /api/views/alarm-lens returns regions list, severities, and matrix"""
        response = requests.get(f"{BASE_URL}/api/views/alarm-lens")
        assert response.status_code == 200
        data = response.json()
        
        assert "regions" in data
        assert "severities" in data
        assert "matrix" in data
        
        # Check severities are exactly SEV1-SEV4
        assert data["severities"] == ["SEV1", "SEV2", "SEV3", "SEV4"], f"Expected SEV1-4, got {data['severities']}"
        
        # Check regions list is non-empty
        assert len(data["regions"]) > 0, "Expected at least one region"
        
        print(f"SUCCESS: Alarm lens - {len(data['regions'])} regions, {len(data['severities'])} severities")
    
    def test_alarm_lens_matrix_cell_counts(self):
        """matrix[sev][region].count is int for each cell"""
        response = requests.get(f"{BASE_URL}/api/views/alarm-lens")
        assert response.status_code == 200
        data = response.json()
        
        matrix = data["matrix"]
        
        for sev in data["severities"]:
            assert sev in matrix, f"Missing severity {sev} in matrix"
            for region in data["regions"]:
                if region in matrix[sev]:
                    cell = matrix[sev][region]
                    assert "count" in cell, f"Missing 'count' in cell [{sev}][{region}]"
                    assert isinstance(cell["count"], int), f"count should be int, got {type(cell['count'])}"
        
        print(f"SUCCESS: Alarm lens matrix cell counts validated")


class TestViewsClusterEvents:
    """Test /api/views/cluster-events endpoint"""
    
    def test_cluster_events_returns_multi_region_only(self):
        """GET /api/views/cluster-events returns rows where multi_region==1"""
        response = requests.get(f"{BASE_URL}/api/views/cluster-events?page=1&page_size=25")
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert data["total"] >= 1, f"Expected at least 1 cluster event, got {data['total']}"
        assert "rows" in data
        
        for row in data["rows"]:
            assert row["multi_region"] == 1, f"Expected multi_region==1, got {row['multi_region']}"
        
        print(f"SUCCESS: Cluster events - total={data['total']}, all have multi_region==1")
    
    def test_cluster_events_have_prediction(self):
        """Each cluster event row has a prediction object"""
        response = requests.get(f"{BASE_URL}/api/views/cluster-events?page=1&page_size=25")
        assert response.status_code == 200
        data = response.json()
        
        for row in data["rows"][:10]:  # Check first 10
            assert "prediction" in row, f"Missing prediction in row: {row['alias']}"
            if row["prediction"]:  # May be None if not scored yet
                pred = row["prediction"]
                assert "decision" in pred
                assert pred["decision"] in ["Yes", "No", "Review"]
        
        print(f"SUCCESS: Cluster events have prediction objects")


class TestViewsServiceRequests:
    """Test /api/views/service-requests endpoint"""
    
    def test_service_requests_returns_sev3_sev4_only(self):
        """GET /api/views/service-requests returns rows with severity in SEV3/SEV4"""
        response = requests.get(f"{BASE_URL}/api/views/service-requests?page=1&page_size=25")
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert "rows" in data
        
        for row in data["rows"]:
            assert row["severity"] in ["SEV3", "SEV4"], f"Expected SEV3/SEV4, got {row['severity']}"
        
        print(f"SUCCESS: Service requests - total={data['total']}, all SEV3/SEV4")
    
    def test_service_requests_have_sr_id_and_priority(self):
        """Each row has sr_id starting 'SR-' and priority in P3/P4"""
        response = requests.get(f"{BASE_URL}/api/views/service-requests?page=1&page_size=25")
        assert response.status_code == 200
        data = response.json()
        
        for row in data["rows"][:10]:  # Check first 10
            assert "sr_id" in row, f"Missing sr_id in row"
            assert row["sr_id"].startswith("SR-"), f"sr_id should start with 'SR-', got {row['sr_id']}"
            
            assert "priority" in row, f"Missing priority in row"
            assert row["priority"] in ["P3", "P4"], f"Expected P3/P4, got {row['priority']}"
        
        print(f"SUCCESS: Service requests have valid sr_id and priority")


class TestViewsBlackouts:
    """Test /api/views/blackouts endpoint"""
    
    def test_blackouts_returns_total_and_rows(self):
        """GET /api/views/blackouts returns total>=1 with rows"""
        response = requests.get(f"{BASE_URL}/api/views/blackouts")
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert data["total"] >= 1, f"Expected total>=1, got {data['total']}"
        assert "rows" in data
        assert len(data["rows"]) >= 1
        
        print(f"SUCCESS: Blackouts - total={data['total']}")
    
    def test_blackouts_row_schema(self):
        """Each row has id 'BO-####', status in ACTIVE/UPCOMING/COMPLETED, impact in Low/Medium/High"""
        response = requests.get(f"{BASE_URL}/api/views/blackouts")
        assert response.status_code == 200
        data = response.json()
        
        for row in data["rows"]:
            # Check id format
            assert "id" in row
            assert row["id"].startswith("BO-"), f"id should start with 'BO-', got {row['id']}"
            
            # Check status
            assert "status" in row
            assert row["status"] in ["ACTIVE", "UPCOMING", "COMPLETED"], f"Invalid status: {row['status']}"
            
            # Check impact
            assert "impact" in row
            assert row["impact"] in ["Low", "Medium", "High"], f"Invalid impact: {row['impact']}"
        
        print(f"SUCCESS: Blackouts row schema validated for {len(data['rows'])} rows")


class TestRegressionCoreEndpoints:
    """Regression tests for core endpoints (quick sanity checks)"""
    
    def test_health_still_works(self):
        """GET /api/health returns 200 with status=ok"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("SUCCESS: /api/health regression passed")
    
    def test_kpis_still_works(self):
        """GET /api/kpis returns 200 with required fields"""
        response = requests.get(f"{BASE_URL}/api/kpis")
        assert response.status_code == 200
        data = response.json()
        assert "total_open_events" in data
        assert "zoom_calls_predicted" in data
        print("SUCCESS: /api/kpis regression passed")
    
    def test_incidents_still_works(self):
        """GET /api/incidents returns 200 with rows"""
        response = requests.get(f"{BASE_URL}/api/incidents?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert len(data["rows"]) > 0
        print("SUCCESS: /api/incidents regression passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
