"""Tests for equity endpoint."""


def test_equity_two_players_specific_hands(client):
    """Test equity calculation with two specific hands."""
    response = client.post("/api/equity", json={
        "players": [
            {"cards": ["Ah", "Kh"]},
            {"cards": ["Qd", "Qc"]}
        ],
        "num_simulations": 1000
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["players"]) == 2
    assert data["players"][0]["index"] == 0
    assert data["players"][1]["index"] == 1

    # Check equity sums to ~1.0
    total_equity = sum(p["equity"] for p in data["players"])
    assert 0.99 <= total_equity <= 1.01


def test_equity_with_board(client):
    """Test equity calculation with board cards."""
    response = client.post("/api/equity", json={
        "players": [
            {"cards": ["Ah", "Kh"]},
            {"cards": ["Qd", "Qc"]}
        ],
        "board": ["7h", "6h", "2c"],
        "num_simulations": 1000
    })
    assert response.status_code == 200
    data = response.json()

    # With flush draw, AhKh should have higher equity
    assert data["players"][0]["equity"] > 0.4


def test_equity_invalid_too_few_players(client):
    """Test equity with too few players."""
    response = client.post("/api/equity", json={
        "players": [{"cards": ["Ah", "Kh"]}],
        "num_simulations": 1000
    })
    assert response.status_code == 422  # Validation error


def test_equity_response_structure(client):
    """Test equity response has correct structure."""
    response = client.post("/api/equity", json={
        "players": [
            {"cards": ["Ah", "Kh"]},
            {"cards": ["Qd", "Qc"]}
        ],
        "num_simulations": 100
    })
    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "players" in data
    assert "total_simulations" in data
    assert "elapsed_ms" in data

    # Check player result structure
    player = data["players"][0]
    assert "index" in player
    assert "hand_description" in player
    assert "equity" in player
    assert "win_rate" in player
    assert "tie_rate" in player
    assert "combos" in player
