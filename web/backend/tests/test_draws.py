"""Tests for draws endpoint."""


def test_flush_draw(client):
    """Test flush draw detection."""
    response = client.post("/api/draws", json={
        "hole_cards": ["Ah", "Kh"],
        "board": ["7h", "6h", "2c"]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["has_flush"] is False
    assert len(data["flush_draws"]) == 1
    assert data["flush_draws"][0]["out_count"] == 9
    assert data["flush_draws"][0]["is_nut"] is True


def test_straight_draw_oesd(client):
    """Test open-ended straight draw detection."""
    response = client.post("/api/draws", json={
        "hole_cards": ["9h", "8c"],
        "board": ["7d", "6s", "2h"]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["has_straight"] is False
    # Should have at least one straight draw
    assert len(data["straight_draws"]) >= 1


def test_combo_draw(client):
    """Test combo draw (flush + straight)."""
    response = client.post("/api/draws", json={
        "hole_cards": ["9h", "8h"],
        "board": ["7h", "6c", "2h"]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["is_combo_draw"] is True
    assert len(data["flush_draws"]) >= 1
    assert len(data["straight_draws"]) >= 1
    assert data["total_outs"] > 0


def test_made_flush(client):
    """Test when player has made flush."""
    response = client.post("/api/draws", json={
        "hole_cards": ["Ah", "Kh"],
        "board": ["Qh", "Jh", "2h"]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["has_flush"] is True
    assert len(data["flush_draws"]) == 0  # No draw when made


def test_made_straight(client):
    """Test when player has made straight."""
    response = client.post("/api/draws", json={
        "hole_cards": ["9h", "8c"],
        "board": ["7d", "6s", "5h"]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["has_straight"] is True
    assert len(data["straight_draws"]) == 0  # No draw when made


def test_draws_invalid_hole_cards(client):
    """Test with invalid number of hole cards."""
    response = client.post("/api/draws", json={
        "hole_cards": ["Ah"],
        "board": ["7h", "6h", "2c"]
    })
    assert response.status_code == 422  # Validation error


def test_draws_invalid_board(client):
    """Test with invalid board size."""
    response = client.post("/api/draws", json={
        "hole_cards": ["Ah", "Kh"],
        "board": ["7h", "6h"]  # Too few cards
    })
    assert response.status_code == 422  # Validation error
