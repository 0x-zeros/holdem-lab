"""Tests for cards endpoints."""


def test_get_canonical_hands(client):
    """Test getting all canonical hands."""
    response = client.get("/api/canonical")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 169
    assert len(data["hands"]) == 169

    # Check first hand (should be AA)
    aa = next(h for h in data["hands"] if h["notation"] == "AA")
    assert aa["is_pair"] is True
    assert aa["num_combos"] == 6
    assert aa["matrix_row"] == 0
    assert aa["matrix_col"] == 0

    # Check AKs
    aks = next(h for h in data["hands"] if h["notation"] == "AKs")
    assert aks["suited"] is True
    assert aks["num_combos"] == 4

    # Check AKo
    ako = next(h for h in data["hands"] if h["notation"] == "AKo")
    assert ako["suited"] is False
    assert ako["num_combos"] == 12


def test_parse_cards_valid(client):
    """Test parsing valid cards."""
    response = client.post("/api/parse-cards", json={"input": "AhKh"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert len(data["cards"]) == 2
    assert data["cards"][0]["notation"] == "Ah"
    assert data["cards"][0]["suit_symbol"] == "♥"


def test_parse_cards_with_spaces(client):
    """Test parsing cards with spaces."""
    response = client.post("/api/parse-cards", json={"input": "Ah Kh Qd"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert len(data["cards"]) == 3


def test_parse_cards_invalid(client):
    """Test parsing invalid cards."""
    response = client.post("/api/parse-cards", json={"input": "XxYy"})
    assert response.status_code == 400
