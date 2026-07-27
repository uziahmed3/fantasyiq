def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "fantasyiq_predictions_total" in r.text or "http_requests" in r.text


def test_get_player(client, seed):
    r = client.get("/api/v1/players/15")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Justin Jefferson"
    assert body["team"] == "MIN"


def test_get_player_404(client, seed):
    assert client.get("/api/v1/players/99999").status_code == 404


def test_list_players_filters_by_position(client, seed):
    r = client.get("/api/v1/players", params={"position": "WR", "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert {p["name"] for p in body["items"]} == {"Justin Jefferson", "Ja'Marr Chase"}

    assert client.get("/api/v1/players", params={"position": "QB"}).json()["total"] == 0


def test_list_players_name_search(client, seed):
    r = client.get("/api/v1/players", params={"name": "jeffer"})
    assert [p["id"] for p in r.json()["items"]] == [15]


def test_player_stats_desc_by_week(client, seed):
    r = client.get("/api/v1/players/15/stats", params={"season": 2023})
    assert r.status_code == 200
    rows = r.json()
    assert [row["week"] for row in rows] == [4, 3, 2, 1]
    assert rows[0]["fantasy_points"] == 11.2


def test_rankings_reads_prediction_table(client, seed):
    r = client.get("/api/v1/rankings", params={"week": 5, "season": 2023, "position": "WR"})
    assert r.status_code == 200
    body = r.json()
    assert [row["name"] for row in body["rankings"]] == ["Ja'Marr Chase", "Justin Jefferson"]
    assert body["rankings"][0]["rank"] == 1
    assert body["rankings"][0]["projected_points"] == 22.1


def test_validation_rejects_out_of_range_week(client, seed):
    r = client.post("/api/v1/predict", json={"player_id": 15, "week": 99, "opponent": "GB"})
    assert r.status_code == 422
