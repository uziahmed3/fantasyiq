"""Load profile for the projections API.

Task weights mirror the real access pattern: the dashboard hammers /rankings, users
occasionally drill into a player, and a small share force a fresh inference. That mix is
what makes the cache hit rate meaningful - a load test that requests a different player
every time measures nothing but cold-path latency.

Run:
    locust -f loadtest/locustfile.py --host http://localhost:8000
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --headless -u 1000 -r 50 -t 3m --csv=loadtest/results
"""

import random

from locust import HttpUser, between, events, task

HOT_PLAYERS = list(range(1, 26))  # the players everyone looks up -> should hit cache
COLD_PLAYERS = list(range(26, 400))  # long tail -> cold path
OPPONENTS = ["GB", "CHI", "DET", "SF", "SEA", "DAL", "PHI", "KC"]
SEASON = 2023


class DashboardUser(HttpUser):
    wait_time = between(0.5, 2.5)

    @task(10)
    def rankings(self):
        self.client.get(
            f"/api/v1/rankings?week={random.randint(4, 12)}&season={SEASON}"
            f"&position={random.choice(['WR', 'RB', 'TE'])}&limit=25",
            name="/rankings",
        )

    @task(6)
    def hot_prediction(self):
        self.client.post(
            "/api/v1/predict",
            json={
                "player_id": random.choice(HOT_PLAYERS),
                "week": 6,
                "season": SEASON,
                "opponent": "GB",
            },
            name="/predict [hot]",
        )

    @task(3)
    def cold_prediction(self):
        self.client.post(
            "/api/v1/predict",
            json={
                "player_id": random.choice(COLD_PLAYERS),
                "week": random.randint(4, 14),
                "season": SEASON,
                "opponent": random.choice(OPPONENTS),
            },
            name="/predict [cold]",
        )

    @task(3)
    def player_detail(self):
        pid = random.choice(HOT_PLAYERS)
        self.client.get(f"/api/v1/players/{pid}", name="/players/{id}")
        self.client.get(
            f"/api/v1/players/{pid}/stats?season={SEASON}", name="/players/{id}/stats"
        )

    @task(1)
    def forced_refresh(self):
        self.client.post(
            "/api/v1/predict?refresh=true",
            json={
                "player_id": random.choice(HOT_PLAYERS),
                "week": 6,
                "season": SEASON,
                "opponent": "GB",
            },
            name="/predict [refresh]",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.test_stop.add_listener
def _summary(environment, **_kwargs):
    stats = environment.stats.total
    print("\n--- load test summary ---")
    print(f"requests      : {stats.num_requests}")
    print(f"failures      : {stats.num_failures} " f"({stats.fail_ratio * 100:.2f}%)")
    print(f"throughput    : {stats.total_rps:.1f} req/s")
    print(f"median        : {stats.median_response_time} ms")
    print(f"p95           : {stats.get_response_time_percentile(0.95)} ms")
    print(f"p99           : {stats.get_response_time_percentile(0.99)} ms")
    hot = environment.stats.get("/predict [hot]", "POST")
    cold = environment.stats.get("/predict [cold]", "POST")
    if hot.num_requests and cold.num_requests:
        print(f"\ncached p95    : {hot.get_response_time_percentile(0.95)} ms")
        print(f"uncached p95  : {cold.get_response_time_percentile(0.95)} ms")
