from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_is_not_reversible():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip():
    token = create_access_token("uzair@example.com")
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "uzair@example.com"


def test_tampered_jwt_rejected():
    token = create_access_token("uzair@example.com")
    assert decode_access_token(token[:-4] + "AAAA") is None


def test_register_login_me_flow(client):
    r = client.post("/api/v1/auth/register", json={"email": "a@b.com", "password": "supersecret1"})
    assert r.status_code == 201

    dupe = client.post(
        "/api/v1/auth/register", json={"email": "a@b.com", "password": "supersecret1"}
    )
    assert dupe.status_code == 409

    tok = client.post(
        "/api/v1/auth/token", data={"username": "a@b.com", "password": "supersecret1"}
    )
    assert tok.status_code == 200
    access = tok.json()["access_token"]

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert me.status_code == 200
    assert me.json()["email"] == "a@b.com"


def test_me_requires_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401


def test_bad_password_is_401_not_404(client):
    client.post("/api/v1/auth/register", json={"email": "c@d.com", "password": "supersecret1"})
    r = client.post("/api/v1/auth/token", data={"username": "c@d.com", "password": "nope"})
    assert r.status_code == 401
    # identical response for a non-existent user - no account enumeration
    r2 = client.post("/api/v1/auth/token", data={"username": "nobody@x.com", "password": "nope"})
    assert r2.status_code == 401 and r2.json() == r.json()
