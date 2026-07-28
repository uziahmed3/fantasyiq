#!/usr/bin/env python3
"""Run FantasyIQ locally with Python only - no Docker, Postgres, Redis or Node.

    python local.py --demo        synthetic data, no network, fastest way to see it work
    python local.py               real NFL data
    python local.py --help        everything else

Same application code as the Docker stack. Two substitutions, both chosen by
environment variable rather than a code fork:

    Postgres  ->  SQLite file (fantasyiq.db)
    Redis     ->  in-process TTL cache

`GET /info` on the running API reports which combination is live.

Deliberately written in Python rather than PowerShell or bash: this file is the entry
point everything else depends on, and Python is the one language guaranteed present
(you are running it) and testable on any platform. Uses only the standard library so it
works before the virtualenv exists.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DB_FILE = ROOT / "fantasyiq.db"
MODEL_DIR = ROOT / ".models"
MANUAL_DIR = ROOT / "data" / "manual"
PIPELINE_DATA = ROOT / ".pipeline-data"
STAMP = VENV / ".deps-installed"

IS_WINDOWS = platform.system() == "Windows"

# Normally we build and use .venv. FANTASYIQ_PYTHON lets you point at an interpreter you
# already have (a conda env, a company-managed Python with the deps preinstalled, or the
# system interpreter in CI) and skip virtualenv creation entirely.
_EXTERNAL_PY = os.environ.get("FANTASYIQ_PYTHON")
USING_EXTERNAL_PY = bool(_EXTERNAL_PY)
VENV_PY = (
    Path(_EXTERNAL_PY)
    if _EXTERNAL_PY
    else VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
)

REQUIREMENTS = [
    ROOT / "backend" / "requirements.txt",
    ROOT / "ml-service" / "requirements.txt",
    ROOT / "pipeline" / "requirements.txt",
]


# --------------------------------------------------------------------------- output
class C:
    """ANSI colours, disabled when the terminal will not render them."""

    _on = sys.stdout.isatty() and (
        not IS_WINDOWS or os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")
    )
    CYAN = "\033[36m" if _on else ""
    GREEN = "\033[32m" if _on else ""
    YELLOW = "\033[33m" if _on else ""
    RED = "\033[31m" if _on else ""
    DIM = "\033[2m" if _on else ""
    OFF = "\033[0m" if _on else ""


def step(msg: str) -> None:
    print(f"\n{C.CYAN}==> {msg}{C.OFF}", flush=True)


def ok(msg: str) -> None:
    print(f"    {C.GREEN}{msg}{C.OFF}", flush=True)


def warn(msg: str) -> None:
    print(f"    {C.YELLOW}{msg}{C.OFF}", flush=True)


def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[valid-type] # noqa: F821
    print(f"\n{C.RED}ERROR: {msg}{C.OFF}\n", flush=True)
    sys.exit(code)


# --------------------------------------------------------------------------- env
def child_env(ml_port: int) -> dict[str, str]:
    """Environment for every child process. This is where SQLite and the in-memory
    cache get selected; nothing in the application code knows the difference."""
    env = dict(os.environ)
    # Internal service calls must not go through a corporate proxy.
    for var in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(var, None)
    env["NO_PROXY"] = "localhost,127.0.0.1"

    env.update(
        {
            # SQLAlchemy needs forward slashes even on Windows.
            "DATABASE_URL_OVERRIDE": "sqlite+pysqlite:///" + DB_FILE.as_posix(),
            "REDIS_URL": "memory://",
            "MODEL_DIR": str(MODEL_DIR),
            "MANUAL_DATA_DIR": str(MANUAL_DIR),
            "PIPELINE_DATA_DIR": str(PIPELINE_DATA),
            "ML_SERVICE_URL": f"http://127.0.0.1:{ml_port}",
            "ENVIRONMENT": "local",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.setdefault("ACTIVE_MODEL_VERSION", "xgboost_v1")
    if not env.get("JWT_SECRET_KEY"):
        env["JWT_SECRET_KEY"] = os.urandom(32).hex()
    return env


def run(
    args: list[str], cwd: Path, env: dict[str, str], label: str, fatal: bool = True
) -> int:
    result = subprocess.run([str(a) for a in args], cwd=str(cwd), env=env)
    if result.returncode != 0 and fatal:
        die(f"{label} failed (exit {result.returncode}). See the output above.")
    return result.returncode


# --------------------------------------------------------------------------- setup
def ensure_venv(force: bool) -> None:
    if USING_EXTERNAL_PY:
        ok(f"Using the interpreter from FANTASYIQ_PYTHON: {VENV_PY}")
        if not VENV_PY.exists():
            die(f"FANTASYIQ_PYTHON points at {VENV_PY}, which does not exist.")
        return
    if force and VENV.exists():
        step("Removing existing virtualenv")
        shutil.rmtree(VENV, ignore_errors=True)
    if not VENV_PY.exists():
        step("Creating virtualenv (.venv)")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
        if not VENV_PY.exists():
            die(f"virtualenv creation failed - expected {VENV_PY}")
        ok("Created.")


def _requirements(skip_torch: bool) -> list[Path]:
    """The requirements files to install.

    --skip-torch writes filtered copies to a temp dir rather than using a pip
    constraints file: a constraint restricts which version may be chosen, it cannot
    remove a requirement, so constraining torch would fail the install rather than
    skip it.
    """
    if not skip_torch:
        return REQUIREMENTS
    tmp = Path(tempfile.mkdtemp(prefix="fantasyiq-req-"))
    out = []
    for req in REQUIREMENTS:
        lines = [
            ln
            for ln in req.read_text().splitlines()
            if not ln.strip().lower().startswith("torch")
        ]
        dest = tmp / f"{req.parent.name}.txt"
        dest.write_text("\n".join(lines) + "\n")
        out.append(dest)
    return out


def ensure_deps(env: dict[str, str], skip_torch: bool = False) -> None:
    if USING_EXTERNAL_PY:
        ok("Skipping dependency install - you supplied the interpreter.")
        return
    if STAMP.exists():
        ok("Dependencies already installed (--reinstall to redo).")
        return
    step("Installing dependencies - 3 to 6 minutes on first run")
    run(
        [VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        ROOT,
        env,
        "pip upgrade",
    )

    cmd = [
        VENV_PY,
        "-m",
        "pip",
        "install",
        # Linux needs this index for the CPU-only torch build; Windows and macOS get a
        # CPU wheel straight from PyPI. Harmless on every platform.
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    ]
    for req in _requirements(skip_torch):
        cmd += ["-r", str(req)]
    if skip_torch:
        warn(
            "Skipping PyTorch. Ridge and XGBoost still train; the bake-off drops torch_v1."
        )

    # Streamed to the terminal (progress matters on a 200MB download) and captured, so
    # the failure message can distinguish a dependency conflict from a network problem
    # instead of guessing.
    proc = subprocess.Popen(
        [str(a) for a in cmd],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    if proc.wait() != 0:
        blob = "".join(captured)
        if "ResolutionImpossible" in blob or "conflicting dependencies" in blob:
            die(
                "pip could not resolve the dependencies - two packages want incompatible\n"
                "versions of the same library. The conflict is named a few lines above.\n"
                "This is a bug in the requirements files, not your machine; please report it."
            )
        if (
            "ProxyError" in blob
            or "Tunnel connection failed" in blob
            or "Network is" in blob
        ):
            die(
                "pip could not reach the package index - this is the network, not the code.\n\n"
                "  set PIP_INDEX_URL to your internal PyPI mirror, or\n"
                "  ask IT for the pip proxy settings, then re-run."
            )
        if "torch" in blob.lower():
            die(
                "pip install failed while fetching PyTorch (a ~200MB download).\n\n"
                "Skip it - the ridge and XGBoost models still train and serve:\n"
                "  python local.py --skip-torch --demo"
            )
        die("pip install failed. The error is above.")
    STAMP.write_text("ok")
    ok("Dependencies installed.")


# --------------------------------------------------------------------------- stages
def migrate(env: dict[str, str]) -> None:
    step(f"Applying migrations to {DB_FILE.name}")
    run(
        [VENV_PY, "-m", "alembic", "upgrade", "head"],
        ROOT / "backend",
        env,
        "Migrations",
    )
    ok("Schema ready.")


def load_demo(env: dict[str, str]) -> None:
    step("Loading synthetic demo data (no network)")
    run([VENV_PY, "-m", "seed_demo"], ROOT / "pipeline", env, "Demo seed")


def load_real(env: dict[str, str], source: str) -> None:
    step(f"Loading NFL data (source: {source})")
    if source != "manual":
        warn("First run downloads a few hundred MB. 2-5 minutes.")
    rc = run(
        [VENV_PY, "-m", "run_weekly", "--skip-score", "--source", source],
        ROOT / "pipeline",
        env,
        "Ingest",
        fatal=False,
    )
    if rc != 0:
        die(
            "Could not load NFL data - your network is probably blocking the download.\n\n"
            "  1) Offline:  python local.py --data-urls\n"
            "               download those files in your browser into data/manual/\n"
            "               python local.py --offline\n\n"
            "  2) Hotspot:  run once on your phone's hotspot; the data then persists\n"
            "               in fantasyiq.db\n\n"
            "  3) Demo:     python local.py --demo    (synthetic data, works right now)"
        )
    ok("Data loaded.")


def build_context(env: dict[str, str], demo: bool) -> None:
    """Assemble preseason context for every season we have.

    Must run BEFORE training: the preseason model trains on context rows, and an earlier
    version of this script trained first and then built context inside the projection
    step - so the preseason model always reported "no training rows" on a fresh run.

    In demo mode the optional feeds are skipped. Synthetic players have made-up ids that
    cannot match real snap-count or depth-chart records, so downloading them costs time
    and matches nothing.
    """
    step("Building preseason context (prior season, role, draft capital)")
    cmd = [VENV_PY, "-m", "context", "--all"]
    if demo:
        cmd.append("--no-optional-feeds")
    if run(cmd, ROOT / "pipeline", env, "Context build", fatal=False) != 0:
        warn(
            "Context build failed - week-1 and rookie projections will be unavailable."
        )


def train(env: dict[str, str]) -> None:
    step("Training models on the loaded data")
    ml = ROOT / "ml-service"
    for module in (
        "train.train_baseline",
        "train.train_xgboost",
        "train.train_torch",
        # Needs two seasons of paired data; skips itself with a clear message otherwise.
        "train.train_preseason",
    ):
        if run([VENV_PY, "-m", module], ml, env, module, fatal=False) != 0:
            warn(f"{module} failed - continuing with the models that did train.")
    step("Model comparison - this is the accuracy on your data")
    run([VENV_PY, "-m", "train.evaluate"], ml, env, "evaluate", fatal=False)


# --------------------------------------------------------------------------- serving
def wait_for(url: str, seconds: int = 90) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def spawn_api(
    cwd: Path, port: int, env: dict[str, str], quiet: bool
) -> subprocess.Popen:
    cmd = [
        str(VENV_PY),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if quiet:
        cmd += ["--log-level", "warning"]
    return subprocess.Popen(cmd, cwd=str(cwd), env=env)


def generate_projections(env: dict[str, str], ml_port: int) -> None:
    """Batch-score every player for the upcoming week, so /rankings has data.

    Needs the ML service, so it is started just for this pass and stopped after.
    """
    step("Generating projections for the upcoming week")
    ml = spawn_api(ROOT / "ml-service", ml_port, env, quiet=True)
    try:
        if not wait_for(f"http://127.0.0.1:{ml_port}/health", 60):
            warn("ML service did not start; skipping batch projections.")
            return
        rc = run(
            [VENV_PY, "-m", "run_weekly", "--score-only"],
            ROOT / "pipeline",
            env,
            "Scoring",
            fatal=False,
        )
        if rc != 0:
            warn("Batch projections failed. On-demand predictions will still work.")
    finally:
        ml.terminate()
        try:
            ml.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ml.kill()


def serve(env: dict[str, str], api_port: int, ml_port: int, open_browser: bool) -> int:
    step("Starting services")
    ml = spawn_api(ROOT / "ml-service", ml_port, env, quiet=True)
    api = spawn_api(ROOT / "backend", api_port, env, quiet=False)
    procs = [ml, api]
    try:
        if not wait_for(f"http://127.0.0.1:{api_port}/health"):
            die("API failed to start - see the output above.")
        ok("API healthy.")
        try:
            import json

            with urllib.request.urlopen(
                f"http://127.0.0.1:{api_port}/info", timeout=5
            ) as r:
                info = json.loads(r.read())
            ok(
                f"database={info['database']}  cache={info['cache_backend']}  "
                f"model={info['active_model_version']}"
            )
        except Exception:
            pass

        bar = "=" * 50
        print(f"\n{C.GREEN}{bar}\n FantasyIQ is running (no Docker)\n{bar}{C.OFF}\n")
        print(f"  Dashboard   http://localhost:{api_port}/app/")
        print(f"  API docs    http://localhost:{api_port}/docs")
        print(f"  ML docs     http://localhost:{ml_port}/docs")
        print(f"  Metrics     http://localhost:{api_port}/metrics")
        print(f"\n{C.DIM}  Press Ctrl+C to stop.{C.OFF}\n")

        if open_browser:
            import webbrowser

            webbrowser.open(f"http://localhost:{api_port}/app/")

        while True:
            time.sleep(1)
            if api.poll() is not None:
                warn("API process exited.")
                return api.returncode or 1
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Stopping...{C.OFF}")
        return 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python local.py",
        description="Run FantasyIQ locally with Python only (no Docker).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python local.py --demo         synthetic data, no network needed\n"
            "  python local.py                real NFL data\n"
            "  python local.py --data-urls    list files to download by hand\n"
            "  python local.py --offline      use files already in data/manual/\n"
            "  python local.py --serve-only   skip setup, just start the servers\n"
            "  python local.py --reset        delete the database and models\n"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Seed a synthetic season instead of downloading real data",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Read parquet files from data/manual/ instead of downloading",
    )
    parser.add_argument(
        "--data-urls",
        action="store_true",
        help="Print the files to download by hand, then exit",
    )
    parser.add_argument("--seasons", default="2024,2025", help="Seasons to ingest")
    parser.add_argument(
        "--skip-train", action="store_true", help="Load data but do not train"
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Skip setup, data and training; just start the servers",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the database, models and cached raw data, then exit",
    )
    parser.add_argument(
        "--reinstall", action="store_true", help="Rebuild the virtualenv"
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Do not install PyTorch (~200MB). Ridge and XGBoost still train.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser"
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ml-port", type=int, default=9000)
    args = parser.parse_args(argv)

    if sys.version_info < (3, 9):
        die(
            f"Python 3.9+ required, found {platform.python_version()}.\n"
            "Install from https://www.python.org/downloads/ (no admin rights needed)."
        )

    if args.reset:
        step("Resetting local state")
        DB_FILE.unlink(missing_ok=True)
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        shutil.rmtree(PIPELINE_DATA, ignore_errors=True)
        ok("Database, models and cached raw data removed.")
        return 0

    print(
        f"{C.DIM}FantasyIQ local runner - Python {platform.python_version()} "
        f"on {platform.system()}{C.OFF}"
    )

    for d in (MODEL_DIR, MANUAL_DIR, PIPELINE_DATA):
        d.mkdir(parents=True, exist_ok=True)

    env = child_env(args.ml_port)
    env["INGEST_SEASONS"] = args.seasons

    ensure_venv(args.reinstall)
    if not args.serve_only:
        ensure_deps(env, skip_torch=args.skip_torch)

    if args.data_urls:
        run(
            [VENV_PY, "-m", "ingest", "--urls"], ROOT / "pipeline", env, "ingest --urls"
        )
        print(f"\n{C.CYAN}Save those files into: {MANUAL_DIR}{C.OFF}\n")
        return 0

    if not args.serve_only:
        migrate(env)
        if args.demo:
            load_demo(env)
        else:
            load_real(env, "manual" if args.offline else "auto")
        build_context(env, demo=args.demo)
        if not args.skip_train:
            train(env)
        generate_projections(env, args.ml_port)

    return serve(env, args.api_port, args.ml_port, not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
