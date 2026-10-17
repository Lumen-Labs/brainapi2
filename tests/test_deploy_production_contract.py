from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from deploy.backup_restore import verify_backup
from scripts.check_release_readiness import _check_latency, evaluate
from scripts.openai_ci_stub import _embedding
from scripts.production_smoke import _latency_summary
from scripts.release_version import parse_release_tag, requires_product_claim_gates


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


@pytest.mark.parametrize("profile", ["light", "heavy"])
def test_only_nginx_publishes_ports(profile):
    data = yaml.safe_load(
        (DEPLOY / f"docker-compose.{profile}.yaml").read_text(encoding="utf-8")
    )
    published = {
        service: config["ports"]
        for service, config in data["services"].items()
        if config.get("ports")
    }
    assert published == {"nginx": ["80:80", "443:443"]}
    assert "version" not in data


def _compose_env(profile: str) -> dict[str, str]:
    env = {
        **os.environ,
        "BRAINAPI_IMAGE": "brainapi:test",
        "BRAINAPI_ENV_FILE": str(DEPLOY / f"env.{profile}.example"),
        "REDIS_PASSWORD": "redis-secret",
    }
    if profile == "light":
        env["POSTGRES_PASSWORD"] = "postgres-secret"
    else:
        env.update(
            {
                "NEO4J_PASSWORD": "neo-secret",
                "MONGO_PASSWORD": "mongo-secret",
                "MINIO_ACCESS_KEY": "minio-access",
                "MINIO_SECRET_KEY": "minio-secret",
                "MILVUS_TOKEN": "root:milvus-secret",
                "MILVUS_ROOT_PASSWORD": "milvus-secret",
            }
        )
    return env


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker unavailable")
@pytest.mark.parametrize("profile", ["light", "heavy"])
def test_compose_profiles_render_with_escaped_healthchecks(profile):
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(DEPLOY / f"docker-compose.{profile}.yaml"),
            "config",
            "--format",
            "json",
        ],
        env=_compose_env(profile),
        text=True,
        capture_output=True,
        check=True,
    )
    config = json.loads(result.stdout)
    assert config["services"]["redis"]["healthcheck"]["test"][1].find(
        "REDIS_PASSWORD"
    ) >= 0
    assert set(config["services"]["nginx"]["ports"][0]) >= {"target", "published"}


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker unavailable")
def test_compose_rejects_missing_required_secret():
    env = _compose_env("light")
    env.pop("REDIS_PASSWORD")
    result = subprocess.run(
        ["docker", "compose", "-f", str(DEPLOY / "docker-compose.light.yaml"), "config", "-q"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "REDIS_PASSWORD is required" in result.stderr


def test_nginx_routes_console_api_and_mcp():
    config = (DEPLOY / "nginx" / "conf.d" / "brainapi.conf").read_text(
        encoding="utf-8"
    )
    assert "location /mcp" in config
    assert "location /" in config
    assert "brainapi_api" in config
    assert "brainapi_mcp" in config


def test_backup_manifest_verification_detects_tampering(tmp_path):
    artifact = tmp_path / "postgres.dump"
    artifact.write_bytes(b"consistent")
    manifest = {
        "format_version": 1,
        "profile": "light",
        "checksums": {
            artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verify_backup(tmp_path, "light")
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_backup(tmp_path, "light")


def test_light_backup_includes_per_brain_postgres_databases():
    source = (DEPLOY / "backup_restore.py").read_text(encoding="utf-8")
    assert "postgres-brains.tar.gz" in source
    assert "brain_*.dump" in source


def test_heavy_neo4j_dump_runs_as_root_with_admin_entrypoint():
    source = (DEPLOY / "backup_restore.py").read_text(encoding="utf-8")
    assert '"--user"' in source
    assert '"0"' in source
    assert '"--entrypoint"' in source
    assert '"neo4j-admin"' in source
    assert "/data/databases" in source
    assert "mindepth" in source
    assert "CREATE DATABASE" in source
    assert '("system", "neo4j")' not in source
    assert '["up", "-d", "--wait", "neo4j"]' in source
    assert '["stop", "neo4j"]' in source


def test_recommendation_openapi_surface_is_preview():
    from src.services.api.app import app

    schema = app.openapi()
    assert schema["paths"]["/retrieve/recommend"]["get"]["x-stability"] == "preview"
    assert schema["paths"]["/retrieve/recommend"]["post"]["x-stability"] == "preview"


def test_production_smoke_percentiles_are_deterministic():
    assert _latency_summary([1, 2, 3, 4, 5]) == {
        "samples": 5,
        "p50_ms": 3,
        "p95_ms": 4.8,
        "p99_ms": 4.96,
    }


def test_product_state_artifact_never_persists_a_brain_token():
    source = (ROOT / "scripts" / "production_smoke.py").read_text(encoding="utf-8")
    state_block = source.split('state = {', 1)[1].split('}', 1)[0]
    assert '"brain_token"' not in state_block


def test_release_latency_gate_requires_both_profiles():
    result = {
        "context": {
            "p50_ms": 999,
            "p95_ms": 1200,
            "p99_ms": 1500,
            "online_llm_retrieval_loops": 0,
        },
        "search": {
            "p50_ms": 199,
            "p95_ms": 250,
            "p99_ms": 300,
            "excludes_embed_query": True,
        },
    }
    _check_latency({"light": result, "heavy": result})
    with pytest.raises(RuntimeError, match="missing the heavy"):
        _check_latency({"light": result})


def test_ci_embedding_stub_is_deterministic_and_normalized():
    first = _embedding("same input", dimensions=8)
    assert first == _embedding("same input", dimensions=8)
    assert first != _embedding("different input", dimensions=8)
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_release_workflow_enforces_artifact_gate_before_publish():
    workflow = (ROOT / ".github" / "workflows" / "release.yaml").read_text(
        encoding="utf-8"
    )
    assert "check_release_readiness.py gate-artifacts" in workflow
    assert '--tag "$GITHUB_REF_NAME"' in workflow
    assert "needs: [validate-tag, verify-required-checks]" in workflow
    assert 'python scripts/release_version.py "$GITHUB_REF_NAME"' in workflow
    assert "steps.tag.outputs.project_version" in workflow


def _gate_artifacts(tmp_path: Path, *, missing_flow: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    flows = {
        "authentication": True,
        "brain_creation": True,
        "plain_ingestion": True,
        "structured_ingestion": True,
        "task_completion": True,
        "context_retrieval": not missing_flow,
        "search": True,
        "console": True,
        "mcp": True,
    }
    smoke = {
        profile: {"flows": flows, "unexpected_5xx": 0} for profile in ("light", "heavy")
    }
    counts = {
        "brains": 1,
        "nodes": 2,
        "edges": 1,
        "chunks": 3,
        "vectors": 3,
        "observations": 0,
    }
    restore = {
        "before": counts,
        "after": counts,
        "retrieval_spot_checks": [{"match": True} for _ in range(10)],
    }
    latency = {
        profile: {
            "context": {
                "p50_ms": 5000,
                "p95_ms": 5500,
                "p99_ms": 5600,
                "online_llm_retrieval_loops": 0,
            },
            "search": {
                "p50_ms": 12,
                "p95_ms": 20,
                "p99_ms": 25,
                "excludes_embed_query": True,
            },
        }
        for profile in ("light", "heavy")
    }
    (tmp_path / "brainapi.spdx.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "smoke.json").write_text(json.dumps(smoke), encoding="utf-8")
    (tmp_path / "latency.json").write_text(json.dumps(latency), encoding="utf-8")
    (tmp_path / "restore-light.json").write_text(json.dumps(restore), encoding="utf-8")
    (tmp_path / "restore-heavy.json").write_text(json.dumps(restore), encoding="utf-8")
    (tmp_path / "security.json").write_text(
        json.dumps(
            {
                "high": 0,
                "critical": 0,
                "image_digest": "sha256:" + ("a" * 64),
                "archive_sha256": "b" * 64,
                "sbom": "brainapi.spdx.json",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_dev_channel_skips_ledger_and_latency_claim_gates(tmp_path):
    artifacts = _gate_artifacts(tmp_path)
    assert parse_release_tag("v2.19.1-dev.0").channel == "dev"
    assert requires_product_claim_gates("v2.19.1-dev.0") is False
    evaluate(artifacts, tag="v2.19.1-dev.0")
    with pytest.raises(RuntimeError, match="LongMemEval"):
        evaluate(artifacts)
    with pytest.raises(RuntimeError, match="LongMemEval"):
        evaluate(artifacts, tag="v2.19.1-rc.1")
    with pytest.raises(RuntimeError, match="missing flows"):
        evaluate(_gate_artifacts(tmp_path / "broken", missing_flow=True), tag="v2.19.1-dev.0")


def test_mcp_has_only_one_lifespan_definition():
    source = (ROOT / "src" / "services" / "mcp" / "app.py").read_text(
        encoding="utf-8"
    )
    assert source.count("async def _lifespan") == 1


def test_ci_profiles_enable_dense_search_without_online_llm():
    override = yaml.safe_load((DEPLOY / "docker-compose.ci.yaml").read_text())
    environment = override["x-ci-model-environment"]
    assert environment["SEARCH_ENABLED"] == "true"
    assert environment["SEARCH_USE_DENSE"] == "true"
    assert environment["SEARCH_USE_BM25"] == "false"
    assert environment["CONTEXT_PASSAGE_MODE"] == "dense"
