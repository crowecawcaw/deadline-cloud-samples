# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Behavior checks for ``utility_scripts/repair_conda_channel``.

The script exists because a channel indexed by ``rattler-index`` 0.31 or later, or
published by ``rattler-build`` 0.75 or later, carries an ``info.repodata_revisions``
field and a ``repodata_shards.msgpack.zst`` index that a service-managed fleet cannot
parse. Every job on the queue then fails as the Conda queue environment starts.

These checks run the script against channels built in a temporary directory, so they
need no network access, no AWS credentials, and no Conda tooling. The S3 code path is
not exercised here; it is the same planning logic behind a different storage class.
"""
from __future__ import annotations

import bz2
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

SCRIPT = (
    REPO_ROOT / "utility_scripts" / "repair_conda_channel" / "repair_conda_channel.py"
)

SUBDIRS = ("linux-64", "noarch")

# The shape rattler-index writes for the field. A mapping is what older readers
# reject, so the fixture has to use a mapping rather than a placeholder value.
REPODATA_REVISIONS = {
    "v0": {"n_packages": 1, "oldest": 1780415880114, "newest": 1788294726616}
}


def _repodata(subdir: str, with_revisions: bool) -> dict:
    info: dict = {"subdir": subdir}
    if with_revisions:
        info["repodata_revisions"] = REPODATA_REVISIONS
    return {
        "info": info,
        "packages": {},
        "packages.conda": {
            "example-1.0.0-h4616a5c_0.conda": {
                "build": "h4616a5c_0",
                "build_number": 0,
                "depends": [],
                "md5": "0" * 32,
                "name": "example",
                "sha256": "0" * 64,
                "size": 1874,
                "subdir": subdir,
                "version": "1.0.0",
            }
        },
        "repodata_version": 2,
    }


def _write_channel(
    root: Path,
    with_revisions: bool,
    with_shard_index: bool,
    with_companions: bool = True,
) -> Path:
    for subdir in SUBDIRS:
        directory = root / subdir
        directory.mkdir(parents=True)
        body = json.dumps(_repodata(subdir, with_revisions)).encode("utf-8")
        (directory / "repodata.json").write_bytes(body)
        if with_companions:
            # Stale bytes on purpose: a reader prefers a compressed companion, so
            # the repair has to replace or remove it rather than leave it behind.
            (directory / "repodata.json.zst").write_bytes(b"stale-zst")
            (directory / "repodata.json.bz2").write_bytes(bz2.compress(body))
        if with_shard_index:
            (directory / "repodata_shards.msgpack.zst").write_bytes(b"shard-index")
            shard = directory / "shards"
            shard.mkdir()
            (shard / f"{'a' * 64}.msgpack.zst").write_bytes(b"shard")
        (directory / "example-1.0.0-h4616a5c_0.conda").write_bytes(b"package")
    return root


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def broken_channel(tmp_path: Path) -> Path:
    return _write_channel(tmp_path / "broken", with_revisions=True, with_shard_index=True)


@pytest.fixture
def readable_channel(tmp_path: Path) -> Path:
    return _write_channel(
        tmp_path / "readable", with_revisions=False, with_shard_index=False
    )


def test_script_exists():
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"


def test_check_reports_and_fails_on_a_broken_channel(broken_channel: Path):
    """--check is a gate: it writes nothing and exits non-zero."""
    before = {
        path: path.read_bytes() for path in broken_channel.rglob("*") if path.is_file()
    }

    result = _run("--check", str(broken_channel))

    assert result.returncode == 1, result.stdout + result.stderr
    for subdir in SUBDIRS:
        assert f"remove info.repodata_revisions from {subdir}/repodata.json" in result.stdout
        assert f"delete {subdir}/repodata_shards.msgpack.zst" in result.stdout
    after = {
        path: path.read_bytes() for path in broken_channel.rglob("*") if path.is_file()
    }
    assert after == before, "--check modified the channel"


def test_repair_removes_the_incompatible_artifacts(broken_channel: Path):
    result = _run(str(broken_channel))

    assert result.returncode == 0, result.stdout + result.stderr
    for subdir in SUBDIRS:
        directory = broken_channel / subdir
        repodata = json.loads((directory / "repodata.json").read_text(encoding="utf-8"))
        assert "repodata_revisions" not in repodata["info"]
        assert repodata["info"]["subdir"] == subdir
        # The package list has to survive the repair.
        assert "example-1.0.0-h4616a5c_0.conda" in repodata["packages.conda"]
        assert not (directory / "repodata_shards.msgpack.zst").exists()
        # Packages and shard objects are left alone; nothing references the
        # latter once the index is gone.
        assert (directory / "example-1.0.0-h4616a5c_0.conda").is_file()
        assert (directory / "shards").is_dir()


def test_repair_replaces_or_removes_the_compressed_companions(broken_channel: Path):
    """A stale companion is served in place of the repaired index, so it cannot stay."""
    assert _run(str(broken_channel)).returncode == 0

    for subdir in SUBDIRS:
        directory = broken_channel / subdir
        expected = json.loads((directory / "repodata.json").read_text(encoding="utf-8"))

        bz2_path = directory / "repodata.json.bz2"
        assert bz2_path.is_file(), "the bz2 companion should be rewritten, not deleted"
        assert json.loads(bz2.decompress(bz2_path.read_bytes())) == expected

        zst_path = directory / "repodata.json.zst"
        if zst_path.exists():
            assert json.loads(_decompress_zst(zst_path.read_bytes())) == expected
        # When the running Python has no Zstandard support the script deletes the
        # companion instead, and readers fall back to repodata.json.


def _decompress_zst(data: bytes) -> bytes:
    try:
        from compression import zstd  # type: ignore[import-not-found]

        return zstd.decompress(data)
    except ImportError:
        pass
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        pytest.fail(
            "the script wrote a .zst companion but no Zstandard implementation is "
            "available to verify it"
        )
    return zstandard.ZstdDecompressor().decompress(data)


def test_repair_is_idempotent(broken_channel: Path):
    assert _run(str(broken_channel)).returncode == 0
    after_first = {
        path.relative_to(broken_channel): path.read_bytes()
        for path in broken_channel.rglob("*")
        if path.is_file()
    }

    second = _run(str(broken_channel))

    assert second.returncode == 0, second.stdout + second.stderr
    assert "Nothing to repair" in second.stdout
    after_second = {
        path.relative_to(broken_channel): path.read_bytes()
        for path in broken_channel.rglob("*")
        if path.is_file()
    }
    assert after_second == after_first
    assert _run("--check", str(broken_channel)).returncode == 0


def test_readable_channel_is_left_alone(readable_channel: Path):
    before = {
        path: path.read_bytes() for path in readable_channel.rglob("*") if path.is_file()
    }

    result = _run(str(readable_channel))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Nothing to repair" in result.stdout
    after = {
        path: path.read_bytes() for path in readable_channel.rglob("*") if path.is_file()
    }
    assert after == before


def test_subdir_option_limits_the_scope(broken_channel: Path):
    result = _run("--subdir", "linux-64", str(broken_channel))

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (broken_channel / "linux-64" / "repodata_shards.msgpack.zst").exists()
    assert (broken_channel / "noarch" / "repodata_shards.msgpack.zst").is_file()


def test_channel_without_repodata_is_an_error(tmp_path: Path):
    empty = tmp_path / "not-a-channel"
    empty.mkdir()

    result = _run(str(empty))

    assert result.returncode == 2
    assert "no subdirectory with a repodata.json" in result.stderr


def test_missing_channel_directory_is_an_error(tmp_path: Path):
    result = _run(str(tmp_path / "absent"))

    assert result.returncode == 2
    assert "is not a directory" in result.stderr


def test_unparsable_repodata_is_an_error(tmp_path: Path):
    channel = tmp_path / "corrupt"
    (channel / "noarch").mkdir(parents=True)
    (channel / "noarch" / "repodata.json").write_text("{not json", encoding="utf-8")

    result = _run(str(channel))

    assert result.returncode == 2
    assert "not valid JSON" in result.stderr
