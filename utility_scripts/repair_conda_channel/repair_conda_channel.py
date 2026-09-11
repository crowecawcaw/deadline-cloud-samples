#!/usr/bin/env python3
"""
Check and repair a Conda channel that AWS Deadline Cloud workers cannot read.

Recent versions of ``rattler-index`` (0.31 and later) and ``rattler-build publish``
(0.75 and later) write a channel index that the Conda queue environment on a
service-managed fleet cannot parse. Jobs fail while the queue environment starts,
with this in the session log:

    openjd_fail: failed to load repodata

Two artifacts in the index cause it:

* ``info.repodata_revisions`` inside ``repodata.json``, written as a mapping where
  older readers expect a sequence.
* ``repodata_shards.msgpack.zst``, the sharded index, whose embedded channel info
  carries the same field.

This script removes both from every subdirectory of a channel, leaving a plain
``repodata.json`` that every Conda client reads. Packages, and the ``shards/``
objects themselves, are left alone. Run it after every publish or index operation
until the fleet reads the newer format.

Examples:
  # Report what needs repair, change nothing
  python repair_conda_channel.py --check s3://my-bucket/Conda/Default

  # Repair a channel in S3, and a channel in a local directory
  python repair_conda_channel.py s3://my-bucket/Conda/Default
  python repair_conda_channel.py ./temp-local-channel
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

# The compressed companions of repodata.json. A reader prefers these over the
# plain file, so a stale copy is worse than a missing one: it is served in place
# of the repaired index.
COMPANION_SUFFIXES = (".zst", ".bz2")

SHARD_INDEX_NAME = "repodata_shards.msgpack.zst"

# The field that older readers reject. Removing it from the channel info block is
# what makes repodata.json parse again.
INCOMPATIBLE_INFO_FIELD = "repodata_revisions"


class ChannelError(Exception):
    """A channel could not be read, or is not a channel at all."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove index artifacts that stop AWS Deadline Cloud service-managed "
            "fleets from reading a Conda channel."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check s3://my-bucket/Conda/Default
  %(prog)s s3://my-bucket/Conda/Default
  %(prog)s ./temp-local-channel
        """,
    )
    parser.add_argument(
        "channel",
        help=(
            "The channel to inspect. Either an S3 URL such as "
            "s3://my-bucket/Conda/Default, or a local directory."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report what would change and exit 1 if any repair is needed. "
            "Writes nothing. Use this in a build pipeline."
        ),
    )
    parser.add_argument(
        "--subdir",
        action="append",
        dest="subdirs",
        metavar="SUBDIR",
        help=(
            "Limit the operation to this subdirectory, such as linux-64. "
            "Repeatable. The default is every subdirectory that has a repodata.json."
        ),
    )
    parser.add_argument(
        "--profile",
        help="The AWS named profile to use for an S3 channel.",
    )
    parser.add_argument(
        "--region",
        help="The AWS Region of the bucket for an S3 channel.",
    )
    return parser.parse_args()


class ChannelStorage:
    """Read and write the index files of a channel, wherever it is stored."""

    def subdirs(self) -> List[str]:
        """Return the subdirectories that hold a repodata.json, sorted."""
        raise NotImplementedError

    def read(self, key: str) -> Optional[bytes]:
        """Return the contents of a file, or None when it does not exist."""
        raise NotImplementedError

    def write(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete a file. Deleting a file that is already gone is not an error."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def describe(self, key: str) -> str:
        """Return a location to show the operator."""
        raise NotImplementedError


class LocalChannelStorage(ChannelStorage):
    def __init__(self, root: Path):
        if not root.is_dir():
            raise ChannelError(f"{root} is not a directory")
        self.root = root

    def subdirs(self) -> List[str]:
        return sorted(
            path.parent.name
            for path in self.root.glob("*/repodata.json")
            if path.is_file()
        )

    def _path(self, key: str) -> Path:
        return self.root / key

    def read(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def write(self, key: str, data: bytes) -> None:
        self._path(key).write_bytes(data)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def describe(self, key: str) -> str:
        return str(self._path(key))


class S3ChannelStorage(ChannelStorage):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        profile: Optional[str] = None,
        region: Optional[str] = None,
    ):
        try:
            import boto3
        except ImportError:
            raise ChannelError(
                "The boto3 library is required for an S3 channel. "
                "Install it with: pip install boto3"
            )
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def subdirs(self) -> List[str]:
        found = set()
        paginator = self.client.get_paginator("list_objects_v2")
        base = f"{self.prefix}/" if self.prefix else ""
        for page in paginator.paginate(Bucket=self.bucket, Prefix=base):
            for item in page.get("Contents", []):
                relative = item["Key"][len(base) :]
                # Only a channel subdirectory holds "<subdir>/repodata.json".
                parts = relative.split("/")
                if len(parts) == 2 and parts[1] == "repodata.json":
                    found.add(parts[0])
        return sorted(found)

    def read(self, key: str) -> Optional[bytes]:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                return None
            raise
        return response["Body"].read()

    def write(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "NotFound", "404"):
                return False
            raise
        return True

    def describe(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"


def open_channel(
    channel: str, profile: Optional[str], region: Optional[str]
) -> ChannelStorage:
    if channel.startswith("s3://"):
        location = channel[len("s3://") :]
        bucket, _, prefix = location.partition("/")
        if not bucket:
            raise ChannelError(f"{channel} does not name a bucket")
        return S3ChannelStorage(bucket, prefix, profile=profile, region=region)
    return LocalChannelStorage(Path(channel))


def compress_zst(data: bytes) -> Optional[bytes]:
    """Compress with Zstandard, or return None when no implementation is available."""
    try:
        from compression import zstd  # type: ignore[import-not-found]

        return zstd.compress(data)
    except ImportError:
        pass
    try:
        import zstandard  # type: ignore[import-not-found]

        return zstandard.ZstdCompressor().compress(data)
    except ImportError:
        return None


def plan_subdir(storage: ChannelStorage, subdir: str) -> Tuple[List[str], Dict]:
    """Describe the repairs a subdirectory needs.

    Returns the list of human-readable actions, and the parsed repodata with the
    incompatible field removed. An empty action list means the subdirectory is
    already readable.
    """
    raw = storage.read(f"{subdir}/repodata.json")
    if raw is None:
        raise ChannelError(f"{storage.describe(f'{subdir}/repodata.json')} is missing")
    try:
        repodata = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ChannelError(
            f"{storage.describe(f'{subdir}/repodata.json')} is not valid JSON: {error}"
        )

    actions: List[str] = []
    info = repodata.get("info")
    if isinstance(info, dict) and INCOMPATIBLE_INFO_FIELD in info:
        del info[INCOMPATIBLE_INFO_FIELD]
        actions.append(
            f"remove info.{INCOMPATIBLE_INFO_FIELD} from {subdir}/repodata.json"
        )
        for suffix in COMPANION_SUFFIXES:
            companion = f"{subdir}/repodata.json{suffix}"
            if storage.exists(companion):
                actions.append(f"replace {companion}")
    if storage.exists(f"{subdir}/{SHARD_INDEX_NAME}"):
        actions.append(f"delete {subdir}/{SHARD_INDEX_NAME}")
    return actions, repodata


def repair_subdir(storage: ChannelStorage, subdir: str, repodata: Dict) -> List[str]:
    """Apply the repairs planned for a subdirectory. Returns what was done."""
    done: List[str] = []
    body = json.dumps(repodata).encode("utf-8")
    storage.write(f"{subdir}/repodata.json", body)
    done.append(f"wrote {subdir}/repodata.json")

    bz2_key = f"{subdir}/repodata.json.bz2"
    if storage.exists(bz2_key):
        storage.write(bz2_key, bz2.compress(body))
        done.append(f"wrote {bz2_key}")

    zst_key = f"{subdir}/repodata.json.zst"
    if storage.exists(zst_key):
        compressed = compress_zst(body)
        if compressed is None:
            # A reader prefers the compressed copy, so a stale one keeps the
            # channel broken. Without a Zstandard implementation the safe repair
            # is to remove it and let readers fall back to repodata.json.
            storage.delete(zst_key)
            done.append(f"deleted {zst_key} (no Zstandard support in this Python)")
        else:
            storage.write(zst_key, compressed)
            done.append(f"wrote {zst_key}")

    shard_index = f"{subdir}/{SHARD_INDEX_NAME}"
    if storage.exists(shard_index):
        storage.delete(shard_index)
        done.append(f"deleted {shard_index}")
    return done


def resolve_subdirs(storage: ChannelStorage, requested: Optional[Iterable[str]]) -> List[str]:
    if requested:
        return list(requested)
    subdirs = storage.subdirs()
    if not subdirs:
        raise ChannelError(
            "no subdirectory with a repodata.json was found. Check the channel "
            "location, and that the channel has been indexed."
        )
    return subdirs


def main() -> int:
    args = parse_arguments()
    try:
        storage = open_channel(args.channel, args.profile, args.region)
        subdirs = resolve_subdirs(storage, args.subdirs)
    except ChannelError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Channel: {args.channel}")
    print(f"Subdirectories: {', '.join(subdirs)}")
    print()

    needs_repair: List[Tuple[str, List[str], Dict]] = []
    for subdir in subdirs:
        try:
            actions, repodata = plan_subdir(storage, subdir)
        except ChannelError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        if actions:
            needs_repair.append((subdir, actions, repodata))
            for action in actions:
                print(f"  {subdir}: {action}")
        else:
            print(f"  {subdir}: already readable by service-managed fleets")

    if not needs_repair:
        print()
        print("Nothing to repair.")
        return 0

    if args.check:
        print()
        print(
            f"{len(needs_repair)} subdirectory or subdirectories need repair. "
            "Run without --check to apply."
        )
        return 1

    print()
    for subdir, _, repodata in needs_repair:
        try:
            for action in repair_subdir(storage, subdir, repodata):
                print(f"  {action}")
        except Exception as error:  # noqa: BLE001 - report the location that failed
            print(f"Error: repairing {subdir} failed: {error}", file=sys.stderr)
            return 2

    print()
    print("Repair complete. Submit a test job to confirm the queue environment starts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
