"""Retarget glibc 2.35+ symbol version requirements that have an older equivalent.

The Isaac Sim 5.0+ wheels are built on glibc 2.35, and a few of their binaries reference
``hypot@GLIBC_2.35`` or ``hypotf@GLIBC_2.35``. glibc 2.35 only added new versions of these
functions (faster implementations with the same behavior); the ``GLIBC_2.2.5`` versions are
still exported by libm on glibc 2.34 hosts such as Amazon Linux 2023. Without a fix, the dynamic
loader refuses to load those binaries, which includes Kit's embedded ``libpython`` and USD
libraries, and Kit fails or crashes during startup.

This script walks a directory tree and, for each ELF file, rewrites a version requirement
(``Vernaux`` entry) newer than ``--max-glibc`` to an older version listed in ``EQUIVALENTS``.
It only edits the 4-byte hash and 4-byte name offset of the entry, and the replacement name
must already exist in the file's ``.dynstr``.

A binary that needs a newer glibc symbol listed in ``TOLERATED`` is left unchanged with a
warning. The loader rejects that whole library, so this is only acceptable for libraries that
are loaded on demand by an optional feature. Any other newer glibc requirement, an unrecognized
``GLIBC_*`` version node, or an ELF file that can't be checked is reported as an error, so the
package build fails instead of producing a package that breaks at runtime. Only 64-bit
little-endian ELF files are checked.

Usage: python patch_glibc_versions.py <root> [--max-glibc 2.34]
"""

import argparse
import os
import re
import struct
import sys

from elftools.elf.elffile import ELFFile
from elftools.elf.gnuversions import GNUVerNeedSection

# symbol -> older glibc version that provides the same function
EQUIVALENTS = {
    "hypot": "GLIBC_2.2.5",
    "hypotf": "GLIBC_2.2.5",
}

# symbol -> why leaving binaries that need it unpatched is acceptable
TOLERATED = {
    # New in glibc 2.35 with no older equivalent; libgcc's C++ exception unwinder calls it.
    # In Isaac Sim 6.x only libomniverse_connection.so (omni.client's Nucleus protocol) needs
    # it. omni.client loads that library only for omniverse:// URLs, so local, HTTP(S), and S3
    # assets still work.
    "_dl_find_object": "omniverse:// (Nucleus) connections are unavailable on glibc 2.34",
}

WARNINGS = []


def elf_hash(name: bytes) -> int:
    h = 0
    for c in name:
        h = (h << 4) + c
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g
    return h & 0xFFFFFFFF


def glibc_tuple(name):
    m = re.fullmatch(r"GLIBC_(\d+)\.(\d+)(?:\.(\d+))?", name)
    return tuple(int(x) for x in m.groups() if x is not None) if m else None


def find_dynstr_offset(data: bytes, dynstr_offset: int, dynstr_size: int, name: str):
    table = data[dynstr_offset : dynstr_offset + dynstr_size]
    needle = name.encode() + b"\0"
    start = 0
    while True:
        i = table.find(needle, start)
        if i < 0:
            return None
        if i == 0 or table[i - 1] == 0:
            return i
        start = i + 1


def plan_patches(path, max_glibc):
    """Return (patches, problems). patches is a list of (file_offset, packed_bytes, description)."""
    patches, problems = [], []
    with open(path, "rb") as fh:
        data = fh.read()
    with open(path, "rb") as fh:
        elf = ELFFile(fh)
        if elf.elfclass != 64 or not elf.little_endian:
            return patches, problems
        verneed = next((s for s in elf.iter_sections() if isinstance(s, GNUVerNeedSection)), None)
        if verneed is None:
            return patches, problems
        dynstr = elf.get_section_by_name(".dynstr")
        dynsym = elf.get_section_by_name(".dynsym")
        versym = elf.get_section_by_name(".gnu.version")
        if dynstr is None or dynsym is None or versym is None:
            problems.append(f"{path}: has symbol version requirements but no .dynstr/.dynsym/.gnu.version to check")
            return patches, problems

        # Walk the Verneed/Vernaux chain manually to know each entry's file offset.
        sec_off = verneed["sh_offset"]
        vn_off = sec_off
        for _ in range(verneed.num_versions()):
            vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, vn_off)
            aux_off = vn_off + vn_aux
            for _ in range(vn_cnt):
                vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, aux_off)
                name = dynstr.get_string(vna_name)
                ver = glibc_tuple(name)
                if ver is None and name.startswith("GLIBC_") and name != "GLIBC_PRIVATE":
                    # For example GLIBC_ABI_DT_RELR (glibc 2.36), which glibc 2.34 doesn't provide.
                    problems.append(f"{path}: needs unrecognized glibc version node {name}")
                elif ver is not None and ver > max_glibc:
                    symbols = sorted(
                        sym.name
                        for i, sym in enumerate(dynsym.iter_symbols())
                        if versym.get_symbol(i).entry["ndx"] == vna_other
                    )
                    targets = {EQUIVALENTS.get(s) for s in symbols}
                    if symbols and all(s in TOLERATED for s in symbols):
                        reasons = "; ".join(sorted({TOLERATED[s] for s in symbols}))
                        WARNINGS.append(f"{path}: left unpatched, needs {name} for {symbols} ({reasons})")
                    elif not symbols or None in targets or len(targets) != 1:
                        problems.append(f"{path}: {name} needed for {symbols} has no known older equivalent")
                    else:
                        target = targets.pop()
                        name_off = find_dynstr_offset(data, dynstr["sh_offset"], dynstr["sh_size"], target)
                        if name_off is None:
                            problems.append(f"{path}: {name} -> {target} but {target} is not in .dynstr")
                        else:
                            new = struct.pack("<I", elf_hash(target.encode()))
                            patches.append((aux_off, new, None))
                            patches.append((aux_off + 8, struct.pack("<I", name_off), f"{name} -> {target} for {symbols}"))
                if not vna_next:
                    break
                aux_off += vna_next
            if not vn_next:
                break
            vn_off += vn_next
    return patches, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--max-glibc", default="2.34")
    args = parser.parse_args()
    max_glibc = tuple(int(x) for x in args.max_glibc.split("."))

    all_problems = []
    patched = 0
    for dirpath, _, filenames in os.walk(args.root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            with open(path, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    continue
            try:
                patches, problems = plan_patches(path, max_glibc)
            except Exception as exc:
                # A file that can't be checked might need a newer glibc, so treat it as a problem.
                all_problems.append(f"{path}: could not check ELF file: {exc!r}")
                continue
            all_problems.extend(problems)
            if patches:
                mode = os.stat(path).st_mode
                os.chmod(path, mode | 0o200)
                with open(path, "r+b") as fh:
                    for offset, value, description in patches:
                        fh.seek(offset)
                        fh.write(value)
                        if description:
                            print(f"patched {os.path.relpath(path, args.root)}: {description}")
                os.chmod(path, mode)
                patched += 1

    print(f"Patched {patched} ELF files.")
    for w in WARNINGS:
        print(f"warning: {w}")
    if all_problems:
        print("ERROR: binaries require glibc versions newer than the target:")
        for p in all_problems:
            print(f"  {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
