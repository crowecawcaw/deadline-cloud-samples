# Run pip, reporting a glibc version of at least 2.35 to wheel tag selection.
#
# The Isaac Sim 5.0 and later wheels on https://pypi.nvidia.com are tagged manylinux_2_35_x86_64,
# but Amazon Linux 2023 (the OS of Deadline Cloud service-managed fleet workers) has glibc 2.34.
# Almost all of the Isaac Sim binaries only require symbols that glibc 2.34 provides, so this
# wrapper lets pip select the wheels anyway. patch_glibc_versions.py handles the few binaries that
# reference newer symbol versions. See the recipe README for how this was verified.
import os
import sys

_confstr = os.confstr


def _confstr_glibc235(name):
    value = _confstr(name)
    if name == "CS_GNU_LIBC_VERSION" and value and value.startswith("glibc "):
        major, minor = (int(v) for v in value.split()[1].split(".")[:2])
        if (major, minor) < (2, 35):
            return "glibc 2.35"
    return value


os.confstr = _confstr_glibc235

from pip._internal.cli.main import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
