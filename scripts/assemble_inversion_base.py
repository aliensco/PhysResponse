"""Reassemble and verify the Standard Core100 inversion base downloaded from Releases."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path


ARCHIVE = "phyresponse_eval_standard.tar.gz"
SHA256 = "1e91e255af96ebea4e63ef54f9ba7d684e424daf7e8b3842684f696f37c0a66e"


def assemble(directory):
    manifest = json.loads((directory / "phyresponse_eval_standard.parts.json").read_text("utf-8"))
    names = [f"{ARCHIVE}.part{i:03d}" for i in range(50)]
    if manifest["sha256"] != SHA256 or [part["name"] for part in manifest["parts"]] != names:
        raise ValueError("Unexpected manifest for the Standard v1 release")
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
        temporary = Path(output.name)
    try:
        with temporary.open("wb") as output:
            for part in manifest["parts"]:
                data = (directory / part["name"]).read_bytes()
                if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                    raise ValueError(f"Checksum mismatch: {part['name']}")
                output.write(data)
                digest.update(data)
        if digest.hexdigest() != SHA256 or temporary.stat().st_size != manifest["bytes"]:
            raise ValueError("Complete archive verification failed")
        temporary.replace(directory / ARCHIVE)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified {ARCHIVE}: {SHA256}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing all release assets")
    assemble(parser.parse_args().directory)
