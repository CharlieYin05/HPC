#!/usr/bin/env python3

import argparse
from pathlib import Path

UINT64_MASK = 0xffffffffffffffff


def toy_hash(data: bytes) -> int:
    h = 0xcbf29ce484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001b3) & UINT64_MASK
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & UINT64_MASK
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & UINT64_MASK
    h ^= h >> 33
    return h & 0x0000ffffffffffff


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the toy_hash of a file.")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    value = toy_hash(args.file.read_bytes())
    print(f"{value:012x}")


if __name__ == "__main__":
    main()
