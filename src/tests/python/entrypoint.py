#!/usr/bin/env python3

import pytest
import sys
import os

if __name__ == "__main__":

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    print(f"current_file_path: {current_file_path}")

    tmp = pytest.main([
        current_file_path,
        "-v",
        "-s",
    ])

    print(f"tmp: {tmp}")

    #sys.exit(tmp)