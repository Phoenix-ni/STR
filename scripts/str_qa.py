#!/usr/bin/env python
import sys

from str_core.__main__ import main

if __name__ == "__main__":
    sys.argv.insert(1, "qa")
    main()
