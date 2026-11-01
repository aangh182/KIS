"""Allow `python -m kis ...` without installing the package."""

from kis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
