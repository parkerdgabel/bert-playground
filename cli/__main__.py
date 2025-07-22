"""Allow mlx-bert to be run as a module.

This enables running the CLI with `python -m cli` or `python -m mlx_bert`.
"""

from . import main

if __name__ == "__main__":
    main()
