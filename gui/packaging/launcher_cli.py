"""Console entry point of the packaged application: lhdetect3-cli <folder> [--scale ...]."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from lhdetect3_gui.cli import main
    sys.exit(main())
