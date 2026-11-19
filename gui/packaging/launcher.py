"""Entry point of the packaged application.

LHDetect3.exe                      window
LHDetect3.exe --selftest out.txt   load the model, process the reference image, write a report
lhdetect3-cli.exe <folder> ...     command-line processing (console program built from the same file)
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        from lhdetect3_gui.selftest import selftest
        sys.exit(selftest(sys.argv[2] if len(sys.argv) > 2 else None))
    from lhdetect3_gui.app import main
    sys.exit(main())
