import sys

if len(sys.argv) > 1 and sys.argv[1] == "--cli":
    from .cli import main as cli_main
    sys.exit(cli_main(sys.argv[2:]))
if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
    from .selftest import selftest
    sys.exit(selftest(sys.argv[2] if len(sys.argv) > 2 else None))

from .app import main

sys.exit(main())
