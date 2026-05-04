"""Allow ``python -m astraea`` to invoke the CLI."""

from astraea.cli import main

raise SystemExit(main())
