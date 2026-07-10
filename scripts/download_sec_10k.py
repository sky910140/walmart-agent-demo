"""SEC-compatible command wrapper retained as the expected interview script entry point."""

from finagent.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["download-sec", *__import__("sys").argv[1:]]))
