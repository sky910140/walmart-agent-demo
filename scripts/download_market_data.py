"""Download reproducible daily CSI 300 close and volume data plus source metadata."""

from finagent.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["download-market", *__import__("sys").argv[1:]]))
