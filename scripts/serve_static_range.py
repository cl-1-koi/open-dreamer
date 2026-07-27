#!/usr/bin/env python3
"""Serve a directory over localhost HTTP with single-range byte requests."""

from __future__ import annotations

import argparse
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeRequestHandler(SimpleHTTPRequestHandler):
    _range: tuple[int, int] | None = None

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            self._range = None
            return super().send_head()
        try:
            file_handle = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(file_handle.fileno()).st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not match:
            self._range = None
            file_handle.close()
            return super().send_head()

        start_text, end_text = match.groups()
        if not start_text and not end_text:
            file_handle.close()
            self.send_error(416, "Invalid range")
            return None
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            start = max(0, size - suffix)
            end = size - 1
        end = min(end, size - 1)
        if start < 0 or start > end or start >= size:
            file_handle.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self._range = (start, end)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        file_handle.seek(start)
        return file_handle

    def copyfile(self, source, outputfile):
        if self._range is None:
            return super().copyfile(source, outputfile)
        remaining = self._range[1] - self._range[0] + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", default=8099, type=int)
    args = parser.parse_args()

    handler = lambda *handler_args, **kwargs: RangeRequestHandler(  # noqa: E731
        *handler_args,
        directory=str(args.directory.resolve()),
        **kwargs,
    )
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(
        f"Serving {args.directory.resolve()} on "
        f"http://{args.bind}:{args.port}/",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
