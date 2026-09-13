#!/usr/bin/env python3
"""Serve the pinball page locally and proxy WebRTC negotiation to a comma."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request


class Handler(SimpleHTTPRequestHandler):
  comma_url = ""

  def do_POST(self):
    if self.path != "/stream":
      self.send_error(404)
      return
    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    request = urllib.request.Request(f"{self.comma_url}/stream", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
      with urllib.request.urlopen(request, timeout=15) as response:
        result = response.read()
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)
    except Exception as error:
      self.send_error(502, str(error))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--comma", default="http://192.168.61.68:5001")
  parser.add_argument("--port", type=int, default=8000)
  args = parser.parse_args()
  Handler.comma_url = args.comma.rstrip("/")
  root = Path(__file__).parents[2] / "openpilot/system/webrtc/pinball"
  server = ThreadingHTTPServer(("127.0.0.1", args.port), lambda *a, **kw: Handler(*a, directory=root, **kw))
  print(f"Open http://127.0.0.1:{args.port}/ (proxying {Handler.comma_url})")
  server.serve_forever()


if __name__ == "__main__":
  main()
