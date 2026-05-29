#!/usr/bin/env python
import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start STR TripletQL API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("str_core.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
