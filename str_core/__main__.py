from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import OpenAICompatibleClient
from .pipeline import convert_path
from .qa import answer_triplet_questions


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m str_core", description="STR conversion and TripletQL QA")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser("convert", help="Convert Excel/HTML/image input to STR triplets")
    p_convert.add_argument("input")
    p_convert.add_argument("-o", "--output")
    p_convert.add_argument("--input-type", choices=["excel", "html", "image"])
    p_convert.add_argument("--use-llm", choices=["auto", "always", "never"], default="auto")
    p_convert.add_argument("--table-id", default="")
    p_convert.add_argument("--sheet")
    p_convert.add_argument("--split-checkpoint")
    p_convert.add_argument("--merge-checkpoint")
    p_convert.add_argument("--ocr-words")
    p_convert.add_argument("--device", default="cuda")

    p_qa = sub.add_parser("qa", help="Ask questions over an STR triplet JSON file")
    p_qa.add_argument("--triplet", required=True)
    p_qa.add_argument("-q", "--question", action="append", required=True)
    p_qa.add_argument("--route-mode", choices=["auto", "full", "partial", "filter", "learnable"], default="auto")
    p_qa.add_argument("--qwen-url")

    p_serve = sub.add_parser("serve", help="Start the FastAPI service")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    if args.cmd == "convert":
        data = convert_path(
            args.input,
            input_type=args.input_type,
            use_llm=args.use_llm,
            llm_client=OpenAICompatibleClient.from_env(),
            table_id=args.table_id,
            sheet=args.sheet,
            split_checkpoint=args.split_checkpoint,
            merge_checkpoint=args.merge_checkpoint,
            ocr_words_path=args.ocr_words,
            device=args.device,
        )
        rendered = json.dumps(data, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    elif args.cmd == "qa":
        triplet = json.loads(Path(args.triplet).read_text(encoding="utf-8"))
        result = answer_triplet_questions(
            triplet,
            args.question,
            client=OpenAICompatibleClient.from_env(),
            route_mode=args.route_mode,
            qwen_url=args.qwen_url,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("str_core.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
