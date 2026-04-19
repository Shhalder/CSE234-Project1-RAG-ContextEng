import argparse
import json
from pathlib import Path

# argument parser, input/output
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline RAG pipeline entrypoint")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")

    return parser.parse_args()

# load input file (list of question_id and questions), return Python list
def load_questions(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f) # parsed json content

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each input item must be a dictionary.")
        if "question_id" not in item or "question" not in item:
            raise ValueError("Each input item must contain question_id and question.")
    return data # return Python list


# build output rows for every input
def build_placeholder_output(questions: list[dict]) -> list[dict]:
    output = []
    for item in questions:
        output.append(
            {
                "question_id": item["question_id"],
                "answer": "Baseline not implemented yet.",
                "sources": [],
            }
        )
    return output # return Python list with answers

# convert Python list to JSON and write to output file
def write_output(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)



# main
def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    questions = load_questions(input_path)
    rows = build_placeholder_output(questions)
    write_output(output_path, rows)


if __name__ == "__main__":
    main()
