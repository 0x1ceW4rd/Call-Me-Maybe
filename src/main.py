import argparse
import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from constrained_generator import ConstrainedGenerator
from grammar import load_vocab
from models import FunctionDefinition, TestCase, Result
from function_validator import validate_against_schema

def load_function_definitions(path: Path):
    with open(path) as f:
        data = json.load(f)
    return [FunctionDefinition(**item) for item in data]

def load_test_cases(path: Path):
    with open(path) as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]

def extract_json(text: str) -> str:
    brace_count = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start != -1:
                return text[start:i+1]
    return ""

def write_results(output_path: Path, results: list):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=Path,
                            default=Path("data/input/functions_definition.json"))
        parser.add_argument("--input", type=Path,
                            default=Path("data/input/function_calling_tests.json"))
        parser.add_argument("--output", type=Path,
                            default=Path("data/output/function_calling_results.json"))
        args = parser.parse_args()

        functions = load_function_definitions(args.functions_definition)
        tests = load_test_cases(args.input)

        model = Small_LLM_Model()
        vocab = load_vocab()
        function_names = [f.name for f in functions]
        generator = ConstrainedGenerator(model, function_names, vocab)

        # Build a compact description of functions for the prompt
        func_desc = []
        for f in functions:
            params = ", ".join(f.parameters.keys())
            func_desc.append(f"{f.name}({params})")
        func_list_str = ", ".join(func_desc)

        results = []
        for idx, test in enumerate(tests):
            print(f"Processing ({idx+1}/{len(tests)}): {test.prompt[:50]}...")

            # Clear, direct prompt asking only for JSON
            prompt = f"""You are a function caller. Output ONLY a valid JSON object with two keys: "name" (one of {func_list_str}) and "parameters" (an object with the required arguments). Do not include any other text.
    User request: {test.prompt}
    Assistant: """

            output = generator.generate(prompt, max_new_tokens=150)
            json_str = extract_json(output)

            if not json_str:
                # Retry with higher temperature
                print("  Retrying with temperature 1.0...")
                output2 = generator.generate(prompt, max_new_tokens=150, temperature=1.0)
                json_str = extract_json(output2)

            if json_str:
                try:
                    data = json.loads(json_str)
                    result = Result(prompt=test.prompt, name=data["name"], parameters=data["parameters"])
                    valid, err = validate_against_schema(result, functions)
                    if not valid:
                        print(f"  Schema error: {err}. Using fallback.")
                        result = Result(prompt=test.prompt, name=function_names[0], parameters={})
                except Exception as e:
                    print(f"  JSON parse error: {e}. Using fallback.")
                    result = Result(prompt=test.prompt, name=function_names[0], parameters={})
            else:
                print("  No JSON extracted. Using fallback.")
                result = Result(prompt=test.prompt, name=function_names[0], parameters={})

            results.append(result)
            write_results(args.output, results)
            print(f"  -> Saved {len(results)} result(s)")

        print(f"\n✅ Done. Final output: {args.output}")
    except KeyboardInterrupt:
        print("Program shuting down...")
        exit(0)

if __name__ == "__main__":
    main()