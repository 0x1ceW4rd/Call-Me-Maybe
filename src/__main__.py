from llm_sdk import Small_LLM_Model
import json
from argparse import ArgumentParser, Namespace
from src.models import Functions, Prompts
from pydantic import ValidationError
from json import JSONDecodeError
from sys import exit, stderr
from src.vocab import VocabManager
import numpy as np
from numpy import intp
from pathlib import Path
from typing import Any, List, Dict
import os
import time

# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
DIM = "\033[2m"


def main() -> None:
    """
    Loads function definitions and test prompts, runs a small LLM to
    select and execute function calls, and writes structured JSON output.
    """
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args: Namespace = parser.parse_args()

    try:
        with open(args.functions_definition, "r") as f:
            fun_defs: Any = json.load(f)
            functions: List[Functions] = [Functions(**fd) for fd in fun_defs]

        with open(args.input, "r") as f:
            props: Any = json.load(f)
            proms: List[Prompts] = [Prompts(**p) for p in props]
    except ValidationError:
        print(f"{YELLOW}- Invalid input data schema{RESET}", file=stderr)
        exit(1)
    except JSONDecodeError:
        print(f"{YELLOW}- Invalid JSON format in input files{RESET}",
              file=stderr)
        exit(1)
    except FileNotFoundError:
        print(f"{YELLOW}- Input file not found{RESET}", file=stderr)
        exit(1)
    except PermissionError:
        print(f"{YELLOW}- Permission denied accessing input files{RESET}",
              file=stderr)
        exit(1)
    except Exception as e:
        print(f"{YELLOW}- Unexpected loading error: {e}{RESET}", file=stderr)
        exit(1)

    model: Small_LLM_Model = Small_LLM_Model(model_name=args.model)
    vocab: VocabManager = VocabManager(model, functions)
    fs: List[Functions] = functions
    names: List[str] = [f"name: {f.name} - description: {f.description}\n"
                        for f in fs]

    os.system("clear")

    print(
        f"\n{CYAN}┌────────────────────────────────────"
        "──────────────────────────────────────────────┐"
    )
    print(
        "│ ██████╗ ███████╗ ██████╗ ███╗   ██╗    ███████╗████████╗"
        " █████╗ ██████╗ ████████╗│"
    )
    print(
        "│ ╚══██║  ██╔════╝██╔═══██╗████╗  ██║    ██╔════╝"
        "╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝│"
    )
    print(
        "│    ██║  ███████╗██║   ██║██╔██╗ ██║   "
        " ███████╗   ██║   ███████║██████╔╝   ██║   │"
    )
    print(
        "│    ██║  ╚════██║██║   ██║██║╚██╗██║    ╚════██║  "
        " ██║   ██╔══██║██╔══██╗   ██║   │"
    )
    print(
        "│ █████║  ███████║╚██████╔╝██║ ╚████║  "
        "  ███████║   ██║   ██║  ██║██║  ██║   ██║   │"
    )
    print(
        "│ ╚════╝  ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝   "
        "╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   │"
    )
    print(
        "│                                      "
        "                                            │"
    )
    print(
        f"│ {RESET}{BOLD}                       "
        "[ CONSTRAINED DECODING ACTIVE ]                           {CYAN}│"
    )
    print(
        f"└─────────────────────────────────────────────"
        f"─────────────────────────────────────┘{RESET}"
    )

    res: List[Dict[str, Any]] = []

    for j, prompt in enumerate(proms, start=1):
        print(
            f"\n{BOLD}──────────────────────────────────"
            "───────────────────────────────{RESET}"
        )
        print(
            f"{MAGENTA}🚀 [{j}/{len(proms)}] Processing Prompt:"
            f"{RESET} {BOLD}'{prompt.prompt}'{RESET}\n"
        )

        n_prompt: str = f'choose a function name from the following functions\
            \n\n{"".join(names)}\nfor the following prompt \
            "{prompt.prompt}"\nchosen name: "'

        # Start displaying the streaming JSON output blocks
        print(f"{DIM}  ↳ Streaming Model Inference Target:{RESET}")
        js: str = '{\n    "prompt": "' + f'{prompt.prompt}",\n    "name": "'
        name: str = ""

        print(f"{DIM}{js}{RESET}{CYAN}", end="", flush=True)

        while True:
            ids = model.encode(n_prompt)[0].tolist()
            tok_id: intp = np.argmax(
                model.get_logits_from_input_ids(ids) + vocab.M_fun_name
            )
            tok: str = model.decode(tok_id)
            if '"' in tok:
                break
            n_prompt += tok
            name += tok
            print(tok, end="", flush=True)

        js += name + '",\n    ' + '"parameters": {'
        print(f'{RESET}{DIM}",\n    "parameters": {{{RESET}',
              end="", flush=True)

        chosen_fun = None
        for fun in functions:
            if fun.name == name:
                chosen_fun = fun

        if chosen_fun is None:
            print(
                f"\n{YELLOW}- Unknown chosen function target: {name}{RESET}",
                file=stderr,
            )
            exit(1)

        entry: Dict[str, Any] = {
            "prompt": prompt.prompt,
            "name": name,
            "parameters": {},
        }

        p_prompt: str = f"""Task: Complete the JSON function call.

Function: {chosen_fun.name}

RULES:
1. You must write the SHORTEST possible pattern.
2. For numbers, you must output EXACTLY [0-9]+ and immediately stop.
3. For vowels, you must output EXACTLY aeiouAEIOU and immediately stop.
4. DO NOT repeat patterns. ALWAYS close the string with a double quote (")!
5. ALWAYS escape double quotes in parameters with (\\)!
6. when generating a path, ALWAYS generate the full path not just the file name
8. NEVER include 'database' for database parameter only the name of database

--- EXAMPLES ---
Input: "Replace all vowels in 'this is a test' with asterisks"
JSON: {{"name": "fn_substitute_string_with_regex", "parameters": \
    {{"source_string": "this is a test", \
        "regex": "[aeiouAEIOU]", "replacement": "*"}}}}

Input: "Replace all numbers in 'Phone 555-1234' with NUMBERS"
JSON: {{"name": "fn_substitute_string_with_regex", "parameters": \
    {{"source_string": "Phone 555-1234", "regex": "[0-9]+", \
        "replacement": "NUMBERS"}}}}

Input: "Run the query 'INSERT INTO logs VALUES (1, 2, 3)' \
    on the system database"
JSON: {{"name": "fn_execute_sql_query", "parameters":
    {{"query": "INSERT INTO logs VALUES (1, 2, 3)", "database": "system"}}}}

Input: "Format template: Say "hello" to {{name}}" \
JSON: {{"name": "fn_format_template", "parameters":
    {{"template": "Say \\"hello\\" to {{name}}"}}}}

Input: "{prompt.prompt}"
JSON:
{js}"""

        parama_len: int = len(chosen_fun.parameters)
        for i, (n, p) in enumerate(chosen_fun.parameters.items()):
            p_prompt += f'"{n}": '
            print(f'\n        {DIM}"{n}": {RESET}', end="", flush=True)

            if p.type == "string":
                p_prompt += '"'
                s_accum: str = ""
                print(f'{DIM}"{RESET}{CYAN}', end="", flush=True)
                while True:
                    ids = model.encode(p_prompt)[0].tolist()
                    s_tok_id: intp = np.argmax(
                        model.get_logits_from_input_ids(ids) + vocab.M_chars
                    )
                    s_tok: str = model.decode(s_tok_id)
                    if '"' in s_tok and '\\"' not in s_tok:
                        if '\\"' in (s_accum + s_tok):
                            continue
                        s_tok = s_tok.split('"', 1)[0] + '"'
                        s_accum += s_tok.split('"', 1)[0]
                        p_prompt += s_tok
                        print(s_tok, end="", flush=True)
                        break
                    p_prompt += s_tok
                    s_accum += s_tok
                    print(s_tok, end="", flush=True)
                    if '\\"' in s_accum:
                        s_accum = s_accum.replace("\\", "")

                print(RESET, end="", flush=True)
                if i < parama_len - 1:
                    p_prompt += ", "
                    print(f"{DIM},{RESET}", end="", flush=True)
                entry["parameters"][n] = s_accum

            if p.type == "number" or p.type == "integer":
                n_accum: str = ""
                print(CYAN, end="", flush=True)
                while True:
                    ids = model.encode(p_prompt)[0].tolist()
                    n_tok_id: intp = np.argmax(
                        model.get_logits_from_input_ids(ids) + vocab.M_numbers
                    )
                    n_tok = model.decode(n_tok_id)
                    if "," in n_tok:
                        break
                    p_prompt += n_tok
                    n_accum += n_tok
                    print(n_tok, end="", flush=True)

                print(RESET, end="", flush=True)
                if i < parama_len - 1:
                    p_prompt += ", "
                    print(f"{DIM},{RESET}", end="", flush=True)
                if p.type == "number":
                    entry["parameters"][n] = float(n_accum)
                else:
                    entry["parameters"][n] = int(n_accum)

        res.append(entry)
        print(f"\n    {DIM}}}{RESET}\n{DIM}}}{RESET}")
        print(f"{GREEN}✔ Step Completed Successfully{RESET}")

    # Output generation finish segment
    path: Path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w") as f:
            json.dump(res, f, indent=2)
        print(f"\n{GREEN}↳ Data successfully saved "
              f"to:{RESET} {BOLD}{path}{RESET}\n")
    except Exception as e:
        print(f"{YELLOW}- Error writing output file: {e}{RESET}", file=stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(
            f"\n{YELLOW}✕ An unhandled execution error occurred: {e}{RESET}",
            file=stderr,
        )
        exit(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠ Process termination signal captured...{RESET}")
        time.sleep(0.5)
        print(f"{YELLOW}The process has been cleanly terminated.{RESET}\n")
        exit(1)
