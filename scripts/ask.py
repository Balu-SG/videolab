"""Talk to the lab partner from the terminal. Each argument is one message in the same conversation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent import LabAgent  # noqa: E402


def main() -> None:
    agent = LabAgent()
    for message in sys.argv[1:]:
        print(f"\n>>> {message}")
        r = agent.ask(message)
        for t in r["tools"]:
            print(f"  [tool] {t['tool']}({t['arguments']})")
        print(f"\n{r['answer']}\n")
        print(f"  grounding: {json.dumps(r['grounding'])}")
        if r["usage"]:
            print(f"  tokens: in {r['usage'].get('input_tokens')} out {r['usage'].get('output_tokens')}")


if __name__ == "__main__":
    main()
