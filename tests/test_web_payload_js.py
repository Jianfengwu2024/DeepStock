from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "bnhmm" / "web" / "static" / "app.js"


def test_percolation_payload_forces_fixed_propagation_lag() -> None:
    harness = r"""
const fs = require("node:fs");
const vm = require("node:vm");

const fields = [
  { name: "strategy", value: "percolation-branching", type: "radio", checked: true },
  { name: "maximum_lag_minutes", value: "3", type: "hidden" },
  { name: "maximum_lag_minutes", value: "30", type: "number" },
  { name: "topology_lookback_sessions", value: "60", type: "hidden" },
];
for (const field of fields) {
  field.disabled = false;
  field.closest = () => null;
}

const form = {
  elements: fields,
  querySelectorAll: selector => selector === "[name]" ? fields : [],
};
const document = {
  querySelector: selector => {
    if (selector === "#experiment-form") return form;
    if (selector === "[name=strategy]:checked") return fields[0];
    return null;
  },
};

let source = fs.readFileSync(process.argv[1], "utf8");
source = source.slice(0, source.indexOf("\ninitialize().catch"));
source += "\nglobalThis.formPayloadUnderTest = formPayload;";
const context = vm.createContext({ document, FormData, fetch, console, setTimeout, clearTimeout });
vm.runInContext(source, context);
process.stdout.write(JSON.stringify(context.formPayloadUnderTest()));
"""
    completed = subprocess.run(
        ["node", "-e", harness, str(APP_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["maximum_lag_minutes"] == 3
    assert payload["topology_lookback_sessions"] == 60

