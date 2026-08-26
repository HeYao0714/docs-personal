\---

name: npu-param-test-workflow

description: End-to-end workflow for designing and implementing NPU server startup parameter tests in sglang. 5-phase pipeline with subagent review at each phase, business scenario modeling, and code branch tracing. Self-contained. Use when asked to design or implement NPU server startup parameter tests.

\---



\# NPU Parameter Test Workflow



\## Design Principles



1\. \*\*Boundary first\*\* — state what the test does NOT cover. Only test APIs that exist today.

2\. \*\*Confirm premise\*\* — one sentence per parameter, user-facing language. User confirms before scaling.

3\. \*\*Code → branches → factors\*\* — never imagine scenarios first. Trace every `if/else` to a line number. Drop constants (fixtures), label construction guarantees (ask: "what change could break this?").

4\. \*\*Priority = CI executability\*\* — P0 runs in nightly CI; P2 needs extra hardware. Start minimal, backfill.

5\. \*\*Kernel ≠ API\*\* — code existing in repo ≠ callable interface. Verify the exact endpoint/function signature exists.

6\. \*\*Every arg needs a reason\*\* — comment why each non-obvious server flag is necessary.

7\. \*\*Reference closest test\*\* — justify differences from the baseline.

8\. \*\*No test > wrong test\*\* — if you can't articulate what a case proves and what it doesn't, cut it.



\---



\## Pre-Phase: Split \& Scan



\*\*Split\*\* the full parameter list by code coupling before any phase begins. 3-5 params per group. Criteria: same consumption module → together; shared validation method → together; unrelated modules → separate. Prioritize the tightest-coupled group first. \*\*User confirms the split.\*\*



\*\*Scan\*\* for existing GPU/NPU tests (`test/`) and docs (`docs/`, `docs\_new/`) for each parameter. Note gaps — a param with no tests and no docs has the highest risk of misunderstood behavior.



\---



\## Workflow



```

Phase 0: Premise Confirmation     → subagent verify → main judge → user confirm

Phase 1: Scenario Modeling        → subagent verify → main judge → user confirm

Phase 2: Branch Tracing + Factors → subagent verify → main judge → user confirm

Phase 3: Test Case Design         → subagent review → main judge → user confirm

Phase 4: Code Implementation      → subagent review → main fix    → user approve

```



\*\*Rules:\*\* Groups execute serially. One subagent per phase per group. Main agent judges all findings before acting — not every finding is adopted. Re-run subagent only for significant rewrites (wrong premise, missed major branch). User confirms before advancing.



\---



\## Phase 0: Premise Confirmation



One-sentence expected behavior per param in \*\*user language\*\* (not code paths).



\*\*Where to read:\*\* CLI args → `server\_args.py` (definition + argparse + `\_handle\_\*` validation methods). Env vars → `environ.py`. Then grep the param name across `python/sglang/srt/` for consumption sites. \*\*Beware:\*\* platform-specific methods like `\_handle\_npu\_backends()` or `\_handle\_cuda\_backends()` may silently override param defaults — the effective value may differ from the `ServerArgs` dataclass default.



\*\*Process:\*\* Main reads all three layers (definition, validation, consumption) → writes premises → subagent verifies premises against actual code behavior (check edge cases, identify any behavior the premise gets wrong) → main judges \& corrects → user confirms.



\---



\## Phase 1: Scenario Modeling



Answer \*\*Who / When / Why / What-if\*\* for each parameter. Judge each scenario: CODE-SUPPORTED / PARTIALLY SUPPORTED / CONTRADICTED. Only code-supported scenarios survive. \*\*Document scenarios\*\* in table form — Phase 3 references them by name.



\*\*Process:\*\* Main infers scenarios from code + docs → subagent validates (Does the code actually support this? What gaps? Scenarios we missed? NPU-relevant?) → main judges → user confirms.



\---



\## Phase 2: Branch Tracing + Factor Identification



Trace `definition → validation → consumption`. Label every branch with `file:line`. Classify each variable: \*\*test factor\*\* (value changes branch) / \*\*fixture\*\* (constant, hardcoded in launch args) / \*\*construction guarantee\*\* (code structure prevents regression — must pass: "what change could break this?").



\*\*Process:\*\* Main traces + classifies → subagent audits (missed branches? misclassified factors? incorrect claims? check ALL files that consume this param, not just obvious ones; verify each classification with code evidence) → main judges → user confirms.



\---



\## Phase 3: Test Case Design



Build `scenario × branch → case matrix`. Each case: parameter value, expected behavior, branch line, priority, reference test, "does NOT cover."



\*\*Priority:\*\* P0 = nightly CI, covers core branch. P1 = boundary/error paths. P2 = special env needed.



\*\*Process:\*\* Main builds minimal matrix (covers each branch once, then backfills) → subagent audits for missing/redundant/wrong cases, API existence, NPU feasibility, branch accuracy, "does not cover" declarations → main judges, cuts cases that can't articulate what they prove → user confirms.



Output: `| # | Scenario | Operation | Expected | Branch | Priority | Reference | Does NOT cover |`



\---



\## Phase 4: Code Implementation



Follow NPU test patterns. Reference: `test/registered/ascend/basic\_function/parameter/test\_npu\_log\_level.py` (HTTP + log verify), `test/registered/ascend/interface/test\_npu\_api.py` (SSE streaming parse), `test/registered/ascend/interface/test\_npu\_penalty.py` (concurrent pattern).



\*\*Checklist:\*\* `CustomTestCase`, `register\_npu\_ci(est\_time=N, suite="nightly-\*-npu-a3", nightly=True)`, `--attention-backend ascend` in every launch, `--mem-fraction-static 0.8` for concurrent, `requests.Session` per thread (`max\_workers ≤ 10`), `kill\_process\_tree` in `finally`, defensive assertions (verify data exists before asserting absence), every non-obvious arg commented.



\*\*Process:\*\* Main writes → subagent reviews for bugs, NPU compat, thread safety, false-pass risks → main judges (CRITICAL→fix, HIGH→fix, MEDIUM→selective, LOW→skip) → user approves.



\---



\## NPU-Specific



\- `--attention-backend ascend` is a fixture, not a test factor. Always include it.

\- Model: smallest available from `sglang.test.ascend.test\_ascend\_utils`.

\- CI: `register\_npu\_ci(est\_time=N, suite="nightly-\*-npu-a3", nightly=True)`. No PR CI.

\- Memory: use `--mem-fraction-static 0.8` for concurrent tests.

\- Ports: sequential launch/teardown on same port → `TIME\_WAIT`. Use different ports.

\- Platform-agnostic params (pure Python) are valid for NPU without NPU-specific assertions.

