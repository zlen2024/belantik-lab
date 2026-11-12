---
name: extropic
description: Run Python functions on Extropic's compute cloud (cpu/l4/a100/h100 tiers, billed in credits) with the `extro-sim` Python SDK and CLI (import `extro_sim`) — @xtr.job, .remote(), .submit(), extro-sim run/jobs/logs/collect/balance. Use when the user wants to run code or GPU work on Extropic, submit or collect a cloud job, mentions extropic or extro-sim, @xtr.job, tiers cpu/l4/a100/h100, api.extropic.dev, or hits OutOfCredits / AtCapacity / JobFailed / JobPreempted / JobTimedOut errors.
---

# Extropic

`import extro_sim as xtr` plus an `extro-sim` CLI. Submit a Python function or a
Python file to a cloud sandbox, then collect its result, its output, and its cost.

## Install

```bash
pip install extro-sim
```

Needs Python 3.10+. On 3.9 or older pip reports only that no matching
distribution was found.

Do NOT `pip install extropic` — that PyPI name is a different, unrelated
project. The distribution is `extro-sim`: the `extro_sim` import package and
the `extro-sim` command.

## Authenticate

Run `extro-sim auth`. It opens a browser, then stores a session under
`~/.config/extropic/credentials.json` and refreshes it automatically; every later
SDK and CLI call uses it. `extro-sim auth status` shows who is logged in,
`extro-sim auth logout` forgets the session.

From an interactive terminal, a first call with no session opens that same browser
flow by itself. Non-interactive runs (CI, pipes) instead fail with
`ExtropicError: 401: ...` — run `extro-sim auth` first, or set `EXTROPIC_TOKEN`.

## Two ways to run code

**SDK — ship a function.** The function and its arguments travel by cloudpickle,
so closures, objects, and non-JSON arguments and return values all work.

```python
import extro_sim as xtr

@xtr.job(tier="cpu")                  # must be CALLED; bare @xtr.job raises TypeError
def fold(seq, steps=1000):
    import random                      # imports must exist in the sandbox image
    ...
    return {"best": best}

fold(seq)                # runs locally, like any function (same as fold.local(seq))
fold.remote(seq)         # runs on Extropic, blocks, returns the result
job = fold.submit(seq)   # returns a Job handle immediately
```

Fan out with many `.submit()`, then `.result()` each.

**CLI — ship a file.** `extro-sim run job.py --entrypoint main --tier cpu` uploads
the file's *source*; the sandbox executes it and calls `entrypoint(*args, **kwargs)`
there. Nothing runs locally. Such a file must be plain Python: no
`import extro_sim`, no `@xtr.job` — the client package is not installed in the
sandbox. Pass arguments with repeatable `--arg VALUE` and `--kwarg key=value`
(each value is parsed as JSON, falling back to a string). A `--entrypoint` name
the file does not define is normally caught locally, before any credits are held.

## Job handle

```python
job.id                      # str
job.status                  # live status (job.state is the same value)
job.refresh()               # -> the current job view dict
job.wait(timeout=None)      # block to terminal, return the final view
job.result(timeout=None)    # block, return the value or raise a typed error
job.logs("stdout")          # "stdout" | "stderr", what has synced so far
job.stream_logs(stream=...) # yield one stream's text live
job.follow()                # yield ("stdout"|"stderr", text), then ("result", value)
                            # — on a FAILED job it raises instead of yielding a result
job.progress()              # structured progress from the LAST fetched view, or None
                            # — call job.refresh() first, every time, to advance it
job.cancel()
job.start_again()           # retry a start that bounced on capacity
```

`Client()` exposes the same surface by job id: `create_job`, `start_job`, `get`,
`cancel`, `list`, `iter_jobs`, `download_artifact`, `billing`, `topup`.

## Tiers

A tier is a GPU type, exactly one GPU per job. These four names are the entire
vocabulary; nothing else parses.

| tier | resources |
|---|---|
| `cpu` | 1 vCPU · 4 GB, no GPU |
| `l4` | 4 vCPU · 16 GB · 1×L4 24GB |
| `a100` | 8 vCPU · 64 GB · 1×A100 80GB |
| `h100` | 8 vCPU · 64 GB · 1×H100 80GB |

Billed per second of run time. `cpu` costs a small fraction of any GPU tier, and
cost rises `cpu` → `l4` → `a100` → `h100`. Default to `cpu`; move up only when the
work actually needs the GPU or the memory.

## Sandbox image

Debian slim, running your local Python minor version, with a matching
`cloudpickle`, `git`, and `thrml`. GPU tiers additionally ship a CUDA-enabled
`jax` — do not reinstall it.

There is no per-job dependency install: a job cannot add packages, and neither the
SDK nor the CLI takes a package list. A function that imports something the image
lacks fails with `JobFailed: ModuleNotFoundError`. Nothing else is shipped either
— the sandbox cannot see local files, so pass data in as arguments and return data
as the result.

## Output and progress

`print()` output IS retrievable. The sandbox captures stdout and stderr, syncs them
to durable storage while the job runs, and flushes both (plus the result, or the
traceback on failure) before it exits — so the tail survives even a sandbox that
dies.

- `job.logs()` / `job.logs("stderr")` — everything synced so far; `""` when nothing
  has synced yet, complete once the job is terminal.
- `job.stream_logs()` — live text from the sandbox while it is alive, then a final
  read from storage. The byte cursor only moves forward; output is never replayed.
- `job.follow()` — both streams plus the result, read entirely from storage; works
  the same for a running and a finished job.
- CLI: `extro-sim logs <id>` prints a stream, `-f` follows it live, `--stderr`
  selects that stream. `extro-sim run` follows both while the job runs.

Reads from storage lag live output by the sync cadence — seconds, and longer for
large outputs.

For structured progress, call the builtin `extropic_report(**fields)` inside the
job and read it back with `job.progress()` while the job is running. The builtin
exists only in the sandbox, so wrap the call in `try: ... except NameError: pass`
if the same code also runs locally.

Two traps:

- `job.progress()` reads the job view the handle last fetched — it does not poll.
  A `.submit()` handle starts with the start response, which carries no runner, so
  a bare `while True: job.progress()` loop returns `None` forever. Call
  `job.refresh()` at the top of each iteration.
- The merged progress object is capped at 16 KiB, and exceeding it **raises
  `ValueError` inside your job** rather than dropping the update — so an
  accumulating report (`extropic_report(losses=every_loss_so_far)`) eventually
  kills the run. `except NameError` does not catch it. Report deltas and
  summaries, not growing collections.

## States

In flight: `uploading` · `provisioning` · `running`.

Terminal: `succeeded` · `failed` · `timeout` · `cancelled` · `out_of_credits` ·
`preempted` · `expired` · `error`.

`at_capacity` is not a status. Capacity shows up as an `AtCapacity` exception at
submit or start time; the job stays in `uploading` with its uploaded input intact.

## Errors

`.result()` and `.remote()` raise subclasses of `ExtropicError` — except the
local-wait timeout: `.result(timeout=)` and `.wait(timeout=)` raise the BUILTIN
`TimeoutError`, which is NOT an `ExtropicError`. `except ExtropicError` alone will
not catch it.

| exception | meaning | do |
|---|---|---|
| `JobFailed` | the job's own code raised. `.traceback` holds the sandbox traceback (also appended to the message). `.stdout` / `.stderr` exist but are always `None` — get the output with `job.logs()` / `extro-sim logs <id>` | read the traceback, fix the function, resubmit |
| `JobTimedOut` | the job hit the `timeout_s` you set and was stopped | raise `timeout_s` or shrink the work |
| `OutOfCredits` | the account cannot cover the job (402 at submit, or terminal status `out_of_credits`) | stop submitting; `extro-sim recharge <usd>`, then finish the card payment in the console |
| `AtCapacity` | no capacity (503). After a start bounce `.job` carries the handle | `err.job.start_again()` or `extro-sim start <id>` — free, no re-upload |
| `JobCancelled` | the job was cancelled | — |
| `JobPreempted` | the compute provider killed the sandbox mid-run | not the user's code; billed only for the seconds that ran, so resubmit |
| `RateLimited` | HTTP 429; `.retry_after` when the server sent one | back off. Status polling retries 429s itself; submits are never auto-retried |
| `ExtropicError: cannot reach the Extropic API at ...` | wrong URL or no network | fix the URL; do not retry-loop |
| `ExtropicError: 401: ...` | missing or expired session | `extro-sim auth`, or fix `EXTROPIC_TOKEN` |
| `TimeoutError` from `.result(timeout=)` / `.wait(timeout=)` | only the local wait ended | the job is still running and still billing — `job.cancel()` unless it should finish |

**Platform faults.** Any message reading *"we couldn't run your job — this is on
us, not your code. Try again; if it keeps happening contact support with job
`<id>`."* means the platform failed, not the submitted function. Do not edit the
user's code in response to it. Resubmit once or twice; if it persists, report the
job id from the message. Terminal statuses `error` and `expired` and any withheld
detail produce this line. `EXTROPIC_DEBUG=1` reveals the raw platform detail.

**There is no default timeout.** Set none and the job runs until it finishes (a
24h ceiling backstops it). `timeout_s` is a hard wall-clock cap on the run: at the
deadline the runner kills your process and the job goes terminal as `timeout`. Set
one when you want a runaway bounded — a job bills for every second it runs.

Two different timeouts: `timeout_s` on the decorator (or `--timeout` on the CLI)
stops the job server-side; `.result(timeout=)` and `.wait(timeout=)` only bound the
local wait.

## Cost

Billing starts when the code starts running. Waiting for capacity, the image build,
and provisioning are all billed at zero, and a capacity bounce costs nothing.

`charged_total` on the job view (from `job.wait()` or `job.refresh()`) is what the
job cost. Every money figure crosses the wire as a decimal STRING, not a number —
`float(view["charged_total"])` before comparing or summing, or you get
`TypeError: '>' not supported between 'str' and 'float'`. `extro-sim balance`
prints credits, reserved, available, and the active job count; `Client().billing()`
returns those same figures (also as strings) from Python.
`extro-sim recharge <usd>` starts a top-up; the card payment itself is completed in
the console at https://extropic.dev.

Guardrails: pick the smallest tier that fits, set a `timeout_s` on anything that
could run away (a job bills until it is terminal, and there is no default cap),
and `job.cancel()` anything abandoned.

## CLI reference

Global flags: `--api-url`, `--no-color`.

| command | does |
|---|---|
| `extro-sim auth [login\|logout\|status]` | browser login / forget session / show identity |
| `extro-sim run FILE` | submit, follow output, print the result |
| `extro-sim submit FILE [--wait]` | submit and print the job id; `--wait` also supervises and collects |
| `extro-sim jobs [--limit N]` | list your jobs, newest first |
| `extro-sim supervise ID` | watch status until the job finishes |
| `extro-sim collect ID` | wait for and print a job's result |
| `extro-sim logs ID [-f] [--stderr]` | print or follow a job's output |
| `extro-sim start ID [--timeout S]` | retry a start that bounced on capacity. `--timeout` is NOT inherited from the original submit (the platform does not record it) — pass it again to re-apply a bound |
| `extro-sim cancel ID` | cancel a job |
| `extro-sim balance` | credit balance |
| `extro-sim recharge AMOUNT` | start a credit top-up |

`run` and `submit` also take `--entrypoint NAME` (default `main`),
`--tier cpu|l4|a100|h100` (default `cpu`), `--arg`, `--kwarg`,
`--timeout SECONDS` (no default — unbounded), and `--poll SECONDS`.

## Environment

| variable | effect |
|---|---|
| `EXTROPIC_TOKEN` | auth token, overrides the stored session |
| `EXTROPIC_DEBUG=1` | show raw platform error detail instead of the neutral line |
| `EXTROPIC_API_URL` · `EXTROPIC_UPLOAD_URL` · `EXTROPIC_DOWNLOAD_URL` | endpoint overrides |

The packaged defaults are production, but two things override them silently: a
session stored by `extro-sim --api-url <host> auth` pins every later call to that
host, and the server tells the client its own artifact host at job-create time.
`extro-sim balance` shows which deployment you are actually talking to.

A job id exists in exactly one environment, so if you override any endpoint,
override all of them to the same one. Note that a mixed set does NOT fail loudly:
the input upload follows the artifact host the create response names, so **the
submit succeeds** and only the reads fail — `logs`, `collect`, and `.result()` hit
the other environment's artifact host and 401/404. A submit that works is not
evidence your endpoints agree.

Skip Extropic for latency-sensitive calls and for code needing packages the image
does not ship.
