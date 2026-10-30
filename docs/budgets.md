# Forecast and limit production spending

Annotate production tasks with `estimatedCostUsd`. Use `0` for tasks you know
are local and free. Missing amounts remain unknown, rather than being counted
as free. Estimates are supplied by you; they are not provider quotes.

```bash
uv run tools/build.py --project projects/demo --budget-usd 5 --json
uv run tools/build.py --project projects/demo --budget-usd 5 --run
uv run tools/budget.py --project projects/demo
```

You can save the cap as `production.budgetUsd` in project.json or the production
plan. The CLI cap overrides it. Preview reports pending estimates, prior cost
reservations, unknown amounts and whether execution is allowed. A capped run
fails before executing any commands if costs are unknown or the projected
total exceeds the cap. Costs of resumed tasks are not charged again.

Each attempted task reserves its estimate in the build checkpoint before
execution. Failed and interrupted attempts keep their reservation, since a
provider may still have charged for them. Reconcile a reservation with the
actual provider charge using the attempt ID printed by `tools/budget.py`:

```bash
uv run tools/budget.py --project projects/demo --attempt ATTEMPT_ID --actual-usd 0.43 --note "Invoice reference"
```

The report distinguishes estimated amounts and reconciled charges. An estimate
cap cannot guarantee the provider's actual bill. The tool controls which work
starts; it cannot stop an already submitted hosted job at a particular dollar
amount. Built-in generator tasks have unknown estimates until you supply them
in an explicit plan. Choose faster/cheaper generation settings in that plan's
command arguments; changing those settings invalidates the relevant outputs.

For reusable draft/final settings, add named `production.profiles` to a plan:

```json
{"profiles": {"draft": {"tasks": {"voice-intro": {"command": ["{python}", "{toolkit}/tools/voiceover.py", "--script", "{project}/public/audio/scenes/intro.txt", "--output", "{project}/public/audio/scenes/intro.mp3", "--provider", "qwen3", "--cloud", "modal"], "estimatedCostUsd": 0.01}}}}}
```

Use the same task IDs as the base plan and complete argv arrays. Select with
`--profile draft` or `--profile final`. Profiles may override commands, cost
estimates, timeouts and declared inputs. They preserve dependencies and output
ownership. Switching quality invalidates the changed tasks automatically.
