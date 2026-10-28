# OpenCV Behavioral Parity

This tool turns the pinned OpenCV declaration inventory into a module behavior
contract. Declaration identities remain an audit input; behavioral completion
is measured by stable behavior IDs with explicit domains, ownership, errors,
Terlan operation selectors, upstream documentation groups, and evidence.

The ordinary gate validates the source ledger, classifies every exact upstream
identity, expands every current Terlan operation into exactly one behavior, and
rejects a stale generated report:

```sh
make opencv-behavioral-parity-check
```

The strict gate additionally fails while any admitted CPU behavior remains
incomplete:

```sh
make opencv-behavioral-parity-strict-check
```

Refresh the deterministic report only after reviewing a ledger or inventory
change:

```sh
make opencv-behavioral-parity-refresh
```

Implemented behaviors require executable package evidence. Incomplete
behaviors retain explicit inventory evidence and keep the strict gate red.
Optional host or platform backends remain classified without blocking CPU
completion.
