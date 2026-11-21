# QSecure

Quantum-inspired threat detection for digital signatures.

QSecure simulates a teleportation-based quantum digital signature (QDS) protocol
between a signer and two recipients, mounts attacks against it (forgery,
impersonation, replay, count starvation, channel tampering, unauthorised
verification), and runs a statistical detector that flags each attack from the
session's own records. A web dashboard runs sessions live and shows the
detector's verdict beside the replayed session.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the dashboard

```bash
python -m qsecure.web
```

Then open http://127.0.0.1:8141. Use `--host` and `--port` to change where it
listens. If the API is unreachable, the frontend falls back to recorded runs.

## Project layout

| Package | What it does |
|---|---|
| `qsecure.core` | Quantum primitives: states, Pauli algebra, measurement, teleportation |
| `qsecure.protocol` | The QDS protocol: key distribution, symmetrisation, signing, verification, check rounds |
| `qsecure.attacks` | Adversaries that attack the protocol through its own interfaces |
| `qsecure.detect` | Detection statistics, thresholds and the composite detector |
| `qsecure.eval` | Evaluation harness: seeded, resumable sweeps, ROC and security curves |
| `qsecure.audit` | Security event log |
| `qsecure.web` | FastAPI backend and the static dashboard frontend |

## Checks

Every numeric claim in the docstrings is a doctest:

```bash
python -m pytest
```

## Usage from Python

```python
import numpy as np
from qsecure import attacks, protocol

transcript = protocol.QDSSession(
    protocol.ProtocolParams(key_length=192), rng=np.random.default_rng(2)
).run(0)
print(transcript.transferable, transcript.repudiated)
```
