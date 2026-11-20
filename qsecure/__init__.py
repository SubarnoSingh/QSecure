"""QSecure - Quantum-Inspired Cyber Threat Detection for Digital Signature Security.

Subpackages: :mod:`qsecure.core` (quantum primitives), :mod:`qsecure.protocol`
(teleportation-based quantum digital signatures) and :mod:`qsecure.attacks`
(adversaries and the randomness-isolation check).

Examples
--------
>>> import numpy as np
>>> from qsecure import attacks, protocol
>>> transcript = protocol.QDSSession(
...     protocol.ProtocolParams(key_length=192), rng=np.random.default_rng(2)
... ).run(0)
>>> transcript.transferable, transcript.repudiated
(True, False)
>>> attacks.assert_attack_isolated(
...     attacks.OutsideForger, attacks.signer_probe()
... ).isolated
True
"""

from qsecure import attacks, core, protocol

__version__ = "0.1.0"

__all__ = ["attacks", "core", "protocol", "__version__"]
