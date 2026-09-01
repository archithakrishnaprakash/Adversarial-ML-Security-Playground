"""
Threat Model Engine.

Before you run an attack, a real security assessment starts with a threat
model: what does the attacker actually have access to? A white-box attacker
with the model's weights and gradients can run FGSM/PGD/DeepFool directly. A
black-box attacker who only sees a prediction API cannot compute a gradient
at all — they're limited to query-based search or attacks transferred from a
substitute model they train themselves.

This module encodes that mapping so the rest of the app (API layer, UI) can
ask "given this capability, which attacks are even applicable?" instead of
silently trying to compute a gradient that shouldn't be available.
"""
from __future__ import annotations

CAPABILITIES = ["white_box", "gray_box", "black_box"]
GOALS = ["untargeted", "targeted", "confidence_reduction", "detection_evasion"]

# access an attacker has under each capability level — informational, shown in
# the UI so the threat model reads like an assessment brief rather than a
# dropdown with no context.
ACCESS_BY_CAPABILITY = {
    "white_box": ["model_parameters", "gradients", "training_data_distribution"],
    "gray_box": ["prediction_probabilities", "some_training_data_knowledge"],
    "black_box": ["api_only"],
}

# attack -> capability -> (applicable: bool, rationale: str)
_ATTACK_RULES = {
    "fgsm": {
        "white_box": (True, "Direct gradient access — single-step gradient attack applies as-is."),
        "gray_box": (True, "No direct gradients, but a surrogate trained on exposed probabilities "
                            "gives a usable gradient signal — run as a transfer attack."),
        "black_box": (False, "No gradient signal available at all without training a surrogate "
                              "first; use 'transfer' or 'query' instead."),
    },
    "pgd": {
        "white_box": (True, "Direct gradient access — the strongest attack available here."),
        "gray_box": (True, "Same surrogate-transfer reasoning as FGSM, iterated."),
        "black_box": (False, "Requires gradients this attacker doesn't have."),
    },
    "deepfool": {
        "white_box": (True, "Direct gradient access needed to linearize the decision boundary."),
        "gray_box": (True, "Runs against a surrogate; boundary estimate is only as good as the "
                            "surrogate's fidelity to the real model."),
        "black_box": (False, "Requires gradients this attacker doesn't have."),
    },
    "random_noise": {
        "white_box": (True, "Doesn't need gradients — always applicable, used as the baseline."),
        "gray_box": (True, "Doesn't need gradients — always applicable, used as the baseline."),
        "black_box": (True, "Doesn't need gradients — always applicable, used as the baseline."),
    },
    "transfer": {
        "white_box": (True, "Available, though pointless — you already have direct gradients."),
        "gray_box": (True, "The intended use case: attack a surrogate, transfer to the target."),
        "black_box": (True, "The realistic black-box strategy — train a substitute model on "
                             "queried predictions, attack that, transfer the result."),
    },
    "query": {
        "white_box": (True, "Available, though pointless — direct gradients are cheaper."),
        "gray_box": (True, "Usable, but a surrogate is usually more query-efficient here."),
        "black_box": (True, "The other realistic black-box strategy — zeroth-order search using "
                             "only the prediction API, no surrogate required."),
    },
}


def applicable_attacks(capability: str) -> dict:
    """Returns every known attack with whether it's applicable under the given
    capability, plus a one-line rationale. Raises ValueError for an unknown
    capability so callers get a clear 400 rather than a silent KeyError.
    """
    if capability not in CAPABILITIES:
        raise ValueError(f"Unknown capability '{capability}'. Must be one of {CAPABILITIES}")

    return {
        attack: {"applicable": rules[capability][0], "rationale": rules[capability][1]}
        for attack, rules in _ATTACK_RULES.items()
    }


def full_matrix() -> dict:
    """The whole capability x attack applicability table, for the threat-model
    selection UI."""
    return {capability: applicable_attacks(capability) for capability in CAPABILITIES}


def is_attack_applicable(attack: str, capability: str) -> tuple[bool, str]:
    """Convenience check used by the attack endpoint before it runs anything.
    Returns (applicable, rationale). Unknown attacks are treated as always
    applicable (rationale explains why) so this never blocks attacks the
    threat model simply doesn't have an opinion about yet.
    """
    if capability not in CAPABILITIES:
        raise ValueError(f"Unknown capability '{capability}'. Must be one of {CAPABILITIES}")
    if attack not in _ATTACK_RULES:
        return True, "No threat-model rule defined for this attack — allowed by default."
    return _ATTACK_RULES[attack][capability]
