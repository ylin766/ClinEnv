"""Submission tool.
Only responsible for validating and receiving submissions for the current decision stage.
"""


def submit_decision(required_fields, payload: dict) -> tuple[bool, str]:
    """Returns (True, msg) if submission is successful, (False, msg) otherwise."""
    raise NotImplementedError
