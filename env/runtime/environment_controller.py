"""Environment controller.
Only responsible for episode workflow scheduling: trigger stage -> distribute tools -> record results -> advance stage.
"""


def run_episode(case_data, mut_client):
    """Run a complete case episode."""
    raise NotImplementedError
