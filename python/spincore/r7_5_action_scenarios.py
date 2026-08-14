from __future__ import annotations

from spincore.solver import Episode

SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


def action_scenario_cycle(domain: str) -> tuple[Episode, ...]:
    """Mechanical package-local copy of the accepted R7.4 scenario cycle.

    R7.5.4 changes action abstraction only. Scenario/stacks/dealer coverage must
    remain exactly the R7.4 domain contract rather than drifting with the new
    worker implementation.
    """
    if domain == "TRUE_HEADS_UP":
        episodes = []
        for stacks in ((0, 750, 750), (0, 500, 1000), (0, 1000, 500)):
            for dealer in (1, 2):
                episodes.append(Episode(1500, True, 0, 10, 20, stacks, dealer, (0,)))
        return tuple(episodes)
    if domain == "THREE_HANDED":
        episodes = []
        profiles = (
            (500, 500, 500),
            (250, 500, 750),
            (250, 750, 500),
            (500, 250, 750),
            (750, 250, 500),
        )
        for stacks in profiles:
            for dealer in (0, 1, 2):
                episodes.append(Episode(1500, False, 0, 10, 20, stacks, dealer, ()))
        return tuple(episodes)
    raise ValueError(f"unsupported action scenario domain {domain!r}")


def scenario_descriptor(episode: Episode) -> dict:
    return {
        "game_is_hu": bool(episode.game_is_hu),
        "stacks": list(episode.stacks),
        "dealer_id": int(episode.dealer_id),
        "dead_players": list(episode.dead_players),
    }
