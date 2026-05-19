"""Action recommendation domain services."""
try:
    from .action_recommender import recommend_actions
except ImportError:
    recommend_actions = None  # type: ignore[assignment,misc]

try:
    from .action_catalog import get_actions_for_domain
except ImportError:
    get_actions_for_domain = None  # type: ignore[assignment,misc]

try:
    from .action_guard import ActionGuard, GuardedAction
except ImportError:
    ActionGuard = None  # type: ignore[assignment,misc]
    GuardedAction = None  # type: ignore[assignment,misc]

__all__ = ["recommend_actions", "get_actions_for_domain", "ActionGuard", "GuardedAction"]
