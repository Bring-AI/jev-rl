"""A deterministic game. Transitions contain facts, never numerical rewards."""

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache

MAP = ("#########", "#...K...#", "#.#.L.#.#", "#.#...#.#", "#.LL#...#", "#......E#", "#########")
ACTIONS = ("up", "right", "down", "left")
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
STARTS = ((1, 1), (1, 3), (1, 5))


@dataclass(frozen=True)
class State:
    x: int
    y: int
    has_key: bool = False


@dataclass(frozen=True)
class Transition:
    before: State
    action: int
    after: State
    event: str
    terminated: bool = False

    def context(self):
        game = Game()
        return {
            "before": game.describe(self.before),
            "action": ACTIONS[self.action],
            "after": game.describe(self.after),
            "event": self.event,
            "terminated": self.terminated,
        }


class Game:
    width = len(MAP[0])
    height = len(MAP)
    n_states = width * height * 2

    def encode(self, state: State) -> int:
        return (int(state.has_key) * self.height + state.y) * self.width + state.x

    def start(self, rng) -> State:
        return State(*STARTS[int(rng.integers(len(STARTS)))])

    def step(self, state: State, action: int) -> Transition:
        if not 0 <= action < 4:
            raise ValueError("Action must be 0, 1, 2 or 3.")
        if MAP[state.y][state.x] == "L" or ((state.x, state.y) == (7, 5) and state.has_key):
            raise ValueError("Reset after a terminal state.")
        dx, dy = DELTAS[action]
        x, y = state.x + dx, state.y + dy
        tile = MAP[y][x] if 0 <= x < self.width and 0 <= y < self.height else "#"
        if tile == "#":
            return Transition(state, action, state, "wall")
        if tile == "E" and not state.has_key:
            return Transition(state, action, state, "locked")
        after = State(x, y, state.has_key or tile == "K")
        event = "move"
        if tile == "L":
            event = "lava"
        elif tile == "E":
            event = "win"
        elif tile == "K" and not state.has_key:
            event = "key"
        return Transition(state, action, after, event, event in ("lava", "win"))

    @staticmethod
    @lru_cache(maxsize=256)
    def distance(state: State) -> int:
        """Engineered route feature for the judge only; the policy never sees it."""
        target = (7, 5) if state.has_key else (4, 1)
        queue = deque([(state.x, state.y, 0)])
        seen = {(state.x, state.y)}
        while queue:
            x, y, distance = queue.popleft()
            if (x, y) == target:
                return distance
            for dx, dy in DELTAS:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < Game.width
                    and 0 <= ny < Game.height
                    and MAP[ny][nx] not in "#L"
                    and (nx, ny) not in seen
                ):
                    seen.add((nx, ny))
                    queue.append((nx, ny, distance + 1))
        return Game.width * Game.height

    def describe(self, state):
        return {
            **asdict(state),
            "objective": "exit" if state.has_key else "key",
            "safe_steps_to_objective": self.distance(state),
        }

    def spec(self):
        return {
            "name": "Key Quest",
            "map": MAP,
            "starts": STARTS,
            "actions": ACTIONS,
            "rules": "Collect the key, then reach the exit. Lava ends the episode.",
        }
