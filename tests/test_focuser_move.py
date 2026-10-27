from unittest.mock import MagicMock

import pytest

import astra.autofocus
from astra.autofocus import AstraFocuser


class FakeFocuser:
    """Starts moving only after `delay` polls, as drivers that report IsMoving late."""

    def __init__(self, position=7000, delay=3, duration=5, stuck=False):
        self.position = position
        self.delay, self.duration, self.stuck = delay, duration, stuck
        self.target = None
        self.polls = 0

    def get(self, name, **kwargs):
        if name == "Absolute":
            return True
        if name == "MaxStep":
            return 100_000
        if name == "Move":
            self.target, self.polls = kwargs["Position"], 0
            return None
        if name == "IsMoving":
            self.polls += 1
            moving = self.delay < self.polls <= self.delay + self.duration
            if (
                self.target is not None
                and not self.stuck
                and self.polls > self.delay + self.duration
            ):
                self.position = self.target
            return moving and not self.stuck
        if name == "Position":
            return self.position
        raise AssertionError(name)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(astra.autofocus.time, "sleep", lambda _: None)


def make_focuser(device):
    observatory = MagicMock()
    observatory.check_conditions.return_value = True
    return AstraFocuser(observatory, device, settle_time=0), observatory


def test_waits_for_a_move_that_starts_late():
    device = FakeFocuser()
    focuser, observatory = make_focuser(device)
    focuser.move_focuser_to_position(8000)
    assert device.position == 8000
    observatory.logger.warning.assert_not_called()


def test_warns_if_the_focuser_does_not_move():
    device = FakeFocuser(stuck=True)
    focuser, observatory = make_focuser(device)
    focuser.move_focuser_to_position(8000, start_timeout=0.01)
    assert device.position == 7000
    messages = [call.args[0] for call in observatory.logger.warning.call_args_list]
    assert any("did not start moving" in message for message in messages)
    assert any("not at the requested 8000" in message for message in messages)


def test_move_to_the_current_position_does_not_wait():
    device = FakeFocuser()
    focuser, observatory = make_focuser(device)
    focuser.move_focuser_to_position(7000, start_timeout=0.01)
    observatory.logger.warning.assert_not_called()
