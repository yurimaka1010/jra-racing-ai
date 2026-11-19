"""Invented multi-month examples, including missing odds and pending payments."""
from datetime import datetime, timedelta

from .core import JST, Outcome, PredictionRace, Schedule, Snapshot, WinPolicy, run_simulation


def demo_inputs():
    predictions, schedules, snapshots, outcomes = [], [], [], []
    for index in range(12):
        start = datetime(2030, 1+index//4, 5+index%4, 15, 0, tzinfo=JST)
        race = ("2030", start.strftime("%m%d"), "05", "01", "01", "01")
        prediction = PredictionRace(race, {"01": .45, "02": .30, "03": .15, "04": .10})
        predictions.append(prediction)
        schedules.append(Schedule(race, start, start.replace(hour=9), tuple(prediction.probabilities), True))
        if index not in (3, 7):
            odds = {"01": "2.0", "02": "4.0", "03": "8.0", "04": "12.0"}
            if index == 5:
                odds.pop("04")
            if index == 10:
                odds = {"01": "1.1", "02": "1.2", "03": "1.3", "04": "1.4"}
            snapshots.append(Snapshot(race, start-timedelta(minutes=12), start-timedelta(minutes=11), odds, verified=True))
            # A future tempting quote must never influence the earlier decision.
            snapshots.append(Snapshot(race, start-timedelta(minutes=4), start-timedelta(minutes=3),
                                      {"01": "99", "02": "99", "03": "99", "04": "99"}, verified=True))
        if index == 8:
            outcomes.append(Outcome(race, "special", reason="special_pending"))
        elif index == 9:
            outcomes.append(Outcome(race, "normal_final", {"02": 310, "03": 620}, True, True))
        else:
            winner = "02" if index % 2 == 0 else "01"
            # Official synthetic amounts deliberately differ from decision odds.
            outcomes.append(Outcome(race, "normal_final", {winner: 110 if index == 1 else 570 if winner == "02" else 190}, True, True))
    return predictions, schedules, snapshots, outcomes


def run_demo():
    inputs = demo_inputs()
    policy = WinPolicy(evaluation_start="2030-01-01", evaluation_end="2030-03-31", model_fit_end="2029-12-31")
    return run_simulation(*inputs, policy=policy), policy
