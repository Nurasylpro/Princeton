"""Recording plans and the label meaning of each state, shared by session.py and train_model.py.

STATE_MAP: state -> (people counted, moving). For the "zone" plan "people" means people
INSIDE the triangle formed by the iPhone and the two ESP32 boards.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    people: int              # people in the room
    state: str               # training label (key of the plan's STATE_MAP), or a teammates-only label
    say: str                 # shown + spoken instruction
    ruview_id: str | None    # RuView recording id (train_* -> RuView activity classifier), None = skip
    prompts: list[str] = field(default_factory=list)  # repeated spoken actions during the chunk
    teammates_block: bool = False


# ---- plan "room": first session (people anywhere in the room) --------------------------------
ROOM_STATES = {"empty": (0, False), "p1_still": (1, False), "p1_moving": (1, True),
               "p2_still": (2, False), "p2_moving": (2, True)}
ROOM_PLAN: list[Chunk] = [
    Chunk("empty_1", 0, "empty", "Nobody in the room.", "train_empty_1"),
    Chunk("empty_2", 0, "empty", "Nobody in the room.", "train_empty_2"),
    Chunk("p1_still_1", 1, "p1_still", "One person. Sit quietly on the chair.", "train_still_p1_1"),
    Chunk("p1_still_2", 1, "p1_still", "One person. Stand quietly.", "train_still_p1_2"),
    Chunk("p1_moving_1", 1, "p1_moving", "One person. Walk around, turn, wave your arms.", "train_moving_p1_1"),
    Chunk("p1_moving_2", 1, "p1_moving", "One person. Keep moving: walk, turn, wave.", "train_moving_p1_2"),
    Chunk("p2_still_1", 2, "p2_still", "Two people. Both sit or stand quietly.", "train_still_p2_1"),
    Chunk("p2_still_2", 2, "p2_still", "Two people. Both stay still, swap positions first.", "train_still_p2_2"),
    Chunk("p2_moving_1", 2, "p2_moving", "Two people. Both move around.", "train_moving_p2_1"),
    Chunk("p2_moving_2", 2, "p2_moving", "Two people. Both keep moving.", "train_moving_p2_2"),
    Chunk("p1_sitstand_1", 1, "p1_sitstand", "One person, standing next to the chair. Follow the commands.",
          None, ["sit down", "stand up"], True),
    Chunk("p1_sitstand_2", 1, "p1_sitstand", "One person, standing next to the chair. Follow the commands.",
          None, ["sit down", "stand up"], True),
    Chunk("p1_jump_1", 1, "p1_jump", "One person, standing. Jump once on each command.",
          None, ["jump"], True),
]

# ---- plan "zone": people inside the iPhone-ESP2-ESP3 triangle ---------------------------------
# "outside" = one person in the room but outside the triangle (still in rep 1, moving in rep 2),
# so the model learns not to count them.
ZONE_STATES = {"empty": (0, False), "in1_still": (1, False), "in1_moving": (1, True),
               "in2_still": (2, False), "in2_moving": (2, True), "outside": (0, False)}
ZONE_PLAN: list[Chunk] = [
    Chunk("empty_1", 0, "empty", "Nobody in the room.", "train_empty_z1"),
    Chunk("empty_2", 0, "empty", "Nobody in the room.", "train_empty_z2"),
    Chunk("in1_still_1", 1, "in1_still", "One person inside the triangle. Sit quietly.", "train_still_in1_1"),
    Chunk("in1_still_2", 1, "in1_still", "One person inside the triangle. Stand quietly.", "train_still_in1_2"),
    Chunk("in1_moving_1", 1, "in1_moving",
          "One person inside the triangle. Move: shift, turn, wave, small steps.", "train_moving_in1_1"),
    Chunk("in1_moving_2", 1, "in1_moving", "One person inside the triangle. Keep moving.", "train_moving_in1_2"),
    Chunk("in2_still_1", 2, "in2_still", "Two people inside the triangle. Both stay still.", "train_still_in2_1"),
    Chunk("in2_still_2", 2, "in2_still",
          "Two people inside the triangle. Swap places, then both stay still.", "train_still_in2_2"),
    Chunk("in2_moving_1", 2, "in2_moving", "Two people inside the triangle. Both move.", "train_moving_in2_1"),
    Chunk("in2_moving_2", 2, "in2_moving", "Two people inside the triangle. Both keep moving.", "train_moving_in2_2"),
    Chunk("outside_1", 1, "outside",
          "One person in the room but OUTSIDE the triangle. Stay still.", None),
    Chunk("outside_2", 1, "outside",
          "One person in the room but OUTSIDE the triangle. Move around outside it.", None),
]

# ---- plan "hall": count people INSIDE the taped triangle, normal crowd outside ----------------
# Every chunk is recorded with the hall's usual activity outside the tape, so the model learns
# to ignore it. "people" in the state map = people inside; 3 means "3 or more".
# Not recorded in RuView (its classifier is not used for the hall).
_OUT = " Everyone else stays outside the tape, at least one metre away."
HALL_STATES = {"in0": (0, False),
               "in1_still": (1, False), "in1_moving": (1, True),
               "in2_still": (2, False), "in2_moving": (2, True),
               "in3p_still": (3, False), "in3p_moving": (3, True)}
HALL_PLAN: list[Chunk] = [
    Chunk("in0_1", 0, "in0", "Nobody inside the triangle." + _OUT, None),
    Chunk("in0_2", 0, "in0", "Nobody inside the triangle." + _OUT, None),
    Chunk("in1_still_1", 1, "in1_still", "One person inside. Sit or stand quietly." + _OUT, None),
    Chunk("in1_still_2", 1, "in1_still", "One person inside, other spot. Stay still." + _OUT, None),
    Chunk("in1_moving_1", 1, "in1_moving", "One person inside. Move: turn, wave, small steps." + _OUT, None),
    Chunk("in1_moving_2", 1, "in1_moving", "One person inside. Keep moving." + _OUT, None),
    Chunk("in2_still_1", 2, "in2_still", "Two people inside. Both stay still." + _OUT, None),
    Chunk("in2_still_2", 2, "in2_still", "Two people inside. Swap places, then stay still." + _OUT, None),
    Chunk("in2_moving_1", 2, "in2_moving", "Two people inside. Both move." + _OUT, None),
    Chunk("in2_moving_2", 2, "in2_moving", "Two people inside. Both keep moving." + _OUT, None),
    Chunk("in3p_still_1", 3, "in3p_still", "Three or more people inside. All stay still." + _OUT, None),
    Chunk("in3p_still_2", 3, "in3p_still", "Three or more inside. Change places, then stay still." + _OUT, None),
    Chunk("in3p_moving_1", 3, "in3p_moving", "Three or more people inside. All move." + _OUT, None),
    Chunk("in3p_moving_2", 3, "in3p_moving", "Three or more people inside. Keep moving." + _OUT, None),
]

PLANS = {"room": (ROOM_PLAN, ROOM_STATES), "zone": (ZONE_PLAN, ZONE_STATES),
         "hall": (HALL_PLAN, HALL_STATES)}
