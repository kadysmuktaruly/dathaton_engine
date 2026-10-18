"""Timeline rules on synthetic VLM answers (no model needed)."""
from pure_vlm import timeline


def obs(duration=20.0, people=None, door=None, safe=None, laptop=None, desk=None, actions=None):
    def ser(d):
        return {f"{t:.2f}": a for t, a in (d or {}).items()}
    return {"duration": duration, "states": {"people": ser(people), "door": ser(door), "safe": ser(safe),
                                             "laptop": ser(laptop), "desk": ser(desk)},
            "actions": actions or [], "moments": {}}


def P(c, v=0):
    return {"cashier": c, "visitors": v}


def codes(o):
    return [(e.code, e.t) for e in timeline.detect(o)]


def test_door_open_long_is_c2_short_pass_is_not():
    # no entry seen: timecode = door opened + 2 s
    assert codes(obs(door={0: "closed", 5: "open", 12: "closed"})) == [("C2", 7.0)]
    # someone comes in at 6 s: timecode = entry + 2 s
    assert codes(obs(door={0: "closed", 5: "open", 14: "closed"},
                     people={0: P("absent"), 6: P("walking"), 8: P("desk")})) == [("C2", 8.0)]
    assert codes(obs(door={0: "closed", 5: "open", 7: "closed"})) == []
    assert codes(obs(duration=12, door={0: "closed", 6: "open"})) == [("C2", 8.0)]  # open until the end, 6 s
    assert codes(obs(duration=10, door={0: "closed", 6: "open"})) == []  # 4 s at the end: too short


def test_safe_open_while_cashier_at_desk():
    o = obs(people={0: P("walking"), 2: P("safe"), 5: P("desk"), 8: P("desk"), 11: P("desk")},
            safe={0: {"open": False}, 2: {"open": True}, 11: {"open": True}})
    assert codes(o) == [("B5", 5.0)]
    o = obs(people={0: P("desk"), 2: P("safe"), 5: P("safe"), 6: P("desk")},
            safe={0: {"open": False}, 2: {"open": True}, 6: {"open": False}})
    assert codes(o) == []


def test_departure_close_out():
    o = obs(people={0: P("desk"), 8: P("walking"), 10: P("absent"), 14: P("absent")},
            desk={0: [], 10: ["cash"], 11: ["cash"]}, laptop={11: "active"}, safe={0: {"open": False}})
    assert sorted(codes(o)) == [("B1", 8.0), ("B9", 8.0)]
    o = obs(people={0: P("desk"), 8: P("walking"), 10: P("absent"), 14: P("absent")},
            desk={0: [], 11: []}, laptop={11: "locked"}, safe={0: {"open": False}, 11: {"open": False}})
    assert codes(o) == []


def test_absent_from_start_is_not_a_departure():
    o = obs(people={0: P("absent"), 4: P("walking"), 6: P("desk")}, desk={0: ["cash"]}, laptop={0: "active"})
    assert codes(o) == []


def test_personal_item_needs_two_sightings():
    assert codes(obs(desk={0: [], 4: ["handbag"], 6: ["handbag", "clothing"]})) == [("B3", 4.0)]
    assert codes(obs(desk={0: [], 4: ["drink"], 8: []})) == []


def W(t0, *keys):
    return {"t0": t0, "t1": t0 + 1.5, **{k: True for k in keys}}


def test_collection_bag_without_id_is_d7_with_id_is_not():
    people = {0: P("desk"), 4: P("desk", 1), 12: P("desk", 1), 14: P("desk", 0)}
    desk = {0: [], 6.5: ["collection_bag"], 8: ["collection_bag"]}
    assert codes(obs(people=people, desk=desk, actions=[W(4, "bag_in"), W(8, "bag_open")])) == [("D7", 6.5)]
    assert codes(obs(people=people, desk=desk, actions=[W(4, "bag_in"), W(8, "card_in")])) == []


def test_old_cash_when_the_next_client_arrives_is_b1():
    people = {0: P("desk", 1), 10: P("desk", 0), 14: P("desk", 1), 18: P("desk", 1)}
    desk = {0: ["phone"], 4: ["phone", "cash"], 8: ["cash"], 13: ["cash"], 16: ["cash"]}
    assert ("B1", 14.0) in codes(obs(people=people, desk=desk))
    desk = {0: ["phone"], 4: ["phone", "cash"], 8: ["cash"], 13: [], 16: ["phone"]}
    assert ("B1", 14.0) not in codes(obs(people=people, desk=desk))


def test_cash_before_any_id_card_is_a6():
    people = {0: P("desk"), 4: P("desk", 1), 16: P("desk", 1)}
    assert codes(obs(people=people, desk={0: [], 8: ["cash"], 10: ["cash"]})) == [("A6", 8.0)]
    assert codes(obs(people=people, desk={0: [], 6: ["phone"], 8: ["cash"], 10: ["cash"]})) == []
