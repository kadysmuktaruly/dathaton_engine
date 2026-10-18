"""Idea 2: a close-out check on the engine's B5 'cashier_left_room' alarms.

5 of the engine's 6 false alarms on the training clips are this alarm (X-12, X-23, X-25, X-27,
X-29). Looking at the frames one second after the room empties:
- X-23, X-29: the cashier is in fact still sitting at the desk (a perception miss), so
  'is anyone in the room?' answers yes;
- X-12, X-25, X-27: the cashier really left, but the workplace is closed out: the laptop is
  closed or its screen is dark, and nothing is left on the desk;
- the labeled violations X-11, X-22, X-28 leave the laptop screen lit, X-24 leaves a tablet
  on the desk, X-26 leaves cash.
So an alarm is kept only if the room is empty AND something is left exposed (lit screen, items
on the desk, open safe). Checked on those frames: presence 7/7, screen 6/6, items 5/5.
Without enough budget the alarm is kept unchanged (recall first).
"""
from __future__ import annotations

from engine.video import iter_frames_at
from orig_and_vlm import config, eyes

YESNO = " Answer 'yes' or 'no' first, then one short reason."
PRESENT = ("Ceiling CCTV view of a small office room with a glass wall. Is any person inside this room right now "
           "(sitting or standing in the room)? Ignore people seen through the glass wall or outside the open "
           "door." + YESNO)
# 'switched on ... even a dark-blue one': X-22 (a violation) leaves a dark-blue lock screen, which the
# first wording ('lit, showing a bright picture') called dark. This one: 6/6 on the exit frames.
SCREEN = ("Zoomed ceiling CCTV view of an office desk. Look at the laptop. Is the laptop screen switched on, showing "
          "any picture, wallpaper, lock screen or document, even a dark-blue one? Answer 'no' only if the laptop "
          "is closed or its screen is completely black." + YESNO)
ITEMS = ("Zoomed ceiling CCTV view of a cash desk. Ignore the laptop, the telephone, the counting machine and the "
         "small banknote detector: they always stand there. Is anything else left lying on the desk, such as "
         "cash, a tablet, papers, a bag or a cup? Answer 'yes' or 'no' first, then name it.")
SAFE = ("Ceiling CCTV view of a bank cash room. Is the door of the safe (the grey metal cabinet) open?" + YESNO)
AFTER_EXIT_S = 1.0  # TUNE: look this long after the last frame with someone in the room


def laptop_box(scene, w: int, h: int):
    """The laptop with room around it (the screen fills a good part of the crop); None if no laptop."""
    b = scene.laptop
    if b is None:
        return None
    return (max(0, int(b.x0 - 2.0 * b.w)), max(0, int(b.y0 - 2.0 * b.h)),  # TUNE
            min(w, int(b.x1 + 2.0 * b.w)), min(h, int(b.y1 + 0.6 * b.h)))


def check(video: str, perc, desk_box, lap_box, t_exit: float) -> dict:
    """Answers at one frame just after the room empties; 'keep' = the alarm stands."""
    t = round(min(t_exit + AFTER_EXIT_S, perc.info.duration - 0.05), 2)
    _, actual, frame = next(iter_frames_at(video, [t], perc.info.crop))
    out = {"t": round(float(actual), 2)}
    out["present"], out["present_text"], _ = eyes.ask(frame, PRESENT)
    if out["present"]:
        out["keep"], out["why"] = False, "someone is still in the room (the departure was a perception miss)"
        return out
    out["screen"], out["screen_text"], _ = eyes.ask(eyes.crop(frame, lap_box or desk_box), SCREEN)
    out["items"], out["items_text"], _ = eyes.ask(eyes.crop(frame, desk_box), ITEMS)
    out["safe"], out["safe_text"], _ = eyes.ask(frame, SAFE)
    exposed = [k for k in ("screen", "items", "safe") if out[k]]
    out["keep"] = bool(exposed)
    out["why"] = ("left exposed: " + ", ".join(exposed)) if exposed else "workplace closed out (screen dark, desk clear, safe shut)"
    return out


def filter_events(video: str, perc, fa, events: list, budget=None) -> tuple[list, list[dict]]:
    """Drop B5 'cashier_left_room' alarms that fail the close-out check. Returns (kept, debug).

    With a fine-tuned detector (it has a `safe_open` class) engine/rules/b5.py keeps a departure
    below the verdict threshold unless the detector also sees an open safe. A passed close-out
    check (screen on / items left / safe open) is that evidence too, so it lifts the alarm back to
    the rule's normal score; with the stock detector the score is already there.
    """
    from engine.rules import b5
    from engine.vlm_scan import desk_crop_box
    crop = perc.info.crop
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    desk, lap = desk_crop_box(fa.scene, w, h), laptop_box(fa.scene, w, h)
    kept, dbg = [], []
    for ev in events:
        if not (ev.code == "B5" and ev.info.get("reason") == "cashier_left_room"):
            kept.append(ev)
            continue
        if budget is not None and not budget.ok(config.CLOSEOUT_NEED_S):
            ev.info["closeout"] = "skipped (budget)"
            kept.append(ev)
            continue
        r = check(video, perc, desk, lap, float(ev.info.get("t_exit", ev.end)))
        ev.info["closeout"] = r
        dbg.append({"t": ev.start, **r})
        if r["keep"]:
            if ev.score < b5.LEAVE_SCORE:
                ev.score = b5.LEAVE_SCORE
                ev.info["closeout_lifted"] = True
            kept.append(ev)
    return kept, dbg
