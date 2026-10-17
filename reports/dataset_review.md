# Dataset review (Step 1)

Sources: `reports/sheets/<clip>_tc<t>_zoom.jpg` (full-resolution crop of the room, every 0.5 s from t−3 to t+3 s), `<clip>_tc<t>.jpg` (t−4 to t+4 s, every 1 s), and `<clip>_overview.jpg` (whole clip, every 2 s). Regenerate them with `uv run python -m tools.contact_sheets`.

**Data facts:** 42 files make 40 unique clips. `X-47` is a byte-identical copy of `X-4`, so X-4 has two labeled violations, at 57 s and 68 s. `X-48` is a copy of `X-5`. That leaves 21 clips with violations (22 timecodes) and 19 clean clips. All clips use the same camera at 464×848 with no black bars. The furniture is in 3 layouts: (a) desk across, cashier facing away from the camera, safe on a white stand to the right; (b) desk along the wall, safe behind the desk by the glass; (c) desk rotated, counting machine on the right, blue chair. Actors play different roles in different clips (the cashier in the pink cardigan or the white shirt; the men are clients, collection officers or outsiders).

Confidence: **H** = clear on the sheets; **M** = plausible, other readings possible; **L** = I can't tell what the staged violation is.

## Labeled violations

| Clip | t (s) | Proposed | Conf. | What is visible |
|---|---|---|---|---|
| X-1 | 22 | A3? / A5? | L | Cashier takes notes from the counting machine (19.5 s), flips through them at face level (20.5–22 s), hands them to the client (22.5–23 s). |
| X-2 | 10 | A5? / A1? | L | Client puts one blue note (or card) on the desk (9–10 s); cashier picks it up and examines it low, near her chest (10.5–13 s), then reaches back toward the client (14 s). |
| X-4 | 57 | **B1** | H | A bundle of notes lies open on the desk, not in the drawer or safe, while the next client walks in (56–59 s). |
| X-4 | 68 | A2? / B1 | L | Client's blue notes are counted by hand on the desk, not in the machine; the earlier bundle is still lying open. |
| X-11 | 18 | **LEAVE** (see below) | M | Cashier alone: turns the safe key/handle (13–16 s), switches off the light, walks out (17.5–21 s). The laptop stays open and on. |
| X-13 | 47 | B7 / A2 | M | Cashier alone counts notes by hand (45–47 s), opens the safe and puts them in (47.5–50 s). |
| X-15 | 15 | **B1** (+LEAVE) | H | Green collection bag / packet left on the desk; cashier leaves the room (15–18 s); something stays in the safe lock. |
| X-17 | 7 | **D1** (alt. D4) | H | Collection officer walks in with a sealed green bag and puts it on the desk (7 s); cashier pulls it in at once with no visible check of the seal or bag (7.5–10 s). |
| X-18 | 20 | **C1** | H | While a client is being served, a second man walks in and hands money to the cashier from the side (19–21 s). |
| X-20 | 5 | **B5** | H | Safe door swung open (2–5 s); cashier walks back to the desk and sits, leaving the safe open behind her (5.5–8 s). |
| X-22 | 14 | **LEAVE** | M | Cashier at the safe (11–13 s), then walks out (14–17 s). Laptop open and on; something visible in the safe lock. |
| X-24 | 8 | **LEAVE** | M | Cashier touches the safe (6–7.5 s), walks out (8–11 s). Tablet and laptop open on the desk. |
| X-26 | 9 | **B1** | H | Cashier counts notes, puts the bundle on the desk (8.5–9 s), gets up and walks away, leaving the cash on the desk (9.5–12 s). |
| X-28 | 7 | **LEAVE** | M | Cashier gets up from the laptop (6.5–7 s) and walks out (8–10 s). Laptop on. No visible interaction with the safe. |
| X-30 | 13 | **C1** | H | Layout (c). A man walks in (12.5–13 s) and goes around the desk to stand right beside the cashier at the counting machine (14–16 s). |
| X-32 | 5 | **C1** | M | Empty cash room; a person walks in (4.5–5 s) and goes to the cashier's chair. (Same actress who plays the cashier elsewhere.) |
| X-33 | 6 | **C1** | M | Empty cash room; a person walks in (5–6 s) and reaches over the counting machine (8.5–9 s). |
| X-34 | 13 | **A1** | H | Client gets up and leaves (11–15 s) while the cashier is still feeding his money into the counting machine (12.5–16 s); the count finishes without the client there. |
| X-38 | 33 | **B1** | H | A bundle of notes lies open on the desk as a new client walks in and sits down (31–36 s). |
| X-40 | 17 | **B3** | H | Cashier brings a coat and a handbag onto the cash desk (14–20 s). |
| X-44 | 16 | **B7** | M | Cashier alone counts a stack of notes by hand at the desk; no client, no second employee (13–19 s). |
| X-46 | 11 | X0? / B3? | L | A bundle of notes lies on the desk; cashier picks up a cup and drinks (10.5–14 s). Drinking at the cash desk isn't a catalog code. |

**LEAVE group (X-11, X-22, X-24, X-28):** the cashier, alone, leaves the cash room. Clean clip X-12 shows the same walk-out (safe, lights off, leaves), so leaving by itself is *not* the violation. The visible differences in the violation clips are that the laptop stays open and on, and in X-11, X-15 and X-22 something seems to be left in the safe lock. A key-in-lock is a few pixels at this resolution, so I can't confirm it. Closest catalog codes: B5 ("ключи оставлены", safe unattended) or A4 (type name "Кассир покинул рабочее место"; its cue assumes a client is present).

## Clean clips checked
| Clip | What happens (all labeled clean) |
|---|---|
| X-3 | Client comes in, puts cash down, cashier counts on the machine and returns change; all in view. |
| X-10 | Cashier alone at the desk, working on papers / small items. |
| X-12 | Cashier works on the laptop, goes to the safe, switches off the light, leaves the room (laptop closed or dark). |
| X-14 | Cashier alone: 40 s with a green collection bag at the safe, then about 100 s at the desk handling small items, possibly cash. **This conflicts with B7 for X-44** unless X-14 involves documents rather than cash. |
| X-35 | Full normal service: client comes in, cash goes through the machine, cashier opens the safe *with the client present*, puts the cash in, closes it; client leaves. |

Effects on the rules: opening the safe while a client is present is normal (X-35), and so is leaving the room after a proper close-out (X-12). Counting alone may be normal (X-14).

## Distinct types (as proposed)
| Type | Count | Clips |
|---|---|---|
| B1 cash left open on the desk | 4 (+1) | X-4@57, X-15, X-26, X-38 (maybe X-4@68) |
| C1 outsider in the cash room | 4 | X-18, X-30, X-32, X-33 |
| LEAVE: cashier leaves the room with the workplace unsecured | 4 | X-11, X-22, X-24, X-28 |
| B7 counting alone / A2 | 2 | X-13, X-44 |
| B5 safe left open | 1 | X-20 |
| A1 counting finished without the client present | 1 | X-34 |
| D1 collection bag accepted without a check | 1 | X-17 |
| B3 personal items at the desk | 1 | X-40 |
| Unclear | 4 | X-1, X-2, X-4@68, X-46 |

## What the detector needs
- COCO stock: `person`, `laptop`, `handbag`/`backpack`/`suitcase` (B3), `cup` (X-46), `cell phone`, `dining table` (desk; to be checked in Step 2).
- Fine-tune candidates, if stock detection fails in Step 2: `cash` (notes and bundles; very small at 464 px, the main risk), `safe`, `safe_open` (door swung open), `counting_machine`, `collection_bag` (green sealed bag), `door_open` (the glass door, for entry/exit events).
- Probably out of reach for a detector: a key in the safe lock. If LEAVE needs it, that type goes to the VLM fallback.

## Confirmed by the user (2026-10-17)
| Code | Clips | Implementation |
|---|---|---|
| B1 | X-4@57, X-15, X-26, X-38 | rule (cash on the desk with no cashier hand nearby, or cash lying open while a new person arrives) |
| C1 | X-18, X-30, X-32, X-33 | rule (a new person enters and reaches the staff side, or the room is empty) |
| B5 | X-20, X-11, X-22, X-24, X-28 | rule (safe open with nobody near it; or the cashier leaves the room alone after a session) + VLM check (key in lock / laptop left on) |
| B7 | X-13, X-44 | rule (lone cashier with banknotes in hand at the desk for ≥ N s) + VLM confirmation that the notes are cash; X-14 must stay clean |
| A1 | X-34 | rule (client leaves while cash is still at the machine or in the cashier's hands) |
| D1 | X-17 | rule (collection bag appears on the desk and is moved off within a short time without a pause to inspect it) |
| B3 | X-40 | rule (handbag/backpack/coat on the desk or staff side) |
| A3 / A5 / A2 / X0 | X-1, X-2, X-4@68, X-46 (best guesses) | VLM catch-all / fallback only |

### Correction (2026-10-17, confirmed by the user)
X-32 and X-33 are **C2 (cash-room door left open)**, not C1. The clean clip X-31 shows the same walk-in, but the cashier closes the glass door behind her (7 s). In X-32 the door stays open from 5 s to the end. In X-33 it stays open and a man appears in the open doorway at 11–13 s. C1 is now X-18 and X-30. C2 is checked by the VLM on frames after an entry (there is no door class in the detector).
