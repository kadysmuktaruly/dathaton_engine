"""Every question the VLM is asked. Short, targeted, one zone each: free captions lose the deciding details
(see orig_and_vlm/TEXTFLOW_REPORT.md), targeted questions on zoomed crops do not.

Bump VERSION when a prompt changes: it is part of the cache key.
"""
from __future__ import annotations

VERSION = "p1"

# ---- layout (once per clip) ----------------------------------------------------------------------
LAYOUT = ("Ceiling CCTV view of a bank cash room. Locate these objects and output a JSON list of "
          '{"label": ..., "bbox_2d": [x1, y1, x2, y2]}, using exactly these labels: '
          '"door" (the glass door of the room), "desk" (the cashier desk top), "safe" (the metal safe cabinet), '
          '"laptop", "counting_machine" (banknote counting machine), "uv_detector" (small banknote UV detector '
          "lamp, only if there is one). Skip objects that are not visible.")

# ---- state questions (one frame, one zone) -----------------------------------------------------------
PEOPLE = ("Ceiling CCTV view of a bank cash room with glass walls. Count only people physically INSIDE this room; "
          "ignore people seen through the glass walls, in the corridor or reflected in the glass. "
          "The cashier is the bank employee who works at the cash desk with the laptop and the counting machine. "
          'JSON only: {"cashier": "desk" (sits or stands at the cash desk) or "safe" (at the safe) or "walking" '
          '(moving elsewhere inside the room) or "absent" (no cashier inside the room), '
          '"visitors": <number of other people inside the room: clients, cash collectors, anyone else>}')

DOOR = ("Zoomed ceiling CCTV view of the glass wall of a bank cash room with a glass door. Is the door open "
        "(swung open, leaving a gap you could walk through) or closed (shut, flush with the glass wall)? "
        "Answer one word: open or closed.")

SAFE = ("Zoomed ceiling CCTV view of a grey metal safe (a small cabinet). JSON only: "
        '{"open": <true if the safe door is open or ajar>, '
        '"key": <true if a key is inserted in the lock, i.e. something sticks out of the keyhole; '
        "false if there is only an empty keyhole>}")

LAPTOP = ("Zoomed ceiling CCTV view of a laptop on a desk. What does the laptop screen show? Answer one word: "
          "active (an open program window: a spreadsheet, table, document, form or browser, usually bright white), "
          "locked (a lock or login screen: only a full-screen wallpaper photo or a plain dark-blue screen, maybe a "
          "clock or a password box, no program windows), "
          "dark (screen off or black), closed (lid closed), none (no laptop).")

DESK = ("Zoomed ceiling CCTV view of a bank cashier's desk. Look only at objects LYING ON the desk surface, not "
        "things held in someone's hands. Ignore the counting machine, the laptop, the telephone and a small "
        "banknote detector: they always stand there. Which of these lie on the desk? "
        '"cash" (paper money: a banknote, a stack or a bundle), '
        '"collection_bag" (a sealed cash-collection bag or sack), '
        '"stamps" (office rubber stamps or seals: small handled stamps, often in a colored plastic case), '
        '"handbag" (a handbag, backpack, purse or other personal bag), '
        '"clothing" (a jacket, coat or other clothing), '
        '"drink" (a cup, mug, glass, bottle or can), '
        '"phone" (a mobile phone or tablet), '
        '"other" (any other personal item: food, cosmetics, wallet, keys). '
        "Answer only with the matching names separated by commas, or none.")

# ---- action windows (several frames in time order) ---------------------------------------------------
ACTIONS = ("These {n} frames, {dt:g} s apart in time order, are a zoomed ceiling CCTV view of a bank cashier's desk "
           "and the people at it. Which of these happen in these frames? "
           '"card_in" (a visitor hands over a small plastic ID card, the size of a bank card, held in one hand; '
           'not banknotes), '
           '"cash_in" (a visitor hands paper banknotes to the cashier or puts them on the desk for her), '
           '"cash_out" (the cashier hands banknotes to a visitor), '
           '"machine" (banknotes are in the counting machine or being fed into it), '
           '"hand_count" (the cashier counts banknotes by hand), '
           '"uv" (a banknote is held under the small UV detector lamp), '
           '"bag_in" (a visitor hands a sealed cash-collection bag to the cashier), '
           '"bag_open" (the cashier opens or cuts a cash-collection bag), '
           '"typing" (the cashier types on the laptop), '
           '"papers" (a visitor shows or hands a document or paper to the cashier). '
           "Answer only with the matching names separated by commas, or none.")
ACTION_KEYS = ["card_in", "cash_in", "cash_out", "machine", "hand_count", "uv", "bag_in", "bag_open", "typing",
               "papers"]
DESK_KEYS = ["cash", "collection_bag", "stamps", "handbag", "clothing", "drink", "phone", "other"]
