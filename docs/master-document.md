# Cashier routine, violations, and tracking list (simplified)

## Context
These are the reference rules for the dathaton engine's violation detector. They come only from the user's clip notes (X-1 … X-47) and the PDF table "Сводная таблица нарушений". **Priority goes to violations seen in the training videos.** PDF-only rules that would need new logic or new detections are dropped.

Simplifying assumptions:
- **One cashier, working alone.** No other bank staff exist. People in the room are the cashier, clients, and the collector (инкассатор). The collector enters the room and hands the bag over the desk; that is normal.
- **Client ID = government ID card (`id_card`) only.** A phone with a digital document is ignored.
- **Collection bags always arrive intact.** There are no damaged-bag cases.
- **Allowed on the desk:** counting machine, UV detector, laptop, wired telephone, the client's `id_card`, cash, sealed collection bag, stamps (печати), journal. **Everything else is a personal item (B3).**
- **Safe key:** it may stay in the safe lock while the cashier is in the room. The cashier must take it out when leaving the workspace.
- **UV detector:** the check is required only if a detector is in the scene.
- **F1 (X-11) = B9** in the PDF.

**Laptop answer:** in the provided material, a missing system entry is a violation **only for the collection bag (D6, X-13)**. For client operations, no code flags a missing laptop entry. A7 and B1 trigger on the next client arriving or cash left on the desk, not on the laptop. So the laptop entry is **optional for client operations and required for collection**. The laptop also matters for **B9**: it must be locked or closed when the cashier leaves.

---

## 1. Ideal cashier routine

### Rules that hold at every moment
- The door is closed. The cashier closes it after every pass. *(C2)*
- The desk holds only allowed items. *(B3)*
- The desk holds only the **current client's** cash. *(B1)*
- The safe is closed except while putting cash or items in or taking them out. The key may stay in the lock while the cashier is in the room. *(B5)*
- One client at a time. *(A7)*

### Phase 1. Opening the workplace (X-30, X-32/33, X-40)
1. Leave the personal bag and jacket outside.
2. Enter.
3. **Close the door.**
4. Turn on the laptop and start work.

### Phase 2. Taking cash from the safe (X-20)
1. Open the safe.
2. Take the cash.
3. **Close the safe.**
4. Sit at the desk and work with the cash.

### Phase 3. Accepting cash from a client (X-2, X-4, X-18, X-34, X-38, X-43/44, X-47/48)
1. The client enters and gives the `id_card`. The cashier **takes it in hand**.
2. The client gives the cash.
3. The cashier counts it on the **counting machine while the client is present**. The client does not leave before counting ends.
4. **If a UV detector is in the scene:** the cashier checks the cash on it.
5. *(Optional)* Enter the operation on the laptop.
6. Put the cash into the **safe** and close it.
7. Return the `id_card`.
8. The client leaves.
9. The next client starts at step 1.

### Phase 4. Issuing cash to a client (X-1)
1. The client gives the `id_card`, and the cashier takes it in hand.
2. The cashier counts the cash on the machine.
3. The cashier separates the amount to issue.
4. **Recount the issued part in front of the client.** Do not flag if the issued part is a whole bank-packed bundle.
5. Hand it to the client.

### Phase 5. Receiving a collection bag (X-13, X-15, X-17)
1. The collector enters with the **collection bag**.
2. **The cashier checks the collector's service ID and power of attorney.** *(D7)*
3. The cashier takes the sealed bag.
4. The cashier opens the bag.
5. **Count the contents.**
6. Put the cash into the safe and close it.
7. **Enter the operation on the laptop.** *(D6)*
- The bag, sealed or opened, is never left on the desk when the cashier leaves. *(B1)*

### Phase 6. Leaving the workplace (X-11, X-22/23, X-24/25, X-26, X-32/33)
1. Put **all cash** into the safe. *(B1)*
2. Put **stamps** into the safe. *(B8)*
3. **Close the safe and take the key.** *(B5)*
4. **Close or lock the laptop.** *(B9)*
5. Leave, then **close the door**. *(C2)*

---

## 2. Violations

### 2a. Step-level violations (a step skipped or done out of order)

| Code | Phase / step | Violation | Clip |
|---|---|---|---|
| C2 | 1.3 / 6.5 | Door left open after passing. Flagged even if nobody enters. | X-30, X-32, X-33 |
| B3 | 1.1 | Personal bag or jacket placed on the desk | X-40 |
| B5 | 2.3 | Safe left open while the cashier works at the desk | X-20 |
| A6 | 3.1 / 4.1 | Cash taken without the cashier ever taking the client's `id_card` in hand | X-2 |
| A1 | 3.3 | Client leaves while the cash is still being counted | X-34 |
| A2 | 3.3–3.4 | Cash not counted, **or** a UV detector is in the scene but not used | X-44 (vs X-43) |
| B1 | 3.6 | Client 1's cash still on the desk when client 2 enters or is served. Worst case: mixed with client 2's cash. Do not flag if the cash belongs to the current client. | X-4, X-38, X-47 (vs X-48) |
| A7 | 3.9 | Client 2's `id_card` or cash taken while client 1 is still present and unfinished | X-18 |
| A3 | 4.4 | Issued part handed over without its own recount | X-1 |
| D7 | 5.2 | Sealed bag taken without checking the collector's ID or power of attorney | X-17 |
| D6 | 5.5–5.7 | Bag opened, bundles put into the safe, with no count and no laptop entry | X-13 |
| B1 | Phase 5 / 6.1 | Sealed bag left on the desk, cashier leaves | X-15 |
| B1 | 6.1 | Cash left on the desk, cashier leaves | X-26 |
| B8 | 6.2 | Stamps left on the desk, cashier leaves | X-24 (vs X-25) |
| B5 | 6.3 | Safe open, **or key left in the safe**, when the cashier leaves | X-22 (vs X-23) |
| B9 | 6.4 | Cashier leaves with the laptop open and active | X-11 |

### 2b. Process-wide violations (checked at any time)

| Code | Condition |
|---|---|
| C2 | Door left open |
| B3 | A non-allowed item is on the desk or used at the workplace, e.g. coffee cup next to cash (X-46), bag or jacket (X-40) |
| B1 | Cash on the desk that is not the current client's |
| B5 | Safe open while the cashier is not putting in or taking out |
| A7 | Two clients' operations overlap |

### 2c. Optional (PDF-only, very cheap with the tracks above)
- **A4:** The cashier leaves the room while a client is present and that client's cash or `id_card` is on the desk. This uses only the person and object tracks above.

### Dropped
A5, B2, B4, B6, B7, C1, C3, D1, D2, D3, D4, D5, E1. They are not in the videos, contradict the setup (solo cashier, collector enters the room, intact bags), or need new detections. A smartphone on the desk is still caught as B3.

---

## 3. What the visual model must track

### Objects
| Object | States / attributes | Used for |
|---|---|---|
| **Person** | role: cashier / client (with track id: client 1, client 2…) / collector; in room or gone | A1, A6, A7, D7, and every "cashier left" rule (B1, B5, B8, B9, C2) |
| **Cash** | location: client hand, cashier hand, desk, counting machine, UV detector, safe; **owner = which client** (or collection bag) | A1, A2, A3, A7, B1, D6 |
| **`id_card`** | client → cashier hand → returned; collector's ID shown to the cashier | A6, A7, D7 |
| **Power of attorney** (collector's paper) | handed to or checked by the cashier | D7 |
| **Counting machine** | cash inside / counting | A1, A2, A3, D6 |
| **UV detector** | present in the scene? cash brought to it? | A2 |
| **Laptop** | open-active vs closed/locked; cashier using it | B9, D6 |
| **Safe** | open / closed; cash, stamps or bag contents go in | B1, B5, B8, D6 |
| **Safe key** | in the lock vs taken by the cashier | B5 |
| **Collection bag** | sealed / opened; on desk / emptied | B1, D6, D7 |
| **Stamps (печати)** | on desk vs in safe | B8 |
| **Door** | open / closed | C2 |
| **Personal items** seen in videos: bag, jacket, cup/drink, plus any other non-allowed object | on desk / in use | B3 |
| Wired telephone, journal | recognised only so they are **not** flagged as B3 | — |

### Zones (fixed per camera)
- Room interior (whether a person is present)
- Desk / workplace
- Safe
- Door

### Events built from the tracks
- `id_card` handover client → cashier and back; collector shows ID and power of attorney
- Cash handover client → cashier and cashier → client; issued part separated and recounted
- Cash goes into the counting machine, the UV detector, or the safe; cash stays on the desk
- Collection bag handed over, opened, emptied into the safe
- Laptop use; laptop closed or locked
- Safe opened or closed; key removed
- Door opened or closed
- Person enters or leaves: cashier, client N, collector
- Personal item placed or used on the desk

