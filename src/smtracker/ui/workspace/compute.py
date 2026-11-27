"""The Compute configuration surface: the SQD fit's settings, and the act.

Ported from `prototype/shadcn/compute.py` (git history, commit 09da35d), with
its three generic mock controls replaced by the SQD populations tab's actual
fit parameters, traced out of `smtracker.browser`'s `ComparisonBrowser.spec`
and `_compute_for` on `main`. Everything below changes the FIT; nothing here
is read by anything that only draws an already-computed result -- see
`toolbar.py`'s module docstring for that half of the split.

* Per-condition rows (`ConditionRow`, on the **Fits** page) -- `spec().models`:
  each condition lights its own models (`Best (BIC)` / `1`..`4` / `saSPT`, one
  `Toggles` tray of pills), resolved by `resolve_n_pop`; two or more on any row
  draw every condition's fits side by side (`spec().side_by_side`). A
  2-population fit and a 3-population one are genuinely different fits, not two
  ways of drawing the same numbers -- so this is a per-condition CHECKBOX (is it
  even fit) plus its own model pills, with an Every condition line that sets
  every row (since 1 Nov 2026; a `lock_model` that disabled the rows came before
  it, see "THE MODELS ARE TICKS ON EVERY ROW" and "FORCE SAME D ACROSS").
* `weighting` -- `ComparisonSpec.fraction_weighting`: displacement/time fraction
  (step, the legacy fit and the default) vs molecule fraction (track). It now
  governs the pooled fit's D as well as the fractions.
* `shared_d` -- `ComparisonSpec.d_sharing_scope`, the setting **Force same D
  across**: Replicates (one D per condition, the simultaneous fit; the default),
  Conditions (one D per value of a folder design's factor, offered only with a
  design), Whole data set (one D for every condition) or Don't force anywhere
  (every replicate its own D) -- and, per condition row since 1 Nov 2026,
  Replicates or Don't force anywhere for that condition alone
  (`ComparisonSpec.d_sharing_by_condition`). The words are `workspace_view`'s.
* `uncertainty` -- WHAT EACH BUBBLE IS, not only its bar:
  `bubble_view.UNCERTAINTY_MODES` -- the pooled fit, the per-day mean ± SD
  (default), or the pooled fit with a movie or cell interval on single-session data.
* `ecdf` -- where along the ECDF the fit is evaluated: 2000 points (Fast, the
  default), every point (Full, sent as `0`, which the library reads as exact --
  `None` means "the default budget" there, so Full used to run the Fast fit), or a
  chosen count (Custom).
* `overlay` -- off (draw the fit Shared D picks) or both arrangements side by side.
  Its state count is the Conditions tab's, not a second control: an `Overlay
  States` pill once drew the overlay at its own count and overrode the
  per-condition choice without saying so.

Deliberately NOT ported: the mock's "Render speed" slider. It tuned how fast
the prototype's placeholder REVEAL animation played, which has no counterpart
here -- the real browser does not animate a figure being drawn.

`COMPUTE_CONTROLS` covers every control for now, because there is only one
subsection ("sqd.populations") -- the prototype's `compute_controls(view)` /
`COMPUTE_SETS` machinery for hiding controls a view does not have is not
needed yet and is left out rather than built against nothing.

THIS WAS A POPOVER, THEN A PAGE, AND BOTH WERE WRONG FOR THE SAME REASON IN
OPPOSITE DIRECTIONS. The popover was too small for a per-condition table that
grows with the project; the page fixed that by giving every row a full window
width to itself, and that was the new mistake -- a `Segmented` row is a few
pills a few pixels tall, and stacking seven of them one-per-line down a
1280 px-wide page reads as a wall of whitespace, everything far from
everything else with no reason to be. **A modal `Dialog` is the right size
for a FORM this shaped**: fixed, compact, and -- unlike the page -- able to
lay controls out in a GRID rather than one per line, so short controls sit
beside each other instead of each claiming a row a page can afford and a
dialog cannot.

`ComputeDialog` is shadcn's `Dialog` (`DialogContent: rounded-lg border
bg-background p-6 shadow-lg`) -- LITERALLY, not just styled to look like one;
see "AN IN-WINDOW OVERLAY, NOT AN OS WINDOW" below for why that distinction
turned out to matter. (`self._card`'s actual radius is `radius_xl`, not the
`radius_lg` the class name above implies -- see `ComputeDialog.__init__`;
MEASURED against a real render of the registry's own Dialog, the template's
default `rounded-lg` reads tighter than what is actually on screen there.)
`ComputePanel` is its form body, kept as a separate
class deliberately -- it carries no title, no dismiss chrome, no act, so it
can be exercised (and was, headlessly) with no dialog machinery at all.
Being an overlay rather than a page also drops the breadcrumb-extension
complexity `window.py` used to carry when this was a page
(`_pre_compute_view`, pushing/popping "Compute Configuration" onto the
trail): an overlay sits on top of whatever page is already on screen rather
than navigating to a new one, so there is nothing to return FROM and nothing
for `Cancel` to restore.

TABS, NOT PICKERS, FOR THE PER-CONDITION AND LOCK CHOICES -- the reason a
popover could not have them, and still true of a dialog. `Segmented`'s width
is the sum of its pills, and a picker was the fix for that when width was the
scarce thing; in a dialog wide enough for the conditions table it is not, and
a picker COSTS something a dialog does not need to pay: opening one to see
what is even selected.

**Lock model is TWO ROWS of tabs sharing ONE seven-way value**, because seven
pills in one row would recreate the exact width problem tabs exist to avoid,
and because the split is a real one: `Per-Condition` / `Best` / `saSPT` are
ways of not naming a count directly, `1`..`4` are the count itself. Only one
pill across BOTH rows is ever lit -- `Segmented.select(-1)` deselects the
OTHER row the moment either one is picked, so the pair behaves as one control
rather than two that can quietly disagree.

**THE GRID GROUPS BY LABEL WIDTH, NOT BY TOPIC.** `Shared D` and `Overlay`
carry this panel's longest pill labels (`Within Condition` / `All
Conditions`; `Independent` / `Simultaneous`), so each gets its own COLUMN
rather than fighting the other four controls for space. Every remaining
control -- `Weighting`, `Uncertainty`, `ECDF`, `Overlay States` -- is a short
2-pill toggle, so `Weighting`+`Uncertainty` share `Shared D`'s column and
`ECDF`+`Overlay States` share `Overlay`'s -- see `_build_wide_grid` for why
that is a real `QGridLayout` and not two rows that happen to look aligned.
`Lock Model`'s two rows stay full-width below the grid always, since a
seven-way control does not fit a quarter column at any dialog width.

**THE WIDE ARRANGEMENT IS A GRID, SO `UNCERTAINTY` AND `OVERLAY STATES` LAND
ON THEIR COLUMN'S RIGHT EDGE AS A PROPERTY OF SHARING ITS WIDTH, NOT A
PIXEL-MATCHED GUESS.** Requested directly, after the first version -- two
independent rows, `Shared D`+`Overlay` in one, all four short controls in
another -- was seen not to line up underneath it at all: nothing tied
`Weighting`+`Uncertainty`'s combined width to `Shared D`'s. `_build_wide_grid`
puts `Shared D` and `Overlay` in row 0 of a `QGridLayout`, and
`Weighting`+`Uncertainty` (a `_grid_row` in its own right) and `ECDF`+
`Overlay States` (likewise) in row 1, one column each. A `QGridLayout` gives
every cell in a column the SAME width -- the widest cell decides it, so
column 0 is `Shared D`'s own width (343px, wider than `Weighting`+
`Uncertainty`'s ~306px) and column 1 is `Overlay`'s (295px, wider than
`ECDF`+`Overlay States`' ~207px) -- so `Uncertainty`'s right edge sits on
`Shared D`'s and `Overlay States`' sits on `Overlay`'s, automatically,
wherever those two columns end up being. Fixing this by hand (padding
`Weighting`+`Uncertainty` to a specific pixel width) would have been a
number to keep re-measuring every time a label changed; a shared grid column
cannot drift out of sync with what it is sharing.

**THE DIALOG'S WIDTH FOLLOWS THE CONDITIONS TABLE, NOT A FIXED NUMBER --
AND THE WIDE GRID IS NOT ALWAYS WHAT GETS DRAWN.** The conditions table
was the thing that first exposed the old fixed-680px dialog as wrong: two
short condition names (`Glucose`, `Malate`) left `ConditionRow`'s model tabs
stranded far from their own name, because the row's trailing
`addStretch(1)` pushed them to an edge the conditions table itself never
asked to reach. Two fixes, and they are linked:

* `ConditionRow` no longer stretches BETWEEN the name and its pills -- the
  stretch moved to AFTER the pills, and every row's name column is widened
  to the SAME fixed width (the longest loaded condition name, measured in
  `set_conditions`), so the pills still line up across conditions the way
  they did before, just without the gap sitting inside one control's own
  row.
* `ComputeDialog` carries no `setMinimumWidth` floor any more -- its natural
  width is whatever its content's `sizeHint` asks for, and the widest
  condition row usually IS that content, so a short-named project like this
  one gets a genuinely narrower dialog rather than a wide one with empty
  margin.

That second change means `Shared D` + `Overlay` side by side (measured:
343 + 295 + 20px gap = 658px) will not always fit beside a conditions table
that only needs ~360-400px for two or three ordinary names -- MEASURED, not
guessed, with a throwaway `Segmented` built purely to ask its own
`sizeHint` (`_measure_pill_width`). `_wide_enough` compares the two;
`_rebuild_grid` builds `_build_wide_grid`'s aligned `QGridLayout` when the
conditions table clears the bar, or `_build_narrow_rows`'s stacked-and-nested
one -- `Shared D` and `Overlay` each on their own row, `Weighting`+`ECDF` and
`Uncertainty`+`Overlay States` each sharing a column of their own small
2x2 grid -- when it does not. Recomputed on every `set_conditions` call,
since which arrangement
fits depends on the project actually loaded, not on the one this was
designed against. That comparison is unaffected by the tab split below --
it is still asking "does the CONDITIONS content clear the bar", the grid
just now lives on a different page than the thing it is being measured
against.

**CONDITIONS AND GENERAL ARE TWO TABS, NOT ONE SCROLLING COLUMN -- because a
single `QScrollArea` around everything turned out to be broken, not just
unwanted.** Wrapping the whole form in one scroll area (the original fix for
"the conditions table can grow with the project") produced a dialog with
BOTH a vertical and a horizontal scrollbar for a two-condition project,
which is a genuine Qt sizing bug, not a scale problem: `QAbstractScrollArea`
does not propagate its child's real height into its own `sizeHint` (measured
on this exact panel -- the form needed 500px, the scroll area reported
360px as ITS `sizeHint`, regardless), so the dialog opened ~140px short, a
vertical scrollbar appeared to cover the shortfall, and that scrollbar then
narrowed the viewport enough to trigger a horizontal one too. So this was
never really about condition COUNT -- it broke at two conditions and six
fixed controls.

The fix has two parts:

* `Conditions` and `General` split onto separate pages of a `QStackedWidget`
  (`self._pages`), switched by `_LineTabs` -- shadcn's `Tabs`,
  `variant="line"` (read off the real registry, not approximated -- see
  `_LineTab`). Line rather than the pill `Segmented` used for every VALUE in
  this dialog: these two are PAGES, and using the same pill language for
  "which page" and "which value" would blur a distinction worth keeping.
* **The height is capped from `General`, not guessed and not left to
  `QScrollArea.sizeHint()`.** `General`'s content -- the settings grid +
  `Lock Model` + the note -- is BOUNDED: always exactly that many controls,
  never more, regardless of how large the project is. `Conditions` is the
  one UNBOUNDED part -- it grows by one row per condition, measured at
  ~42px each. `set_conditions` measures `General`'s own `sizeHint` height
  (directly, a plain `QWidget`'s `sizeHint` is not the broken quantity here)
  and fixes `self._pages` to it, so switching tabs never resizes the dialog,
  `General` never needs a scrollbar at any project size, and `Conditions`
  only grows a scrollbar once a project's condition count exceeds what that
  budget covers -- MEASURED at ~284px / ~42px per row, i.e. roughly 6-7
  conditions before it bites, comfortably above the "rarely more than 8" this
  was sized against. A scrolling LIST reads as ordinary; a scrolling SETTINGS
  FORM (the original bug) does not.
* `Conditions`' own scroll area disables its HORIZONTAL scrollbar outright
  (`Qt.ScrollBarAlwaysOff`) -- the guard here is explicitly a VERTICAL one,
  never both -- and `_cond_lay` reserves a right margin equal to the
  platform's own scrollbar width (`QStyle.PM_ScrollBarExtent`) so that
  reservation is baked into the CONTENT's own `sizeHint` rather than
  requiring the dialog to guess an allowance. Without it, the vertical
  scrollbar appearing (many conditions) would narrow the viewport below the
  row's fixed minimum width and, with no horizontal scrollbar to fall back
  on, silently CLIP the rightmost pills instead of merely scrolling past
  them -- worse than the bug being fixed.

**IT IS AN IN-WINDOW OVERLAY, NOT AN OS WINDOW -- see `widgets.dialogs.
OverlayDialog`, the shared base this and `dialogs.RegroupDialog` both
build on.** `ComputeDialog` used to be a `QDialog`; the blurred
backdrop, the disable-not-just-cover blocking, the fade+zoom open/close
animation and the Escape-does-nothing rule all live there now, tuned
once for both dialogs rather than per copy. `content=central` (passed
at construction in `workspace/window.py`) is the widget this overlay
blurs and disables -- explicit, not read off `QMainWindow.
centralWidget()` the way an earlier revision did, since that API is not
available to `OverlayDialog`'s other caller.


**SWITCHING TABS SLIDES THE HEIGHT, NOT JUST THE PAGE.** Two conditions only
need ~150px of `Conditions`' own content, but `self._pages` used to sit
fixed at `General`'s height regardless (~377px measured), leaving a stretch
of empty dialog between the two condition rows and the footer's `Cancel` /
`Run` -- reported directly, after the height cap above was built. Requested
as: animate `self._pages`' height to whichever tab's own target it is
switching TO, at the sidebar's own collapse speed (`slide_ms`, `Sidebar
.set_collapsed`, `QVariantAnimation` + `QEasingCurve.Linear` -- copied
exactly, including its own signal-blocking guard against `stop()` leaving
`currentTime` at the end of the previous run, which without the guard makes
`setEndValue` recompute and EMIT at that stale time on every single switch;
measured on the sidebar at 79 of 80 cycles).

`General`'s target is always its own natural height, unchanged -- it is the
cap `Conditions`' target is measured against. `Conditions`' target is
`min(its own natural content height, General's)`: shorter when a project's
condition count fits under the cap (something to actually animate towards),
equal to it once the count needs a scrollbar. That second case is not a
branch anywhere -- `_slide_pages` already treats `target == now` as a no-op,
and once both tabs share the identical height number, switching between
them always hits exactly that case. The one thing this could not reach for
free: `ComputeDialog`'s card recentres itself as `self._pages` grows or
shrinks with no extra code, since the card was already sized by nothing
more than its own layout's sizeHint, which includes `self._pages`' current
height.

`Conditions`' "own natural content height" stopped meaning just the rows
once `Lock Model` moved there -- see "LOCK MODEL MOVED TO CONDITIONS"
below for what it means now (fixed chrome around a stretched scroll
area) and what that cost `General`'s own cap.

**EVERY CONTROL IN GENERAL LOST ITS HOVER TOOLTIPS TO A CLICK-TRIGGERED HELP
POPOVER -- THE SIX `COMPUTE_CONTROLS` ENTRIES, THEN `LOCK MODEL`, THEN
`MODELS` ON CONDITIONS.** Requested directly: a pill's hover tooltip
"blooms" the instant the pointer crosses it, which overwhelms a user who
already knows what `Shared D` does and is just trying to click a pill.
`_control` (the sole builder behind `Shared D`, `Overlay`, `Weighting`,
`Uncertainty`, `ECDF` and `Overlay States`) builds its `Segmented` with no
`tips=` at all, and `_control_head` -- shared with `Lock Model` and
`Models` (below) -- puts a small muted `_HelpButton` beside every
section's own label; clicking it opens a `Panel`
(`ComputePanel._open_help_popover`) holding a lead sentence
(`CONTROL_HELP[key]`, or `LOCK_HELP` for Lock Model) above each option's
own `Label: description` line -- the SAME text the tooltips used to
carry, read once rather than written twice. `Lock Model`'s popover lists
BOTH its rows together (`LOCK_ROW_1 + LOCK_ROW_2`), since the two act as
one seven-way value (see "Lock model is TWO ROWS of tabs" above).
`ConditionRow`'s per-condition model tabs got the same treatment one
follow-up later: `ComputePanel._show_conditions_help` explains the
checkbox, then `Best` / `1-4` / `saSPT` by name, reading `MODEL_CHOICES`
the same way `_show_lock_help` reads `LOCK_ROW_2` -- and `ConditionRow.
model`'s `Segmented` carries no `tips=` either, so that popover is the
only place those three descriptions live. A second info icon on
`_LineTab` itself (one per tab) was considered and dropped: the tabs
already carry a differentiating icon whose job is telling the two pages
apart at a glance, and a second icon with a different job (explaining,
not identifying) would blur that.

**THE HELP BUTTON'S OWN POSITION MOVED THREE TIMES BEFORE LANDING WHERE
IT SITS NOW -- SUPERSEDED BY THE `MODELS` SECTION BELOW, RECORDED HERE
FOR THE LESSONS RATHER THAN THE DESIGN.** In order: (1) a header inside
the Conditions page, width-pinned to the tabs' own width and left-aligned
-- numerically exact, still read as unbalanced, since a whole row of
empty space between two things that merely share a WIDTH does not read
as ATTACHED; (2) flush against `General`, on the tabs' own row -- fixed
that, broke a different way, reported as "afloat next to Conditions and
General" (an icon explaining page CONTENT has no relationship to the TAB
LABELS it sat beside); (3) right-aligned on that same row, landing above
`Run` instead. All three are gone now that `Models` (below) gives the
button an actual label to sit beside, which is what the whole hunt was
missing. Two REUSABLE lessons survived the churn: a `QWidget`'s default
`Preferred` size policy still lets a `QBoxLayout` grow it to soak up
unclaimed space with NO stretch factor telling it to -- disproven wrong
here with a two-line isolated reproduction, and only a stretch ITEM (not
merely "the other widget has nowhere to grow") reliably claims surplus
space; and `_LineTab`'s own `(8, 4, 8, 4)` content margins indent the
FIRST tab's icon past everything else sharing `card_lay`'s margin unless
zeroed on that one tab specifically (`_LineTabs.__init__`, top/right/
bottom left untouched) -- still true and still fixed, independent of
where the Help button ended up.

**LOCK MODEL MOVED TO CONDITIONS, BENEATH A NEW `MODELS` SECTION IT NOW
SHARES A PAGE WITH.** Requested directly, and for a reason sharper than
tidiness: the confusion this file's own dimming fix (`Segmented.
changeEvent` / `_TabInk.set_rest`, above) had to solve existed BECAUSE
`Lock Model` and the rows it disables lived on different tabs -- a
control clicked on `General` silently disabling something on a page not
even on screen is what made "I can't select anything in Conditions any
more" read as a bug instead of a cause and effect. Putting `Lock Model`
directly beneath the rows it locks makes that relationship visible
without needing a dimmed pill to explain itself at all. It carries its
own `_control_head` unchanged (`_show_lock_help`, same seven-option
popover) -- only its PARENT changed, from `general_lay` to `cond_page_
lay`, and it is stored as `self._lock_row` (was a bare local) so `set_
conditions` can measure its height as part of Conditions' own chrome.
`General` is left with just the settings grid + the stale-settings note,
so its own height budget (`_general_h`) is now genuinely SMALLER than it
used to be -- see below for what that costs.

**A NEW `MODELS` LABEL NAMES THE ROW LIST, SO IT IS NOT CONFUSED WITH
`LOCK MODEL` NOW THAT BOTH SHARE A PAGE.** No repeated "Conditions"
heading (the tab above still says that) -- "Models" is the word this
section actually adds, and it carries the SAME `_control_head` as every
other section here, so the Help button that used to float on the tabs'
own row (see above) now sits where every other Help button in this
dialog does: beside the label of the thing it explains. `self._models_
head` is kept (a bare `QHBoxLayout`, not a widget) so `set_conditions`
can read its `sizeHint()` the same way it reads `self._lock_row`'s.

**ONLY `MODELS`' OWN ROW LIST SCROLLS -- `LOCK MODEL` STAYS PINNED,
NEVER SCROLLED PAST.** Requested directly, and it is a real layout
change, not a paint trick: `cond_page_lay` (`self._cond_page`'s own
`QVBoxLayout`) gives `self._models_head` and `self._lock_row` NO stretch
(their own sizeHint height, always, whatever else happens) and gives
`self._cond_scroll` stretch `1` -- the ONLY item that can be squeezed
below its natural content height. `set_conditions` computes `chrome_h`
(both fixed sections' heights plus the two gaps between them) once, and
the rows' own budget is `self._conditions_h - chrome_h`, not `self.
_conditions_h` itself as it was when `_cond_scroll` was the whole page --
`needs_scrollbar` (which still drives `_cond_lay`'s conditional
`_scrollbar_w` margin, see "CONDITIONS AND GENERAL ARE TWO TABS" above
for why that margin exists and must stay conditional) is read off THAT
comparison now, never off `_general_h` directly.

**AND `QScrollArea.sizeHint()` WENT BACK TO BEING BROKEN THE MOMENT
`_cond_scroll` STOPPED BEING `self._pages`' DIRECT CHILD -- MEASURED
AGAIN, NOT ASSUMED FIXED.** An earlier version of this file reported
that quirk closed, on the strength of `_cond_scroll.sizeHint()`
correctly mirroring `_cond_host` once it was `self._pages`' own child.
Wrapping it back inside `self._cond_page` (needed so `Models` and `Lock
Model` can sit above and below it) brought the SAME `(14, 0)` non-answer
straight back, confirming the earlier read was about nesting DEPTH, not
about something that had been permanently fixed. Consequence:
`_cond_page.sizeHint()` was driven by `self._lock_row`'s width instead
of the (usually wider) rows', silently narrowing the card whenever the
rows needed more room than `Lock Model` did -- the same clipped-`saSPT`
shape this file has already recorded once. Fixed the same way `_HelpButton.
SIZE` was: `self._cond_page.setMinimumWidth(...)`, taking the MAX of the
rows', `self._lock_row`'s and `self._models_head`'s own sizeHints -- a
declared minimum is honoured by the layout stacking the widget even when
its own `sizeHint()` lies about needing less.

**A LOCKED ROW NOW SHOWS THE SHARED VALUE, AND UNLOCKING RESTORES EACH
ROW'S OWN PICK -- NOT JUST "DISABLED, WHATEVER IT LAST DISPLAYED".**
Requested directly, once `Lock Model` sat close enough to `Models` that
the two visibly disagreeing became obvious: a row locked to `saSPT`
still showing `Best` (dimmed) does not describe what is actually being
used, and it is the ONE thing this file's own dimming fix (`Segmented.
changeEvent`, above) could not by itself fix -- dimming says "you cannot
change this", not "here is what it currently is". `ComputePanel.
_set_lock` now pushes the lock's value onto every row's own `Segmented`
(`r.model.select(idx)`) whenever `now_locked`, and restores each row's
own remembered pick (`self._own_model`, `{condition key: its last value
while UNLOCKED}`) the moment the lock returns to `Per-Condition`.

**THE REMEMBER STEP RUNS ONLY AT THE per_condition -> LOCKED EDGE, NEVER
BETWEEN TWO LOCKED VALUES.** Switching `Lock Model` from `Best` straight
to `saSPT` (both locked) must not capture `Best` as anyone's "own" pick
-- it was never a per-condition choice, it was the PREVIOUS lock's own
reflection. `_set_lock` computes `was_locked` / `now_locked` from `self.
_lock_value` before overwriting it, and only writes `self._own_model`
when `now_locked and not was_locked`; a `Best -> saSPT` transition (both
locked) skips that write entirely, and later returning to `Per-Condition`
still restores whatever was genuinely picked before the FIRST lock, not
whatever the lock displayed most recently. A `set_conditions` reload
while already locked builds each fresh `ConditionRow` with the LOCKED
value from the start (`model=self._lock_value if locked else
DEFAULT_MODEL`), for the same reason a stale `Best` at construction would
have been -- a project loaded under `Lock Model = 3` must not show a
disabled row highlighting `Best`.

**PUSHING A VALUE VIA `.select()` ON A DISABLED `Segmented` SILENTLY
UN-DIMMED IT -- `select()`'S OWN REBUILD NEVER CONSULTED `isEnabled()`.**
`Segmented.select()` rebuilds every pill from scratch (`_pill`, chosen
over an in-place restyle because active/inactive differ in fill, shadow
AND ink at once -- see its own docstring), and `_pill` computed colours
from ONLY `active`, never from `self.isEnabled()`. So `r.model.select
(idx)`, called here to push the lock's value onto a row already disabled
via `set_locked`, rebuilt that row's pills at FULL STRENGTH regardless --
correct fill/position, wrong colour, the exact class of bug `Segmented.
changeEvent` exists to prevent, reached through a DIFFERENT door.
MEASURED, not assumed: printed the rebuilt active pill's label
stylesheet immediately after a locked `.select()` and got full-strength
`#09090B`, not the dimmed `#84848…` `_restyle_enabled` produces. Fixed at
the ROOT rather than by calling `_restyle_enabled()` after every
`.select()`: `_pill_colors(active)` is now the ONE place both `_pill`
(fresh build) and `_restyle_enabled` (in-place retint) compute fill and
ink, folding the `isEnabled()` check into it once so the two paths
cannot drift apart the way they just did.

**EVERY POPOVER'S LABELS ARE PINNED TO A FIXED WIDTH
(`ComputePanel._HELP_CONTENT_W`), NOT LEFT TO WRAP AT THEIR OWN SIZEHINT.**
Reported directly: `Overlay States`' popover (two short one-line options)
showed visibly larger gaps between its lines than `Shared D`'s did, from the
identical `spacing(8)` VBox. MEASURED: a word-wrapped `QLabel`'s
`sizeHint()` is computed at whatever width it happens to have when asked,
which for a short option is its OWN narrow natural width, not the column's
eventual shared width -- `<b>2:</b> Overlay drawn at 2 populations.`
reported a two-line `sizeHint()` at a width the layout went on to render as
one line, leaving the layout-allocated second line's height as a visible
gap. Worse the SHORTER the text, which is why the shortest control showed
it worst. See `_open_help_popover`'s own docstring for the full measurement.

**LOCK MODEL BECAME TICKS (30 Oct 2026).** Asked: "i run say 1 and 2 populations, and then i go to condition tab and
only can choose one ... we should move all of this on the 2nd page and just tick boxes of what to fit and show". The
two Lock Model rows above (`LOCK_ROW_1` / `LOCK_ROW_2`, Per-Condition / Best / saSPT and 1-4) and the Overlay's
"Counts" (28-29 Oct, General page) are ONE row of ticks now, "Fit every condition at", one per `MODEL_CHOICES` entry:
nothing ticked = each row's own model, one = that model locked (the rows show it, disabled, as the lock did), two or
more = `SEVERAL`: every condition fitted with each and drawn side by side (`ComparisonSpec.side_by_side`), the rows
disabled and highlighting none. `values()` carries the ticks as ``fit_models`` and the one locked model, if any, as
``lock_model``; `apply_values` reads an older staging (a lock, or "Counts") through `workspace_view.ticked_models`.
What the paragraphs above say of the two rows' pills, their popover and their dimming is history; the remember step
(`_own_model`, the per_condition -> locked edge) is unchanged.

**EACH ROW TICKS THE FITS IT SHOWS (31 Oct 2026).** Asked: "sqd bubble plot real data 1 and 2 pop fit, simulation
only 1 pop fit ... choose for every protein / condition how many and which fits are shown and not for whole data
set". Under `SEVERAL` a row no longer sits disabled highlighting none: it trades its pills for one tick per model
ticked (`ConditionRow.set_fit_choices`, `fits_host`), and what it UNTICKS is remembered (`_hidden_fits`), so a model
ticked later shows on every row until a row unticks it. A row keeps at least one. `values()` carries the rows
that show fewer as ``shown_fits``; `workspace_view.shown_fits` keeps them to the models ticked and
`ComparisonSpec.side_by_side_shown` carries them to `bubble_view`.

**EACH ROW HAS ITS OWN SHARED D (1 Nov 2026).** Asked, the design agreed beforehand: a condition row carries, after
its model pills, its own Shared D (`ConditionRow.shared_d`, a `_SharedDChoice`: Within Condition / Not shared, the
two `ROW_SHARED_D_CHOICES` read off `COMPUTE_CONTROLS`). A row FOLLOWS the General page's Shared D until the user
picks one on it; from then on it keeps its own (`_own_shared_d`, remembered the way `_own_model` is, across a lock and
a reload). The General page's All Conditions and Within <factor> share one D across conditions -- a property of the
set, not of a condition -- so while either is picked every row trades its two pills for ONE disabled pill showing it
(`_SharedDChoice.set_wide`, as a locked row shows the locked model) and `values()["shared_d_rows"]` is empty; going
back to Within Condition or Not shared gives every row back its own. "Shared D for every condition" is the same
control in the "Models and Shared D" header, STANDING OVER THE ROWS' PILLS with a muted "Every condition" before it
(`_align_every_shared_d`; where the header's label is wider than a row's lead -- short names with fit ticks in place
of the pills -- it sits just after the label instead, unnamed but for the Help popover; no hover tooltip, as on no
pill here): one click sets every row, as clicking each would, and it lights the rows' common value. It went
there and not on a line of its own under "Fit every condition at" because that line cost the pinned chrome 42 px,
MEASURED (the card 537 -> 579 px with five long-named rows); the header costs 14. The rows' pills widen the dialog:
a two-condition project's card measured 600 -> 790 px, which the wide General grid then fits. `values()` carries the
rows fitted otherwise than the General page says; `workspace_view.shared_d_rows` keeps them to the Run's conditions,
refuses one beside a set-wide Shared D or an Overlay, and `ComparisonSpec.d_sharing_by_condition` carries them to the
fit, where each condition is made by `condition_replicate_fit` at its own scope. `apply_values(own_shared_d=)` puts
the rows' own back (the theme switch, sessions); the Options screen's standards leave them alone.

**THE MODELS ARE TICKS ON EVERY ROW, SHARED D SAYS WHAT IT IS SHARED ACROSS, AND THREE LOOKS AT THE WHOLE DIALOG (1 Nov
2026, later the same day).** Asked: "within condition means replicates get fitted simultaneously. maybe we should call
it like that ... but where is everything fit together simultaneously (whole data set?)"; "the clicking of populations
fit make no sense. just show the check boxes from beginning"; "and there is saspt but not bic best. pls take 3 looks at
the whole screen i think its a bit messy". Four changes, the values a Run reads unchanged:

* SHARED D'S WORDS, the experimenter's set: Not shared / **Across replicates** (was Within Condition) / **Across <factor>**
  (was Within <factor>, `workspace_view.group_label`) / **Across all conditions** (was All Conditions), LEAST SHARED TO
  MOST, on both pages, in the Help popovers and everywhere a figure says it (`bubble_view.scope_words`; the sample line
  "· D across replicates" / "· D not shared"). The values ("condition", "replicate", "group:<factor>", "global") stay.
* THE MODELS ARE TICKS FROM THE BEGINNING (`ConditionRow.ticks`, a `_ModelTicks`): every row ticks the models it is
  fitted at, one at least while it can (`keep_one`). Its pills, its disabled state under a lock, the fit ticks that
  replaced the pills under two or more models (31 Oct) and "Fit every condition at" with its lock (30 Oct) are gone,
  and with them `SEVERAL`, `_set_lock`, `_own_model`'s remembering and `_hidden_fits`. The Run reads the encoding it
  has read since 30-31 Oct (`values()`): every row ticking one = each condition its own model (``fit_models`` [], the
  model in `condition_values`); any included row ticking two = every condition fitted at the UNION
  (`side_by_side`, ``fit_models``) and each row drawing its own (``shown_fits``). An older staging's lock or "Counts"
  is put on every row (`apply_values`, through `workspace_view.ticked_models`).
* **"BEST (BIC)"**: saSPT names the method that picks its count and "Best" named none, so the BIC choice read as
  missing. It was always there, as the rows' first, raised pill.
* A TABLE, AND THE THREE LOOKS (the dialog rendered shown, offscreen, in a subprocess, at a 1366 x 768 laptop at
  125 % and at 1920 x 1080). Look 1 (the ticks beside a repeated name on every row): "Best (BIC) 1 2 3 4 saSPT" written
  five times over, and the box columns uneven -- 105 px from Best's box to 1's, 45 between the others. So the names are
  written ONCE, centred over their columns (`_ModelNames`, one width per model from `_model_columns`, which every
  `_ModelTicks` uses too, so the boxes stand under their names by construction); "Models" and its Help button end the
  lead of that line, "Shared D" and its Help button stand over the pills. Under them the **Every condition** line --
  the same ticks and pills as a row, setting every row (`_set_every_model`: a tick ticks the model on every included
  row, unticking takes it off every row that keeps another; it shows ticked where every included row has the model;
  the pills as the header's did) -- and a hairline, then the rows. It replaced both "Fit every condition at" (under
  the rows, its ticks under nothing they set) and the header's "Every condition" pills (beside a label that hid when
  the gap was short); `_align_every_shared_d`'s measured offset is gone, since every line of the table is built from
  `ConditionRow`'s gaps with one lead width (`_lay_columns`). Look 2, the General page: three lines of pairs, each
  stretched to its column, put Time lag's left edge 230 px right of Uncertainty's and left a hole under ECDF; now two
  columns of single controls on one left edge each (`_WIDE_LEFT`, `_WIDE_RIGHT`), ECDF last on its line so its Custom
  field opens into room already held. The narrow arrangement is retired (`_build_wide_grid`: measured 563 px tall
  against 419). Look 3: the title, the one Title Case line, is "Configure SQD" with the siblings' "Nothing runs until
  you press Run." The card: 740 x 529 px with five long-named conditions (was 790 x 551), inside the ~550 px a
  maximised laptop window has.

**FORCE SAME D ACROSS, PILLS FOR THE MODELS, AND THE FITS PAGE (1 Nov 2026, the evening).** Three rulings, the values
a Run reads unchanged:

* A NAMING CONVENTION, the experimenter's ("yes we need a good naming convention. Force same D across / Replicates
  (default) | Conditions | Whole data set | Don't force anywhere"): the setting is **Force same D across**
  (`workspace_view.SHARED_D_NAME`, was Shared D), its pills **Replicates** (was Across replicates, "condition") ·
  **Conditions** ("group:<factor>", was Across <factor>) · **Whole data set** (was Across all conditions, "global") ·
  **Don't force anywhere** (was Not shared, "replicate"), in that order, Replicates the default. Without a folder
  design Conditions would be the Whole data set's fit, so it is not offered (three pills); with one factor it reads
  "Conditions" and its Help line names the factor ("One D per protein"); with two or more there is one pill per
  factor, "Conditions, per protein" / "Conditions, per medium" (`workspace_view.group_label`). A row offers the two a
  single condition can have, Replicates and Don't force anywhere (`ROW_SHARED_D_CHOICES`, read off the table in its
  order). Every word is spelled ONCE, in `workspace_view` (`SHARED_D_PILLS`, `scope_words` for a sentence: "same D
  across replicates", "same D across conditions, per protein", "same D across the whole data set", "D not forced:
  each replicate its own"; `SCOPE_SAMPLE_WORDS` for the sample line, "· same D across replicates" / "· D free per
  replicate"; `ARRANGEMENT_NAMES` for the Overlay's two series, "Same D across replicates" / "Same D across the whole
  data set", which were "Independent" / "Simultaneous" and clashed once Replicates named the simultaneous fit).
* THE MODELS ARE PILLS, NOT CHECKBOXES (asked: "condition picker needs to be really good ... i think clickable fields
  instead of checkable boxes are better like we have in JD histograms"): each row, and the Every condition line, is
  ONE `Toggles` tray (`_ModelPills`), the control the Jump distances page's Show uses, each pill named by its model,
  so the line of model names above the columns is gone and every row's tray stands in one column by construction
  (the same labels, the same width). `Toggles` is controlled: a click says which pill, and the rule is headless --
  `workspace_view.toggle_model` (a click toggles the model, the row's last lit model stays lit) and
  `toggle_every_model` (on the Every condition line a lit pill, every included row having it, comes off every row
  but never empties one; an unlit pill goes on every included row; `every_models` lights what every included row
  has). The column titles are "Models" over the trays and "Force same D across" over the rows' pills, each with its
  Help button.
* THE TAB "Conditions" IS **Fits** (General · Fits; tab key "fits"): what is chosen there is the fits of each
  condition, its models and whether its D is forced across its replicates. Every help text and error that pointed to
  "the Conditions page" names the Fits page.

**FORCE SAME D ACROSS ON TOP OF THE FITS PAGE, AND A SET-WIDE VALUE DISABLES THE ROWS (1 Nov 2026, the night).** Asked,
answering the offer to put the set-wide choice on the Fits page: "true. but then there should be a button on top.
clicking it, disables everything else in the fit screen." WHAT THE FIT READS, checked first: under Whole data set the
count is chosen once for every condition (`comparison._global_joint_D`: the lock if it is a number, else Best on the
pooled data, saSPT included) and a row's model never reached it -- every row at 2 with nothing locked ran Best; under
Conditions a group's count is the lock, else the number every member names, else Best on the group
(`comparison._group_count`). So:

* THE CONTROL ON TOP (`fits_shared_d`, built with the General page's by `_rebuild_grid`) IS the General page's Force
  same D across, one value in two places: a click selects the General page's pill (`_on_fits_shared_d`), whose signal
  stages it; `_sync_shared_d` moves the top with it. Title and pills on ONE line, the title before them.
* A SET-WIDE VALUE (`workspace_view.is_set_wide`): every row's model tray and its Force same D across show the set's,
  disabled -- the existing disabled pill look (`ConditionRow.set_wide_models`, `_SharedDChoice.set_wide`) -- and the
  Every condition tray alone sets the models of the one fit (`_wide_models`; a click toggles them,
  `workspace_view.toggle_model`). Picked, the set starts from the models every included row has, else Best (BIC)
  (`workspace_view.set_wide_models`), which is what the joint fit chose before. Each row KEEPS its own models
  (`ConditionRow.own_models`, `_own_model` for the window's theme switch and sessions) and shows them again when
  Replicates or Don't force anywhere comes back. The include boxes stay live: they choose the conditions the one D is
  forced across. A line under the control says why (`wide_note`, `workspace_view.set_wide_note`: "One D for every
  condition: the conditions share one fit, so they share its models."); its Help popover adds what the Every
  condition tray sets and that Best chooses one count on the pooled data (`SET_WIDE_HELP`).
* THE RUN: `values()` is unchanged in shape; under a set-wide value every row stages the set's models, and
  `workspace_view.spec_from_values` locks the one NUMBER they name (the joint fit reads only the lock) and refuses rows
  that differ or draw fewer fits side by side, by name. The two shared scopes' cache keys now say what their count is
  chosen from (`ComparisonSpec.sharing_settings`), since a condition's own model in a key did not.
* THE HEIGHT: the line on top cost the laptop's card what it did not have (575 px on Fits, 598 with the note, against
  about 550 in a maximised 1366 x 768 window at 125 %). The card now fits the window it opens in
  (`ComputeDialog._fit_window`, `ComputePanel.set_pages_room`): the Fits page shows as many rows as fit -- all five at
  1920 x 1080, four there on Replicates, three under Whole data set (where the rows are disabled) -- and scrolls the
  rest; the gaps around the Every condition line and over the four-populations line are 4 and 12 (were 6 and 16).

NOTHING HERE COMPUTES. `ComputeDialog.run` is a bare signal; the caller
decides what pressing it means (today: nothing -- see `window.py`).
"""
from __future__ import annotations

from qtpy.QtCore import (
    QEasingCurve,
    QEvent,
    Qt,
    QVariantAnimation,
    Signal,
)
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ...workspace_view import (ARRANGEMENT_NAMES, CHUNKS_DEFAULT, CHUNKS_RANGE, ECDF_FAST_POINTS, GROUP_PREFIX,
                               PER_CONDITION, ROW_SHARED_D, SET_WIDE_HELP, SHARED_D_NAME, SHARED_D_PILLS, group_label,
                               group_tip, is_set_wide, set_wide_models, set_wide_note, shared_d_pill, ticked_models,
                               toggle_every_model, toggle_model)
from ...workspace_view import every_models as models_every_row_has
from ..theme import typography as ty
from ..widgets.controls import Button, NumberField, Segmented, Toggles
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.menus import Panel
from ..widgets.primitives import (
    Checkbox,
    HoverFill,
    Surface,
    icon,
    label,
    mix,
    slide_ms,
)
from ..widgets.primitives import discard

__all__ = ["COMPUTE_CONTROLS", "CONTROL_HELP", "DEFAULT_COMPUTE", "DEFAULT_LOCK",
          "DEFAULT_MODEL", "MODELS_HELP", "MODEL_CHOICES", "ROW_SHARED_D_CHOICES", "ROW_SHARED_D_HELP",
          "ComputeDialog", "ComputePanel", "ConditionRow"]

#: `Best (BIC)` / `1`..`4` / `saSPT` -- `(value, label, tooltip)`, shared by every row's pills, the Every condition
#: row's and the Models Help popover, so they cannot silently drift apart. "BEST (BIC)", NOT "BEST" (1 Nov 2026,
#: asked: "there is saspt but not bic best"): saSPT names the method that picks its count, and "Best" alone named
#: none, so the BIC choice read as missing beside it.
MODEL_CHOICES: tuple = (
    ("best", "Best (BIC)",
     ("The state count with the lowest BIC among 1-3 populations (1-4 if ticked "
      "below), chosen automatically. BIC recommends a count and does not settle the biology; "
      "conditions can end up with different counts.")),
    ("1", "1", "Force a single population."),
    ("2", "2", "Force 2 populations."),
    ("3", "3", "Force 3 populations."),
    ("4", "4", "Force 4 populations."),
    ("saspt", "saSPT",
     ("saSPT's count of resolved modes. It assumes one D per track, so molecules "
      "that switch states blur its spectrum and it can undercount. Check the saSPT "
      "spectrum before trusting it.")),
)
DEFAULT_MODEL = "best"

#: THE MODELS ARE TICKS ON EVERY ROW (1 Nov 2026, asked: "the clicking of populations fit make no sense. just show the
#: check boxes from beginning"): each condition row ticks the models it is fitted at -- one is that condition's own
#: model, two or more draw its fits side by side -- and the Every condition row ticks or unticks a model on every row
#: at once. `values()` keeps the encoding the Run has read since 30-31 Oct: no row with two = each row its own model
#: and ``fit_models`` empty; any row with two = every condition fitted at the UNION (``fit_models``,
#: `ComparisonSpec.side_by_side`), each row drawing its own (``shown_fits``). There is no lock any more: ``lock_model``
#: is always this, "no lock, each row decides" -- `workspace_view.PER_CONDITION`, the one definition the settings
#: rules (`workspace_view.spec_from_values`) compare against -- and an older staging's lock is put on every row.
DEFAULT_LOCK = PER_CONDITION

#: `(key, label, [(value, label, tooltip), ...])`.
#:
#: THE TEXT HERE IS WHAT THE HELP POPOVERS SHOW, so it is written as the claim a
#: reader will act on and checked against the code that runs, not against the
#: label. Five of the first versions were wrong in a way a user would notice:
#: "Fast" called a subsample of DISPLACEMENTS what is a thinning of where the fit
#: is evaluated (every displacement is always used -- the distinction CLAUDE.md's
#: "REDUCE N MEANS TWO DIFFERENT THINGS" exists to keep); "Full" and "Fast" ran the
#: SAME fit, because "Full" sent `None`, which the library reads as the default
#: budget; "Bootstrap" promised intervals nothing drew; "Overlay" described a curve
#: the SQD panel does not have; and "Overlay States" competed with the per-condition
#: model choice, and silently won.
COMPUTE_CONTROLS: tuple = (
    # FORCE SAME D ACROSS (ruled 1 Nov 2026: "yes we need a good naming convention. Force same D across / Replicates
    # (default) | Conditions | Whole data set | Don't force anywhere"): the name and the pills are `workspace_view`'s
    # (`SHARED_D_NAME`, `SHARED_D_PILLS`), in the experimenter's order. A folder design's "Conditions" pills go between
    # Replicates and Whole data set (`ComputePanel._controls_table`); without a design there is none.
    ("shared_d", SHARED_D_NAME, (
        ("condition", SHARED_D_PILLS["condition"],
         ("One D per condition, fitted on all of its replicates together; each replicate's fractions "
          "are then fitted with that D held. The default. A population under 5 % is "
          "not kept (fit one fewer): its D is poorly determined, and an error in it moves every replicate "
          "together.")),
        ("global", SHARED_D_PILLS["global"],
         ("One D for the whole data set: every selected condition is fitted together (always "
          "step-weighted) and each condition's fractions are fitted with that D held. The fractions then "
          "measure the same thing everywhere, but no condition has this D on its own.")),
        ("replicate", SHARED_D_PILLS["replicate"],
         ("Every replicate is fitted on its own, D included; each bubble is the mean over its replicates. "
          "A replicate with few tracks can give a noisy D; the replicate page shows every replicate.")))),
    ("overlay", "Overlay", (
        ("off", "Off",
         f"Draw only the fit {SHARED_D_NAME} picks. The default."),
        ("both", "Both",
         (f"Draw {ARRANGEMENT_NAMES['independent']} (each condition its own D) beside "
          f"{ARRANGEMENT_NAMES['simultaneous']} (one D for every condition) to show what a shared D "
          "costs. Always a mean over replicates; needs every row on the same single model "
          "(Fits page).")))),
    ("gradient", "Optimiser", (
        ("analytic", "Fast",
         ("The fit with the model's exact derivatives: about twofold faster at 3 populations, "
          "with the same fits as Standard. The default.")),
        ("numeric", "Standard",
         ("The fit with derivatives estimated numerically: slower, the same numbers. "
          "For checking a result against an older one.")))),
    ("lag", "Time lag", (
        ("1", "1",
         ("Displacements between consecutive frames. The default.")),
        ("2", "2",
         ("Displacements two frames apart, fitted at twice the frame interval, so D is the apparent D at that "
          "lag. A diffusing state keeps its D; a population of localisation error alone halves.")),
        ("3", "3", "Three frames apart, fitted at three frame intervals."),
        ("4", "4",
         ("Four frames apart. A track gives fewer displacements at a longer lag, and they overlap, "
          "so the spread stays over replicates or cells.")),
        ("1-4", "1–4",
         ("SMTracker 2's “1-4 together”: one fit to the displacements 1, 2, 3 and 4 frames apart, with the same D "
          "and fractions at every lag. The count is chosen at lag 1. "
          "Localisation error is not modelled, so a population at the floor reads lower than at lag 1.")))),
    # PER-SPOT PRECISION (6 Nov 2026; asked: the option in the SQD section, "yes sure"): `workspace_view.PRECISION_*`
    ("precision", "Localisation error", (
        ("off", "Not modelled",
         ("Every population has one typical step length. A bound molecule's step is its "
          "localisation error, which differs from spot to spot, so part of a large bound population can read "
          "as a slow moving one. The default.")),
        ("per_spot", "Per spot",
         ("Each step's localisation error comes from its two spots' precision, through a calibration "
          "measured on immobile spots (a file in the project folder, else the "
          "tracker's built-in one). Maximum likelihood; D is corrected for localisation error but not "
          "motion blur, and a bound population's D is not resolved. Unavailable with a D forced across "
          "conditions or with lags 1-4.")))),
    ("weighting", "Weighting", (
        ("step", "Step",
         ("Every displacement counts once, so a long track counts more: the "
          "share of observed time in each population, as in SMTracker 2. The default.")),
        ("track", "Track",
         ("Every track counts once, however long: the share of molecules in "
          "each population. Long tracks are mostly slow, so the slow "
          "fraction reads lower than with Step.")))),
    ("uncertainty", "Uncertainty", (
        ("none", "None",
         ("Every replicate is pooled into one fit and drawn with no "
          "interval.")),
        ("day", "Replicates",
         ("Fractions fitted per replicate; each bubble is their mean ± SD. "
          "The SD is a spread between replicates and not a confidence interval. The default.")),
        ("movie", "Movies",
         ("The pooled fit with a 95 % interval from resampling movies. For single-session data only. "
          "Slow.")),
        ("cell", "Cells",
         ("The pooled fit with a 95 % interval from resampling cells. For single-session data only: "
          "with two or more replicates, cells in a replicate are not independent, so the interval is "
          "withheld and the caption says why. Slow.")),
        ("chunk", "Chunks",
         ("Each condition's movies are dealt at random into chunks (5 unless you set another number; "
          "cells where a condition has fewer than two movies per chunk) and fitted as replicates. Each "
          "bubble is their mean ± SD. For data without replicates only: a condition with two or more "
          "replicates is drawn as its pooled fit. The chunks show how precisely these cells were "
          "measured and say nothing about another culture.")))),
    ("ecdf", "ECDF", (
        ("fast", "Fast",
         ("Every displacement goes into the ECDF; the fit is evaluated at 2000 points along it. "
          "Fractions change by under 0.001 percentage points. The default.")),
        ("full", "Full",
         "The fit is evaluated at every point of the ECDF. Exact, and slower."),
        ("custom", "Custom",
         ("Evaluate the fit at a number of ECDF points you choose. Below about "
          "1000 it is no faster and loses accuracy.")))),
)

#: A ROW'S OWN FORCE SAME D ACROSS (1 Nov 2026): the two `COMPUTE_CONTROLS` "shared_d" entries a single condition can
#: take (`workspace_view.ROW_SHARED_D`: Replicates, Don't force anywhere), READ OFF that table -- in its order -- so the
#: row's pills, the General page's and the Help popovers carry one label and one description. Conditions and Whole
#: data set force one D across conditions -- a property of the set -- so a row shows them, disabled, and never offers
#: them.
ROW_SHARED_D_CHOICES: tuple = tuple(
    o for o in dict((k, opts) for k, _n, opts in COMPUTE_CONTROLS)["shared_d"] if o[0] in ROW_SHARED_D)

#: The ECDF point budget "Fast" uses is `ECDF_FAST_POINTS`, defined once in
#: `workspace_view` (where the settings rules read it) and imported above. Also the
#: value a Custom field opens on.
ECDF_CUSTOM_RANGE = (100, 200_000)

#: The General page's arrangement is `ComputePanel._WIDE_LEFT` / `_WIDE_RIGHT` (`_build_wide_grid`), the one
#: arrangement since 1 Nov 2026.

#: STEP, PER DAY, FAST, OVERLAY OFF -- ruled by the experimenter: step because it is
#: the legacy fit, per day because the day is the replicate and it is the legacy
#: summary, fast because the measured cost of thinning the curve is under 0.001 pp,
#: and overlay off so what is drawn is what Shared D says. THE OPTIMISER FAST, with
#: Standard one click away (ruled 16 Oct 2026: "fast default, but make the before mode
#: clickable when setting up the fit"). FORCE SAME D ACROSS REPLICATES (called Within Condition until 1 Nov 2026, then
#: Across replicates until the evening), the simultaneous fit of a condition's days (ruled 17 Oct 2026, after a morning
#: on Per Day: the SynR power run showed the shared D too confident only with a population of a few per cent, and a
#: population under 5 % is not kept -- one fewer is fitted; `docs/synthetic_synr.md`). "Don't force anywhere" (every
#: day of every condition fitted on its own; called "Per Day" until 17 Oct 2026: "We should call this differently",
#: then Not shared) one click away.
DEFAULT_COMPUTE: dict = {"weighting": "step", "shared_d": "condition",
                         "uncertainty": "day", "ecdf": "fast",
                         "overlay": "off", "gradient": "analytic", "lag": "1", "precision": "off"}

#: One lead sentence per General-tab control, shown above its own options
#: in the click-triggered Help popover (`_HelpButton` / `ComputePanel.
#: _show_help`) -- these REPLACED the per-pill hover tooltips `COMPUTE_
#: CONTROLS` already carries (requested directly: the hover tooltips
#: "bloom" over every pill and overwhelm a user who already knows the
#: controls). Kept as its own small dict rather than a fourth element on
#: `COMPUTE_CONTROLS`'s tuples: that tuple is unpacked as `(key, name,
#: options)` at half a dozen call sites (`_rebuild_grid`, `values()`,
#: `set_stale()`, `_measure_pill_width`), and widening its shape would
#: touch every one of them for a value only the Help popover reads.
CONTROL_HELP: dict = {
    "shared_d": ("What one diffusion coefficient (D) is forced across. It is fitted on all of that data together, "
                 "and each fraction is then fitted with it held. Each condition can choose its own, "
                 f"{SHARED_D_PILLS['condition']} or {SHARED_D_PILLS['replicate']}, on the Fits page."),
    "overlay": (f"Draws the other arrangement beside the one {SHARED_D_NAME} picks (Both). "
                "To fit several models side by side, pick them on each condition's row on the Fits page."),
    "gradient": ("How the fit finds its optimum: Fast (exact derivatives, the default) or "
                 "Standard (derivatives estimated numerically). Both give the same fits."),
    "lag": ("How many frames apart the displacements are (SMTracker 2's Time lag). At a longer lag a diffusing state "
            "keeps its D, while a population that is only localisation error reads lower. 1 is the default; "
            "1-4 is SMTracker 2's “1-4 together”."),
    "precision": ("Whether each step's own localisation error enters the fit (Per spot), or every population has one "
                  "typical step length (Not modelled). Per spot needs the tracker's per-spot precision and a "
                  "calibration."),
    "weighting": ("What a population's fraction counts: displacements (time) or tracks (molecules). "
                  "It applies to the fitted D as well as to the fractions."),
    "uncertainty": ("What each bubble shows and what its spread means: a pooled fit, the mean of the replicates, "
                    "a pooled fit with a movie or cell interval, or the mean of random chunks (for data without "
                    "replicates). The point moves with it, not only the bar."),
    "ecdf": ("How many points along the ECDF the fit is evaluated at. Every displacement is always used."),
}

#: The Fits page's "Models" column, its Help popover's lead sentence (`ComputePanel._show_models_help`) -- same
#: idea as `CONTROL_HELP`, kept apart since the models are not one of `COMPUTE_CONTROLS`' entries. Replaced Lock
#: Model's, then "Fit every condition at"'s, on 1 Nov 2026.
MODELS_HELP = ("Which models each condition is fitted at. Click a pill on its row to light it. One lit pill is that "
               "condition's own model; two or more draw their fits side by side.")
#: The Fits page's "Force same D across" column, its Help popover's lead sentence
#: (`ComputePanel._show_row_shared_d_help`).
ROW_SHARED_D_HELP = (f"Each condition's own {SHARED_D_NAME}. A row follows the General page's until you pick one "
                     "on it.")


def _choice_index(choices, value) -> int:
    """`choices` is a `(value, label, tip)` tuple; -1 if `value` is not in it
    -- the state a `_SharedDChoice` shows when no pill is raised."""
    for i, (v, _l, _t) in enumerate(choices):
        if v == value:
            return i
    return -1


def _models_in_order(models) -> list:
    """``models`` (values of `MODEL_CHOICES`) in `MODEL_CHOICES` order, each once. Raises `ValueError` naming one that
    is not a model -- a staging that names one is refused, never trimmed."""
    wanted = [str(m) for m in models]
    known = [v for v, _l, _t in MODEL_CHOICES]
    bad = [m for m in wanted if m not in known]
    if bad:
        raise ValueError(f"{bad[0]!r} is not a model; the models are {', '.join(known)}")
    return [m for m in known if m in wanted]


class _SharedDChoice(QWidget):
    """A Force same D across one condition can have (`ROW_SHARED_D_CHOICES`, Replicates / Don't force anywhere), or --
    while the General page forces one D across conditions -- that setting itself, shown on ONE disabled pill
    (`set_wide`), the way a row shows a set-wide value rather than a stale pick of its own. Used twice: on each
    `ConditionRow` and on the Fits page's Every condition row. Single-choice `Segmented` pills, clickable fields as the
    models' are.

    `changed` is a USER's pick only. `set_value` moves the pill without it, so the panel can push the General page's
    value onto a row that follows it without the row reading as one the user changed (`ComputePanel._own_shared_d`).
    ``set_value(None)`` raises no pill: the every-row control's answer when the rows disagree."""

    changed = Signal(str)

    def __init__(self, t, value: str | None = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._quiet = False
        # No `tips=`, as on every pill in this dialog: the Fits page's Help popovers describe both values
        self.pills = Segmented(t, [lab for _v, lab, _tip in ROW_SHARED_D_CHOICES],
                               current=_choice_index(ROW_SHARED_D_CHOICES, value) if value else -1, size="sm")
        self.pills.changed.connect(self._on_pills)
        lay.addWidget(self.pills)
        #: The set-wide Force same D across (Whole data set, Conditions), disabled: added to the layout before it is
        #: hidden
        self.wide = Segmented(t, [SHARED_D_PILLS["global"]], current=0, size="sm")
        self.wide.setEnabled(False)
        lay.addWidget(self.wide)
        self.wide.hide()

    def _on_pills(self, index: int) -> None:
        if not self._quiet and index != -1:
            self.changed.emit(ROW_SHARED_D_CHOICES[index][0])

    def value(self) -> str | None:
        """The pill raised (`ROW_SHARED_D`), None when none is (or the set-wide value shows)."""
        index = self.pills.current()
        return None if index == -1 or self.is_wide() else ROW_SHARED_D_CHOICES[index][0]

    def set_value(self, value: str | None) -> None:
        """Raise ``value``'s pill (None: none), without `changed`."""
        index = -1 if value is None else _choice_index(ROW_SHARED_D_CHOICES, value)
        if value is not None and index == -1:
            raise ValueError(f"a condition's own {SHARED_D_NAME} is one of {ROW_SHARED_D}, got {value!r}")
        self._quiet = True
        try:
            self.pills.select(index)
        finally:
            self._quiet = False

    def set_wide(self, text: str | None) -> None:
        """Show the set-wide Force same D across ``text`` on the disabled pill in place of the two (None: the two
        again)."""
        if text:
            self.wide.set_labels([text])
            self.pills.hide()
            self.wide.show()
        else:
            self.wide.hide()
            self.pills.show()

    def is_wide(self) -> bool:
        return not self.wide.isHidden()


#: The row control the Apparent Diffusion and HMM Run dialogs reuse (4 Nov 2026, found by the overnight tester: their
#: rows showed a lit "Replicates" under a set-wide value, where this one shows the set-wide value itself)
SharedDChoice = _SharedDChoice

#: The models' values in `MODEL_CHOICES` order: the order every rule of `workspace_view` keeps a row's models in
MODEL_ORDER: tuple = tuple(v for v, _l, _t in MODEL_CHOICES)


class _ModelPills(Toggles):
    """THE MODELS, CLICKED (1 Nov 2026, asked: "condition picker needs to be really good ... i think clickable fields
    instead of checkable boxes are better like we have in JD histograms"): one `Toggles` tray, a pill per
    `MODEL_CHOICES` entry named by its model -- the control the Jump distances page's Show is -- on each condition row
    and on the Fits page's Every condition line. THE SAME TRAY, THE SAME LABELS, IN EVERY LINE, so every row's pills
    stand in one column by construction (the same width), with no line of names above them.

    CONTROLLED, as `Toggles` is: a click emits `model_pressed` (the model's value) and lights nothing; the owner works
    out what is lit -- `workspace_view.toggle_model` for a row, `toggle_every_model` for the Every condition line --
    and lights it with `set_models`, which is silent. No hover tooltip, as on no control of this dialog: the Models
    title's Help popover describes them."""

    model_pressed = Signal(str)

    def __init__(self, t, models=(), parent=None):
        super().__init__(t, [lab for _v, lab, _tip in MODEL_CHOICES], size="sm", parent=parent)
        self.pressed.connect(lambda index: self.model_pressed.emit(MODEL_ORDER[index]))
        self.set_models(models)

    def models(self) -> list:
        """The models lit, in `MODEL_CHOICES` order."""
        return [MODEL_ORDER[i] for i in self.on()]

    def set_models(self, models) -> None:
        """Light exactly ``models``, silently. Raises `ValueError` naming a value that is not a model."""
        self.set_on([MODEL_ORDER.index(m) for m in _models_in_order(models)])

    def pill_of(self, model: str):
        """``model``'s pill as it stands now (`Segmented` rebuilds its pills on every change), for a click."""
        return self.pill(MODEL_ORDER.index(str(model)))


class ConditionRow(QWidget):
    """One condition: is it in the fit at all, the models it is fitted at (one pill each, 1 Nov 2026), and its own
    Force same D across (1 Nov 2026).

    A CHECKBOX, not the same one `toolbar.ConditionsBody` draws -- that one
    is DISPLAY (which conditions are shown, discussed in `toolbar.py`'s
    module docstring); this one is `spec().conditions`, i.e. whether a fit
    runs for this condition at all. Keeping the port's overlap between "shown"
    and "fit" an open decision (per the mapping this was built from) means
    both surfaces get their own control for now rather than sharing one.

    THREE COLUMNS, each at a width every row shares: the lead (box and name, the name widened to the panel's longest
    by `ComputePanel.set_conditions`), the model pills (`_ModelPills`, the same width in every row) and the Force same
    D across pills -- `LEAD_GAP` and `COLUMN_GAP` apart, which the Fits page's header lines are built with too.
    """

    toggled = Signal(str, bool)
    #: ``(key, its first model)`` -- kept for callers of the pills' signal; `models_changed` says all of them
    model_changed = Signal(str, str)
    #: ``(key, the models this row lights now)`` -- a user's click (never `set_models`)
    models_changed = Signal(str, list)
    #: ``(key, "condition" | "replicate")`` -- the user picked this row's own Force same D across (never a value
    #: pushed onto it).
    shared_d_changed = Signal(str, str)

    #: The include box to the name
    BOX_GAP = 8
    #: The lead (box and name) to the model pills
    LEAD_GAP = 16
    #: The model pills to the Force same D across pills: a second choice, not a seventh model
    COLUMN_GAP = 24

    def __init__(self, t, key: str, name: str, *, checked: bool = True,
                 models=(DEFAULT_MODEL,), shared_d: str | None = None, parent=None):
        super().__init__(parent)
        self.t, self.key = t, key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.box = Checkbox(t, checked=checked)
        self.box.toggled.connect(lambda on: self.toggled.emit(self.key, on))
        lay.addWidget(self.box)
        lay.addSpacing(self.BOX_GAP)
        #: Widened to a shared column by `ComputePanel.set_conditions` (the
        #: longest loaded condition name), so the pills line up across rows.
        #: Natural width until then, for a `ConditionRow` used on its own.
        self.title = label(name, ty.ui("sm"), t.foreground)
        lay.addWidget(self.title)
        lay.addSpacing(self.LEAD_GAP)
        #: THE MODELS, PILLS (1 Nov 2026): one per `MODEL_CHOICES` entry, at least one lit (`workspace_view.
        #: toggle_model`: a click toggles the model, the row's last lit model stays lit).
        if not models:
            raise ValueError("a condition is fitted at one model at least")
        #: THE ROW'S OWN MODELS, and -- while Force same D across forces one D across conditions -- the set's, shown
        #: in their place on the disabled tray (`set_wide_models`, 1 Nov 2026); its own come back when it lifts
        self._own = _models_in_order(models)
        self._wide: list | None = None
        self.model_pills = _ModelPills(t, models)
        self.model_pills.model_pressed.connect(self._on_model_pressed)
        lay.addWidget(self.model_pills)
        lay.addSpacing(self.COLUMN_GAP)
        #: ITS OWN FORCE SAME D ACROSS (1 Nov 2026): Replicates or Don't force anywhere, the General page's value
        #: until the user picks one here (`ComputePanel._sync_shared_d`). Every fit of a condition is made at it.
        self.shared_d = _SharedDChoice(t, shared_d or DEFAULT_COMPUTE["shared_d"])
        self.shared_d.changed.connect(lambda value: self.shared_d_changed.emit(self.key, value))
        lay.addWidget(self.shared_d)
        # The stretch sits AFTER the pills, not between the name and them --
        # a dialog that ends up wider than this row's own content (a longer
        # sibling condition, say) leaves its slack at the row's right edge
        # rather than inside one control's own row. See the module
        # docstring's "THE DIALOG'S WIDTH FOLLOWS THE CONDITIONS TABLE".
        lay.addStretch(1)

    def models(self) -> list:
        """The models this condition is fitted at, in `MODEL_CHOICES` order: its own, or under a set-wide Force same D
        across the set's (what the tray shows)."""
        return list(self._wide) if self._wide is not None else list(self._own)

    def own_models(self) -> list:
        """The row's own models, kept while a set-wide Force same D across shows the set's."""
        return list(self._own)

    def set_models(self, models) -> None:
        """Make ``models`` (one at least) the row's own, without `models_changed`; shown unless the set's are."""
        models = _models_in_order(models)
        if not models:
            raise ValueError("a condition is fitted at one model at least")
        self._own = models
        if self._wide is None:
            self.model_pills.set_models(models)

    def set_wide_models(self, models) -> None:
        """ONE D ACROSS CONDITIONS (1 Nov 2026): show the set's ``models`` on the tray, disabled -- the existing
        disabled pill look -- or, with None, the row's own again, live."""
        self._wide = _models_in_order(models) if models else None
        self.model_pills.setEnabled(self._wide is None)
        self.model_pills.set_models(self.models())

    def is_wide(self) -> bool:
        return self._wide is not None

    def _on_model_pressed(self, model: str) -> None:
        """A pill clicked: the row's models as `workspace_view.toggle_model` says (the last lit one stays lit). Never
        while the set's are shown (the tray is disabled then)."""
        if self._wide is not None:
            return
        models = toggle_model(self._own, model, MODEL_ORDER)
        if models == self._own:
            return
        self.set_models(models)
        self.model_changed.emit(self.key, models[0])
        self.models_changed.emit(self.key, models)


class _LineTab(QWidget):
    """One shadcn `Tabs`, `variant="line"` trigger: dimmed text, or
    full-strength text with a thin bar underneath. No background box at any
    state -- that absence is what marks this as PAGE navigation rather than
    the pill `Segmented` used everywhere else in this dialog for VALUES.

    Read off the real registry (`tabsListVariants` / `TabsTrigger` in
    `ui/tabs.tsx`), not approximated: inactive text is `text-foreground/60`
    (`mix(foreground, background, 0.6)` here, since Qt stylesheets do not
    composite alpha the way Tailwind's utility does), hover brightens it to
    full `text-foreground` even before a click, and the active tab's `after:`
    bar is `bg-foreground`, `h-0.5`. Simplified in one place: the registry
    fades that bar in (`opacity-0` -> `opacity-100`); here it switches
    instantly. A disclosed simplification, not an oversight.
    """

    clicked = Signal()

    #: One notch under the registry's `size-4` (16px), matching `Button`'s
    #: own "13/14px glyph beside sm text" judgement call rather than the
    #: literal spec -- see `controls.Button.__init__`.
    _GLYPH_PX = 14

    def __init__(self, t, text: str, glyph: str, parent=None):
        super().__init__(parent)
        self.t = t
        self._active = False
        self._hovered = False
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(5)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self._icon = icon(glyph, self._GLYPH_PX, t.foreground, stroke=2.0)
        top.addWidget(self._icon)
        self._label = label(text, ty.ui("sm", "medium"), t.foreground)
        top.addWidget(self._label)
        lay.addLayout(top)
        self._bar = QWidget()
        self._bar.setFixedHeight(2)
        lay.addWidget(self._bar)
        self._restyle()

    def set_active(self, on: bool) -> None:
        if on == self._active:
            return
        self._active = on
        self._restyle()

    def _restyle(self) -> None:
        t = self.t
        lit = self._active or self._hovered
        ink = t.foreground if lit else mix(t.foreground, t.background, 0.6)
        self._label.setStyleSheet(f"color:{ink}; background:transparent;")
        self._icon.colour = ink
        self._icon.update()
        self._bar.setStyleSheet(
            f"background:{t.foreground if self._active else 'transparent'}; "
            "border:none;")

    def enterEvent(self, e):
        super().enterEvent(e)
        self._hovered = True
        self._restyle()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._hovered = False
        self._restyle()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()


class _LineTabs(QWidget):
    """shadcn's `Tabs`, `variant="line"` -- the switcher between
    `ComputePanel`'s two pages. See `_LineTab` for why this is a different
    visual language than the pill `Segmented` used for every VALUE here.

    `items` is `[(key, text, glyph), ...]` -- the icon is not decoration:
    plain-text tabs are easy to skim past, and `General` in particular holds
    real settings nobody would think to go looking for a second tab to find.
    """

    changed = Signal(str)

    def __init__(self, t, items, *, current: str | None = None, parent=None):
        super().__init__(parent)
        self.t = t
        self._items = list(items)
        self._current = current if current is not None else (
            self._items[0][0] if self._items else None)
        self._tabs: dict[str, _LineTab] = {}

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        for i, (key, text, glyph) in enumerate(self._items):
            tab = _LineTab(t, text, glyph)
            if i == 0:
                # `_LineTab`'s own (8, 4, 8, 4) content margins give every
                # tab breathing room from its neighbour -- paid by the
                # FIRST tab too, which indents its icon 8px past the
                # title/subtitle/condition rows sharing this dialog's one
                # `card_lay` margin. Reported directly. Zeroed on the
                # first tab only; its top/right/bottom stay untouched, so
                # neither the click target nor the gap before the next
                # tab moves.
                m = tab.layout().contentsMargins()
                tab.layout().setContentsMargins(0, m.top(), m.right(), m.bottom())
            tab.set_active(key == self._current)
            tab.clicked.connect(lambda k=key: self.select(k))
            lay.addWidget(tab)
            self._tabs[key] = tab
        lay.addStretch(1)

    def select(self, key: str) -> None:
        if key == self._current or key not in self._tabs:
            return
        self._current = key
        for k, tab in self._tabs.items():
            tab.set_active(k == key)
        self.changed.emit(key)

    def current(self) -> str | None:
        return self._current


#: The page tabs the apparent-D and HMM Run dialogs reuse (4 Nov 2026, `run_dialog.RunDialog`: General · Fits, as here)
LineTabs = _LineTabs


class _HelpButton(Surface):
    """A small, muted `info` glyph beside a General-tab control's label --
    opens a click-triggered `Panel` explaining that control, in place of
    the per-pill hover tooltip. See `CONTROL_HELP` / `ComputePanel.
    _show_help` for why this replaced hover entirely on these six controls.

    Deliberately NOT plain black: `t.muted_foreground` at rest (the same
    ink the control's own label uses), so it reads as a quiet affordance
    beside the label rather than competing with it -- styled like
    `_CloseButton` / `SidebarTrigger` (a borderless `Surface` that only
    fills on hover), just smaller, since this sits inline with a `sm` label
    rather than in a dialog's own corner.
    """

    clicked = Signal()

    #: The one source of truth for this button's rendered size -- read back
    #: by `ComputePanel.set_conditions` to budget the Conditions page's
    #: lone header row. NOT `sizeHint()`: a `setFixedSize` widget's own
    #: `sizeHint()` is unclamped and reports whatever its INTERNAL layout
    #: would prefer (here, the 13px icon alone) -- the fixed 18px only
    #: takes effect once a PARENT layout wraps this widget in its own
    #: `QWidgetItem`, which is what a plain `.sizeHint()` call on the
    #: button itself bypasses. MEASURED, not assumed: `.sizeHint()` read
    #: back 13 while `.minimumSize()` / `.maximumSize()` / `.size()` all
    #: agreed on 18 -- the 5px gap was enough to force an unneeded vertical
    #: scrollbar onto a two-condition project, which then clipped the
    #: model tabs' own rightmost pill.
    SIZE = 18

    def __init__(self, t, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        self.setObjectName(f"helpbtn{id(self)}")
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        HoverFill(self, lambda c: (f"QWidget#{self.objectName()} "
                                   f"{{ background:{c}; border:none;"
                                   f" border-radius:{t.radius_sm:g}px; }}"),
                  t.background, t.accent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(icon("info", 13, t.muted_foreground, stroke=1.75),
                      alignment=Qt.AlignCenter)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()


class ComputePanel(QWidget):
    """The Compute Dialog's form body: `Conditions` and `General`, as two
    tabbed pages sharing one height budget.

    Purely the settings surface -- no title, no dismiss chrome, no act (see
    the module docstring for why those moved to `ComputeDialog`). Changing a
    control here only stages a value in `values()` / `condition_values()`;
    nothing runs until whoever hosts this panel decides `Run` was pressed.
    """

    changed = Signal(str, str)
    condition_toggled = Signal(str, bool)
    condition_model_changed = Signal(str, str)
    #: Emitted on every tick of the `Conditions`/`General` height slide --
    #: `ComputeDialog` connects it to its own `update()`. Not this panel's
    #: job to know why (it has no idea it lives inside a translucent,
    #: hand-painted backdrop); see `ComputeDialog.__init__`'s own connection
    #: for that.
    pages_resizing = Signal()

    def __init__(self, t, conditions=(), parent=None, *, factor_names=()):
        super().__init__(parent)
        self.t = t
        #: A FOLDER DESIGN'S FACTORS: Force same D across offers "Conditions" for one, "Conditions, per <factor>" for
        #: each of several (`_controls_table`), none without a design
        self._factor_names = tuple(factor_names or ())
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self._lay = lay

        # `flask-conical` / `sliders-horizontal` -- the SAME two glyphs the
        # toolbar rail already uses for "Conditions" and "Visualization
        # properties" (`toolbar.py`'s `_TOOLS`), not new ones invented for
        # this dialog: one glyph, one meaning, across the whole workspace.
        # Added because a plain text tab is easy to miss -- General carries
        # settings a researcher might never think to look for a second tab
        # to find. "FITS", NOT "CONDITIONS" (1 Nov 2026): what is chosen there is the fits of each condition -- its
        # models and whether its D is forced across its replicates.
        self.tabs = _LineTabs(
            t, [("general", "General", "sliders-horizontal"),
               ("fits", "Fits", "flask-conical")],
            current="general")
        self.tabs.changed.connect(self._on_tab_changed)
        lay.addWidget(self.tabs)

        # -- the Fits page (rebuilt 1 Nov 2026): a TABLE OF THREE COLUMNS -- the condition, its Models (one pill
        # each) and its Force same D across -- under a line of column titles (each with its Help button) and the Every
        # condition line, which sets every row; the rows SCROLL past `MAX_VISIBLE_ROWS`, "Best (BIC) also tries 4"
        # under them stays put. See the module docstring's "FORCE SAME D ACROSS, PILLS FOR THE MODELS". --
        self._cond_host = QWidget()
        self._cond_host.setStyleSheet("background:transparent;")
        self._cond_lay = QVBoxLayout(self._cond_host)
        # Right margin reserves the platform's own scrollbar width IN THE
        # CONTENT'S OWN sizeHint, so a vertical scrollbar appearing later
        # narrows the viewport into space already budgeted for it, rather
        # than into the row's fixed minimum width -- see the module
        # docstring's "CONDITIONS AND GENERAL ARE TWO TABS". Kept on
        # `self` (not a local) so `set_conditions` can EXCLUDE this same
        # reservation from the card's width when no scrollbar will ever
        # show -- see its own comment for why that distinction matters.
        self._scrollbar_w = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self._cond_lay.setContentsMargins(0, 0, self._scrollbar_w, 0)
        # 6 px between two rows, as the apparent-D and HMM dialogs' (`run_dialog.ROW_SPACING`; was 10: 7 of 10 rows fit
        # a 1366 x 768 window, 8 now)
        self._cond_lay.setSpacing(6)
        # Keeps rows pinned to the top when the page has more height than
        # they need (a short project under General's height budget), rather
        # than the rows spreading out to fill it.
        self._cond_lay.addStretch(1)
        self.rows: dict[str, ConditionRow] = {}

        self._cond_scroll = QScrollArea()
        self._cond_scroll.setWidgetResizable(True)
        self._cond_scroll.setFrameShape(QScrollArea.NoFrame)
        self._cond_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._cond_scroll.setStyleSheet("QScrollArea { background:transparent; }")
        self._cond_scroll.setWidget(self._cond_host)

        self.controls: dict = {}
        #: The currently open Help popover (`_show_help`), or `None` --
        #: kept so a second Help button can close whichever one is already
        #: open rather than stacking a second popover on top of it.
        self._help_popover: Panel | None = None
        #: `apply_values` puts a staging back with `_sync_shared_d` held until every part of it is in
        self._restoring = False

        # THE SET-WIDE FORCE SAME D ACROSS, ON TOP OF THE FITS PAGE (1 Nov 2026, the night; asked: "true. but then
        # there should be a button on top. clicking it, disables everything else in the fit screen"): the General
        # page's setting, ONE VALUE IN TWO PLACES -- its pills (`fits_shared_d`, built with the General page's by
        # `_rebuild_grid`) select the General page's, and `_sync_shared_d` moves them with it. ON ONE LINE, its title
        # before the pills: above them it cost the laptop's card a line it does not have. While it forces one D across
        # conditions (Conditions, Whole data set) the conditions share one fit and its models
        # (`workspace_view.is_set_wide`): every row's model tray and Force same D across pills show the set's,
        # disabled, the Every condition tray alone sets the models (`_wide_models`), and a line under it says why
        # (`wide_note`, `workspace_view.set_wide_note`). The include boxes stay live: they choose the conditions the
        # one D is forced across.
        self._wide_host = QWidget()
        self._wide_host.setStyleSheet("background:transparent;")
        wide_lay = QVBoxLayout(self._wide_host)
        wide_lay.setContentsMargins(0, 0, 0, 0)
        wide_lay.setSpacing(4)
        self._wide_line = QWidget()
        self._wide_line.setStyleSheet("background:transparent;")
        line_lay = QHBoxLayout(self._wide_line)
        line_lay.setContentsMargins(0, 0, 0, 0)
        line_lay.setSpacing(0)
        self._wide_title = self._head_widget(SHARED_D_NAME, self._show_fits_shared_d_help)
        line_lay.addWidget(self._wide_title)
        line_lay.addSpacing(self._WIDE_TITLE_GAP)
        line_lay.addStretch(1)
        wide_lay.addWidget(self._wide_line)
        self.wide_note = label("", ty.ui("sm"), t.muted_foreground)
        wide_lay.addWidget(self.wide_note)
        self.wide_note.hide()
        #: The pills, rebuilt with the General page's (a folder design's Conditions pills among them)
        self.fits_shared_d: Segmented | None = None
        #: The models every condition is fitted at while Force same D across is set-wide, None otherwise
        self._wide_models: list | None = None

        # THE TABLE'S HEAD, TWO LINES OVER THE ROWS' THREE COLUMNS (the lead, the models, the Force same D across),
        # each built from `ConditionRow`'s own gaps with its lead held to the rows' lead width (`_lay_columns`), so
        # everything in it stands over its column by construction, never by a measured offset.
        #
        # The titles: "Models" with its Help button over the rows' model pills (its width held to theirs, so the next
        # title starts where their column does), and "Force same D across" with its Help button over the rows' own
        # pills. Each title is the SAME `_control_head` every General control uses, so each Help button sits beside the
        # label of what it explains. The pills name their models themselves: no line of names (1 Nov 2026).
        self._titles = QWidget()
        self._titles.setStyleSheet("background:transparent;")
        titles_lay = QHBoxLayout(self._titles)
        titles_lay.setContentsMargins(0, 0, 0, 0)
        titles_lay.setSpacing(0)
        self._titles_lead = QWidget()
        self._titles_lead.setStyleSheet("background:transparent;")
        titles_lay.addWidget(self._titles_lead)
        titles_lay.addSpacing(ConditionRow.LEAD_GAP)
        self._models_title = self._head_widget("Models", self._show_models_help)
        titles_lay.addWidget(self._models_title)
        titles_lay.addSpacing(ConditionRow.COLUMN_GAP)
        self._shared_title = self._head_widget(SHARED_D_NAME, self._show_row_shared_d_help)
        titles_lay.addWidget(self._shared_title)
        titles_lay.addStretch(1)

        # THE EVERY CONDITION LINE, A ROW OF THE TABLE THAT SETS THE OTHERS: its name at the end of the lead, in
        # muted ink like the titles above it, then the same model pills (`_ModelPills`) and the same Force same D
        # across pills as a row. A click on a model's pill there goes through `workspace_view.toggle_every_model`
        # (`_set_every_model`): lit -- every included row has it -- it comes off every row, never emptying one; unlit,
        # it goes on every included row. It lights what every included row has (`_show_every_models`); its Force same
        # D across pills set every row's own (`_set_every_shared_d`) and show the rows' common value.
        self._every_row = QWidget()
        self._every_row.setStyleSheet("background:transparent;")
        every_lay = QHBoxLayout(self._every_row)
        every_lay.setContentsMargins(0, 0, 0, 0)
        every_lay.setSpacing(0)
        self._every_lead = QWidget()
        self._every_lead.setStyleSheet("background:transparent;")
        every_lead_lay = QHBoxLayout(self._every_lead)
        every_lead_lay.setContentsMargins(0, 0, 0, 0)
        every_lead_lay.setSpacing(0)
        every_lead_lay.addStretch(1)
        self._every_label = label("Every condition", ty.ui("sm", "medium"), t.muted_foreground)
        every_lead_lay.addWidget(self._every_label)
        every_lay.addWidget(self._every_lead)
        every_lay.addSpacing(ConditionRow.LEAD_GAP)
        self.every_models = _ModelPills(t)
        self.every_models.model_pressed.connect(self._set_every_model)
        every_lay.addWidget(self.every_models)
        every_lay.addSpacing(ConditionRow.COLUMN_GAP)
        #: `{condition key: the Force same D across its row was SET to}` -- a row not named follows the General
        #: page's (`_sync_shared_d`). Kept across `set_conditions`.
        self._own_shared_d: dict[str, str] = {}
        self.every_shared_d = _SharedDChoice(t)
        self.every_shared_d.changed.connect(self._set_every_shared_d)
        every_lay.addWidget(self.every_shared_d)
        every_lay.addStretch(1)
        # A hairline under it: the line above it sets the rows, it is not one of them
        self._every_rule = QWidget()
        self._every_rule.setFixedHeight(1)
        self._every_rule.setStyleSheet(f"background:{t.border}; border:none;")

        # FOUR POPULATIONS IS OPT-IN for "Best (BIC)" -- see `comparison.DEFAULT_BEST_MAX_POP`. Under the rows, its
        # box in the rows' include-box column; part of the page's pinned chrome, never scrolled past.
        self._four_row = QWidget()
        self._four_row.setStyleSheet("background:transparent;")
        four = QHBoxLayout(self._four_row)
        four.setContentsMargins(0, 0, 0, 0)
        four.setSpacing(ConditionRow.BOX_GAP)
        self.best_tries_four = Checkbox(t, checked=False)
        four.addWidget(self.best_tries_four)
        four.addWidget(label("Best (BIC) also tries 4 populations (about fourfold slower)",
                             ty.ui("sm"), t.foreground))
        four.addStretch(1)

        # The titles, the Every condition line, its rule and the four-populations line are FIXED height (no
        # stretch); `_cond_scroll` alone takes stretch 1, so it is the only part of this page that ever shrinks
        # below its natural content height -- `set_conditions` computes exactly how much of `_conditions_h` that
        # leaves it (`_cond_chrome_h`) and decides `needs_scrollbar` from THAT.
        self._cond_page = QWidget()
        self._cond_page.setStyleSheet("background:transparent;")
        cond_page_lay = QVBoxLayout(self._cond_page)
        cond_page_lay.setContentsMargins(0, 0, 0, 0)
        cond_page_lay.setSpacing(0)
        cond_page_lay.addWidget(self._wide_host)
        cond_page_lay.addSpacing(self._WIDE_GAP)
        cond_page_lay.addWidget(self._titles)
        # 4 and 12 below (6 and 16 until 1 Nov 2026, the night): with the Force same D across on top, four rows of
        # five fit a maximised laptop window at 125 % (`ComputeDialog._fit_window`)
        cond_page_lay.addSpacing(4)
        cond_page_lay.addWidget(self._every_row)
        cond_page_lay.addSpacing(10)
        cond_page_lay.addWidget(self._every_rule)
        cond_page_lay.addSpacing(10)
        cond_page_lay.addWidget(self._cond_scroll, 1)
        cond_page_lay.addSpacing(12)
        cond_page_lay.addWidget(self._four_row)

        # -- the General page: the settings grid + note. `Lock Model` no
        # longer lives here -- see above. --
        self._general_page = QWidget()
        self._general_page.setStyleSheet("background:transparent;")
        general_lay = QVBoxLayout(self._general_page)
        general_lay.setContentsMargins(0, 0, 0, 0)
        general_lay.setSpacing(16)
        self._general_lay = general_lay

        # The `QGridLayout` (`_build_wide_grid`) -- built empty here, filled by
        # `_rebuild_grid` from `set_conditions`, since a project's folder design
        # adds Shared D pills. See the module docstring.
        self._grid_host = QWidget()
        self._grid_host.setStyleSheet("background:transparent;")
        self._grid_lay = QVBoxLayout(self._grid_host)
        self._grid_lay.setContentsMargins(0, 0, 0, 0)
        self._grid_lay.setSpacing(16)
        general_lay.addWidget(self._grid_host)

        # HOW MANY CHUNKS (asked 25 Oct 2026: "for plotting in chunks, maybe we can choose the chunk size
        # somewhere?"). A row UNDER the grid, there only while Uncertainty is Chunks: beside the pills it would
        # widen the grid's left column, and under the grid no control moves when it appears -- the page slides to
        # its new height instead (`_show_chunks`).
        self._chunk_row = QWidget()
        self._chunk_row.setStyleSheet("background:transparent;")
        chunk_lay = QHBoxLayout(self._chunk_row)
        chunk_lay.setContentsMargins(0, 0, 0, 0)
        chunk_lay.setSpacing(self._ECDF_PAIR_GAP)
        chunk_lay.addWidget(label("Chunks per condition", ty.ui("sm", "medium"), t.muted_foreground))
        self.chunks = NumberField(t, *CHUNKS_RANGE, CHUNKS_DEFAULT, width=self._ECDF_FIELD_W)
        self.chunks.setToolTip("How many random chunks each condition's movies are dealt into. More chunks give "
                               "more points for a test, each from less data. A chunk needs at least two movies "
                               "(or two cells); Run reports a condition with too few.")
        self.chunks.changed.connect(lambda v: self.changed.emit("chunks", str(v)))
        chunk_lay.addWidget(self.chunks)
        chunk_lay.addStretch(1)
        general_lay.addWidget(self._chunk_row)
        self._chunk_row.setVisible(False)

        # The note. Hidden unless a setting has moved since the last Run --
        # see `set_stale`.
        self.note = label("", ty.ui("sm"), t.muted_foreground)
        self.note.setWordWrap(True)
        self.note.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.note.hide()
        general_lay.addWidget(self.note)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._cond_page)      # index 0
        self._pages.addWidget(self._general_page)  # index 1
        self._pages.setCurrentIndex(1)              # General opens first
        lay.addWidget(self._pages)

        #: `General`'s own natural height (the cap) and `Conditions`' own
        #: target (its natural height, or the cap itself once it needs a
        #: scrollbar) -- both set by `set_conditions`, read by
        #: `_on_tab_changed`. `None` until the first `set_conditions` call.
        self._general_h: int | None = None
        self._conditions_h: int | None = None
        #: The most height the pages may take in the window the dialog is open in (`set_pages_room`), None for none
        self._pages_room: int | None = None
        self._height_anim = QVariantAnimation(self)
        # LINEAR, as the sidebar's own collapse is -- see `Sidebar.set_collapsed`
        # and its own note on the registry's constant-speed slide.
        self._height_anim.setEasingCurve(QEasingCurve.Linear)
        self._height_anim.valueChanged.connect(self._apply_pages_height)

        self.set_conditions(conditions)

    def _apply_pages_height(self, v) -> None:
        """One tick of the height slide. `pages_resizing` fires alongside
        the resize itself, not after -- `ComputeDialog`'s repaint needs to
        cover the region `self._card` just vacated by shrinking/recentring,
        and that region only exists for the ONE FRAME between this resize
        and the next; emitting it late would repaint a rect already
        overwritten by the next tick's resize instead."""
        self._pages.setFixedHeight(round(v))
        self.pages_resizing.emit()

    def _on_tab_changed(self, key: str) -> None:
        """Switch the visible page FIRST, then slide `self._pages`' height
        to that page's own target -- see `_slide_pages` for the animation
        itself, and the module docstring for why a project needing
        `Conditions`' own scrollbar makes this slide a no-op rather than a
        special case. The Models and Force same D across Help buttons live INSIDE the
        Fits page (`self._titles`), so switching pages hides them for
        free -- no visibility bookkeeping needed here."""
        self._pages.setCurrentIndex(0 if key == "fits" else 1)
        target = self._conditions_h if key == "fits" else self._general_h
        if target is not None:
            self._slide_pages(target)

    def _slide_pages(self, target: int, *, animate: bool = True) -> None:
        """Tween `self._pages`' height to `target` -- `ComputeDialog`'s own
        card recentres itself for free as it does, since it only ever sizes
        to `self._pages`' current height via the surrounding layout.

        Mirrors `Sidebar.set_collapsed`'s own signal-blocking guard, and
        widens it by one call. The sidebar's reason still applies:
        `stop()` on a FINISHED animation leaves `currentTime` sitting at
        that run's own end, so an unguarded `setEndValue` recomputes at
        that stale time and EMITS the value just assigned -- a snap to the
        old finished height and back. `slide_ms`, not a fixed duration,
        because unlike the sidebar's always-the-same travel, `target - now`
        varies with the project loaded -- and THAT is what the sidebar's
        own guard does not cover: MEASURED here (interrupting a switch
        mid-flight, not merely re-triggering a finished one), an unguarded
        `setDuration` alone can emit too, if the new duration is SHORTER
        than whatever `currentTime` was left at by the interrupted run --
        progress clamps to 1.0 against the OLD start/end still in effect,
        emitting the interrupted animation's stale target. `Sidebar` never
        hits this because its duration is set once, at construction, and
        never changes; this one has to reset it on every call. All three
        setters -- `setDuration`, `setStartValue`, `setEndValue` -- are
        blocked together here for exactly that reason.
        """
        now = self._pages.height()
        self._height_anim.stop()
        if not animate or target == now or not self.isVisible():
            self._pages.setFixedHeight(target)
            return
        blocked = self._height_anim.blockSignals(True)
        self._height_anim.setDuration(slide_ms(target - now))
        self._height_anim.setStartValue(float(now))
        self._height_anim.setEndValue(float(target))
        self._height_anim.blockSignals(blocked)
        self._height_anim.start()

    #: Every Help popover's content is built to THIS width, not left to its
    #: labels' own natural sizeHint -- see `_open_help_popover`'s docstring
    #: for why a variable width was a real bug, not a cosmetic one.
    _HELP_CONTENT_W = 260

    #: Condition rows shown before the list scrolls where there is NO window to measure (`set_pages_room(None)`); with
    #: one, the window's room decides (4 Nov 2026). Before, where the window has room for them: since the Force same D
    #: across on top (1 Nov 2026, the night) the window the dialog opens in decides how many of these fit
    #: (`set_pages_room`), so a laptop at 125 % shows fewer and scrolls the rest.
    MAX_VISIBLE_ROWS = 5
    #: ... and shown whatever the window's room
    MIN_VISIBLE_ROWS = 2
    #: The Fits page's Force same D across (on top): its title to its pills, and the block to the table under it
    _WIDE_TITLE_GAP = 12
    _WIDE_GAP = 10

    def _control_head(self, name: str, on_help) -> QHBoxLayout:
        """A section label + `_HelpButton`, shared by every `_control` cell
        AND the `Lock Model` row below the grid -- so the two carry the
        same affordance rather than Lock Model growing its own copy.
        `on_help` is called with the button itself (the popover's anchor)
        when it is clicked."""
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(label(name, ty.ui("sm", "medium"), self.t.muted_foreground))
        btn = _HelpButton(self.t)
        btn.clicked.connect(lambda b=btn: on_help(b))
        head.addWidget(btn)
        head.addStretch(1)
        return head

    def _control(self, key: str, name: str, options) -> QWidget:
        """One `Shared D` / `Overlay` / ... cell: a label + Help button
        above a `Segmented`. The pills carry NO hover tooltip any more --
        `_open_help_popover` is where their descriptions now live,
        click-triggered rather than hovering every pill in turn. See
        `CONTROL_HELP`."""
        t = self.t
        cell = QWidget()
        cell.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(cell)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        cl.addLayout(self._control_head(
            name, lambda b, k=key, o=options: self._show_help(k, o, b)))
        seg = Segmented(t, [lab for _v, lab, _tip in options],
                        current=[v for v, _l, _tip in options].index(
                            DEFAULT_COMPUTE[key]),
                        size="sm")
        seg.changed.connect(
            lambda i, k=key, o=options: self.changed.emit(k, o[i][0]))
        self.controls[key] = seg
        if key == "shared_d":
            # THE ROWS FOLLOW IT (1 Nov 2026): each row not set on its own shows this value; a set-wide one (Whole
            # data set, Conditions) disables them all and shows itself (`_sync_shared_d`)
            seg.changed.connect(lambda _i: self._sync_shared_d())
        if key == "uncertainty":
            chunk = [v for v, _l, _t in options].index("chunk")
            seg.changed.connect(lambda i, c=chunk: self._show_chunks(i == c))
            self._show_chunks(seg.current() == chunk)
        if key != "ecdf":
            cl.addWidget(seg)
            return cell
        # CUSTOM NEEDS A NUMBER, and it sits beside the pills rather than in a row of
        # its own: it only exists while Custom is picked, and a row that appears and
        # disappears would move every control below it.
        pair = QHBoxLayout()
        pair.setContentsMargins(0, 0, 0, 0)
        pair.setSpacing(self._ECDF_PAIR_GAP)
        pair.addWidget(seg)
        low, high = ECDF_CUSTOM_RANGE
        self.ecdf_points = NumberField(t, low, high, ECDF_FAST_POINTS, width=self._ECDF_FIELD_W)
        self.ecdf_points.setToolTip("ECDF points the fit is evaluated at")
        self.ecdf_points.changed.connect(
            lambda v: self.changed.emit("ecdf_points", str(v)))
        pair.addWidget(self.ecdf_points)
        pair.addStretch(1)
        cl.addLayout(pair)
        custom = [v for v, _l, _t in options].index("custom")
        self.ecdf_points.setVisible(seg.current() == custom)
        seg.changed.connect(lambda i, c=custom: self.ecdf_points.setVisible(i == c))
        return cell

    def _controls_table(self) -> tuple:
        """`COMPUTE_CONTROLS` with this project's Force same D across: the folder design's Conditions pill -- one per
        factor where there are several, "Conditions, per <factor>" (`workspace_view.group_label`) -- after Replicates
        and before Whole data set, the experimenter's order (asked 25 Oct 2026 as "Within <factor>", named 1 Nov
        2026). Without a design it is `COMPUTE_CONTROLS` itself: Conditions would be the Whole data set's fit."""
        if not self._factor_names:
            return COMPUTE_CONTROLS
        out = []
        n = len(self._factor_names)
        for key, name, options in COMPUTE_CONTROLS:
            if key == "shared_d":
                at = [v for v, _l, _t in options].index("condition") + 1
                across = tuple((GROUP_PREFIX + f, group_label(f, n), group_tip(f)) for f in self._factor_names)
                options = options[:at] + across + options[at:]
            out.append((key, name, options))
        return tuple(out)

    def _show_chunks(self, on: bool) -> None:
        """Show the chunk count while Uncertainty is Chunks, and slide General to its new height."""
        self._show_row(self._chunk_row, on)

    def _show_row(self, row, on: bool) -> None:
        """A row under the grid shown or hidden, and General slid to its new height."""
        if row.isHidden() != on:
            return
        row.setVisible(on)
        if getattr(self, "_general_h", None) is None:
            return
        self._general_h = self._general_page.sizeHint().height()
        if self.tabs.current() == "general":
            self._slide_pages(self._general_h)

    def _show_help(self, key: str, options, trigger: _HelpButton) -> None:
        """The six General controls' own Help popover: `CONTROL_HELP[key]`
        above each option's `Label: description` line."""
        self._open_help_popover(CONTROL_HELP.get(key), options, trigger)

    def _head_widget(self, name: str, on_help) -> QWidget:
        """`_control_head` in a widget of its own -- a column title on the Fits page, which `_lay_columns` can hold to
        its column's width."""
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        host.setLayout(self._control_head(name, on_help))
        return host

    def _show_models_help(self, trigger: _HelpButton) -> None:
        """The Models column's Help popover: `MODELS_HELP`, then `Best (BIC)` and `saSPT` by their own descriptions and
        `1`-`4` collapsed into ONE line ("force that many"), read off `MODEL_CHOICES` so the two cannot drift apart;
        then what two lit on a row do, what the Every condition line does, and the include box."""
        best, one, _two, _three, four, saspt = MODEL_CHOICES
        options = (
            best,
            (None, f"{one[0]}-{four[0]}", "Force this condition to that many populations."),
            saspt,
            (None, "A click",
             "Lights a model, or puts it out again. A row always keeps at least one lit model."),
            (None, "Two or more on a row",
             ("Every condition is fitted at every model lit on any row, and each row draws the ones it lights, side "
              "by side, in shades of its colour: 1 and 2 populations on measured data, 1 alone "
              "on a simulation. A test between two fits of one condition is refused.")),
            (None, "Every condition",
             ("Clicking a model there lights it on every included row. Clicking a lit one puts it out on every "
              "row that keeps another model.")),
            (None, "The box before a name",
             "Include this condition in the fit. An unchecked condition is loaded but skipped."),
        )
        self._open_help_popover(MODELS_HELP, options, trigger)

    def _show_fits_shared_d_help(self, trigger: _HelpButton) -> None:
        """The Fits page's Force same D across (on top): the General page's popover -- the same setting -- and what a
        set-wide value does to the page (`workspace_view.SET_WIDE_HELP`)."""
        options = tuple(self._shared_d_options()) + ((None, "On this page", SET_WIDE_HELP),)
        self._open_help_popover(CONTROL_HELP["shared_d"], options, trigger)

    def _show_row_shared_d_help(self, trigger: _HelpButton) -> None:
        """The Force same D across column's Help popover: `ROW_SHARED_D_HELP`, the two values a row takes by their own
        descriptions (`ROW_SHARED_D_CHOICES`), then the Every condition pills and the set-wide values (Conditions only
        where the project has a folder design, as the General page offers it)."""
        wide = [shared_d_pill("group")] if self._factor_names else []
        wide.append(SHARED_D_PILLS["global"])
        options = tuple(ROW_SHARED_D_CHOICES) + (
            (None, "Every condition",
             (f"These pills set every row's {SHARED_D_NAME} at once. They "
              "show the rows' common value, or none while the rows differ; the General page's setting is unchanged.")),
            (None, ", ".join(wide),
             (("Picked at the top of this page or on the General page, these force one D across conditions"
               if len(wide) > 1 else
               "Picked at the top of this page or on the General page, it forces one D across conditions")
              + f". Every row then shows it and cannot change it, and its models are the Every condition line's. "
                f"Pick {SHARED_D_PILLS['condition']} or {SHARED_D_PILLS['replicate']} there to get each row's own "
                f"setting back.")),
            (None, "On the figure",
             ("Where conditions differ, each column says under it how its D was fitted. A test between two "
              "conditions fitted differently is refused, because the difference would include the method.")),
        )
        self._open_help_popover(ROW_SHARED_D_HELP, options, trigger)

    def _open_help_popover(self, lead: str | None, options,
                           trigger: _HelpButton) -> None:
        """Build and show a Help popover -- `lead` above each option's own
        `Label: description` line, reusing the tooltip text `COMPUTE_
        CONTROLS` / `LOCK_ROW_1` / `LOCK_ROW_2` already carry rather than
        writing it twice.

        Follows `toolbar._open_popover`'s own shape: a `Panel` (the
        in-page popover, never a `QMenu` -- see its module docstring for
        why) holding one bordered `holder` widget built with its own
        margins and spacing, since `Panel.add_widget` itself adds none.
        `self.window()` resolves to `WorkspaceWindow` when this panel sits
        inside a real `ComputeDialog`, and to the dialog itself in the
        parentless / headless construction every test in this file uses --
        no reference-threading needed either way, same as `toolbar.py`'s
        `button.window()`.

        **EVERY LABEL IS PINNED TO `_HELP_CONTENT_W`, NOT LEFT TO WRAP AT
        WHATEVER WIDTH THE LAYOUT EVENTUALLY SETTLES ON.** Reported
        directly: `Overlay States`' popover (two short one-line options)
        showed a visibly larger gap between its lines than `Shared D`'s
        (three wrapped, multi-line options) did, from the SAME `spacing(8)`
        VBox. MEASURED, not guessed: a word-wrapped `QLabel`'s `sizeHint()`
        is computed at whatever width it happens to have *at that moment*,
        which for a label shorter than its siblings is its own narrow
        natural width -- `<b>2:</b> Overlay drawn at 2 populations.`
        reported `sizeHint() == (130, 34)`, i.e. TWO lines' worth of
        height, at a width where the layout went on to actually render it
        as ONE line once stretched to the column's real width (228px). The
        layout allocates the STALE two-line height, the label paints one
        line inside it, and the leftover height reads as an oversized gap
        -- worse the SHORTER the option text is, which is exactly why
        `Overlay States` (the shortest control here) showed it worst and
        `Shared D` (genuinely multi-line at any width) did not. Giving
        every label a FIXED width up front makes `heightForWidth` resolve
        against the real, final width from the start, so every popover
        wraps -- and therefore spaces its lines -- identically regardless
        of how short its shortest option happens to be.
        """
        t = self.t
        if self._help_popover is not None:
            self._help_popover.close_chain()
            self._help_popover = None

        menu = Panel(t, self.window())
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(12, 10, 12, 10)
        hl.setSpacing(8)
        content_w = self._HELP_CONTENT_W - 2 * 12
        if lead:
            head = label(lead, ty.ui("sm"), t.foreground)
            head.setWordWrap(True)
            head.setFixedWidth(content_w)
            hl.addWidget(head)
        for _v, opt_label, tip in options:
            line = label(f"<b>{opt_label}:</b> {tip}", ty.ui("sm"),
                        t.muted_foreground)
            line.setWordWrap(True)
            line.setFixedWidth(content_w)
            hl.addWidget(line)
        menu.add_widget(holder)
        menu.set_minimum_visible_width(self._HELP_CONTENT_W)

        def _on_closed() -> None:
            if self._help_popover is menu:
                self._help_popover = None

        menu.closed.connect(_on_closed)
        self._help_popover = menu
        menu.popup_under(trigger)

    def _measure_pill_width(self, key: str) -> int:
        """A throwaway `Segmented`'s own natural width -- never added to any
        layout, built only to be asked its `sizeHint` before the real grid
        (and its live `self.controls`) exists. Used solely by
        `_wide_columns`."""
        _name, options = {k: (n, o) for k, n, o in self._controls_table()}[key]
        probe = Segmented(self.t, [lab for _v, lab, _tip in options],
                          current=0, size="sm")
        return probe.sizeHint().width()

    #: The `_build_wide_grid` gap between columns, and between the two controls of one line.
    _GRID_GAP = 20
    #: ECDF's Custom count field and its gap from the pills (`_control`), named once so
    #: `_wide_columns` measures what `_control` builds.
    _ECDF_FIELD_W = 112
    _ECDF_PAIR_GAP = 8

    #: THE GENERAL PAGE (1 Nov 2026, the second of three looks at the whole dialog): two columns, each control at its
    #: own width at its cell's left edge -- on the left Shared D, Uncertainty, and Weighting with ECDF beside it; on
    #: the right Overlay, Time lag and the Optimiser, one above the other on ONE left edge. Before, three lines of
    #: pairs each stretched to its column put Time lag's left edge 230 px right of Uncertainty's and left a hole
    #: under ECDF. ECDF sits LAST on its line so its Custom field appears in room the left column already holds
    #: (`_wide_columns` counts it), moving nothing; on the right it made that column 311 px wide, the page 746, past a
    #: five-condition table's 690.
    _WIDE_LEFT = (("shared_d",), ("uncertainty",), ("weighting", "ecdf"))
    _WIDE_RIGHT = ("overlay", "lag", "gradient", "precision")

    def _wide_columns(self) -> tuple:
        """``(left, right)``: the widths the General page's two columns need, from each control's own pills (a pair
        on one line is both and the gap between them; ECDF counts its Custom field, which appears beside it)."""
        def width(key):
            extra = self._ECDF_PAIR_GAP + self._ECDF_FIELD_W if key == "ecdf" else 0
            return self._measure_pill_width(key) + extra

        left = max(sum(width(k) for k in line) + self._GRID_GAP * (len(line) - 1) for line in self._WIDE_LEFT)
        right = max(width(k) for k in self._WIDE_RIGHT)
        return left, right

    def _build_wide_grid(self, controls_by_key: dict) -> QWidget:
        """The General page: a `QGridLayout` of two columns (`_WIDE_LEFT`, `_WIDE_RIGHT`), each control at its own
        width at its cell's left edge, so the right column's three share one left edge as a property of the grid. The
        columns are held to `_wide_columns` (ECDF's field included), so picking Custom moves no control, and the
        page's slack sits after the right column (column 2 takes the stretch), not between the two.

        THE ONLY ARRANGEMENT SINCE 1 NOV 2026. A narrow one -- Shared D and Overlay each on a line of their own above a
        small grid -- was built when a condition row was a name and six pills (~360 px) and the two-column grid would
        have widened the card past it. A row is never that narrow now (its model ticks and Shared D pills alone are
        ~490 px), and the narrow arrangement is the taller: MEASURED with five conditions, the card 563 px tall
        against 419, past the ~550 px a maximised window has on a 1366 x 768 laptop at 125 %. Where the rows are
        narrower than this grid, the grid widens the card instead."""
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(self._GRID_GAP)
        grid.setVerticalSpacing(16)
        from itertools import zip_longest

        # the right column may run a line longer than the left (Localisation error, 6 Nov 2026): that line's left
        # cell stays empty
        for row, (line, right) in enumerate(zip_longest(self._WIDE_LEFT, self._WIDE_RIGHT, fillvalue=())):
            if not line:
                pass
            elif len(line) == 1:
                (key,) = line
                name, options = controls_by_key[key]
                grid.addWidget(self._control(key, name, options), row, 0, Qt.AlignLeft)
            else:
                pair = QWidget()
                pair.setStyleSheet("background:transparent;")
                pair_lay = QHBoxLayout(pair)
                pair_lay.setContentsMargins(0, 0, 0, 0)
                pair_lay.setSpacing(self._GRID_GAP)
                for key in line:
                    name, options = controls_by_key[key]
                    pair_lay.addWidget(self._control(key, name, options))
                pair_lay.addStretch(1)
                grid.addWidget(pair, row, 0, Qt.AlignLeft)
            name, options = controls_by_key[right]
            grid.addWidget(self._control(right, name, options), row, 1, Qt.AlignLeft)
        left, right = self._wide_columns()
        grid.setColumnMinimumWidth(0, left)
        grid.setColumnMinimumWidth(1, right)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)
        return host

    def _rebuild_grid(self) -> None:
        """Tear down and rebuild `self._grid_host` -- called from `set_conditions`, since a freshly loaded project
        can carry another folder design (its Conditions pills widen Force same D across)."""
        while self._grid_lay.count():
            item = self._grid_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self.controls = {}
        controls_by_key = {k: (n, o) for k, n, o in self._controls_table()}
        self._grid_lay.addWidget(self._build_wide_grid(controls_by_key))
        # the Fits page's copy of Force same D across, with the same pills (`_sync_shared_d` moves it)
        if self.fits_shared_d is not None:
            discard(self.fits_shared_d)
        options = self._shared_d_options()
        self.fits_shared_d = Segmented(self.t, [lab for _v, lab, _tip in options],
                                       current=self.controls["shared_d"].current(), size="sm")
        self.fits_shared_d.changed.connect(self._on_fits_shared_d)
        self._wide_line.layout().insertWidget(2, self.fits_shared_d)

    def set_conditions(self, conditions) -> None:
        """Re-point at a freshly loaded project's own conditions, each
        defaulting to included + `DEFAULT_MODEL` -- or, for a condition
        already shown, the models its row ticks now. Also lines the columns up
        (every name widened to the longest loaded, `_lay_columns`), rebuilds
        the settings grid (which arrangement fits depends on the conditions
        just loaded), and re-derives both of `self._pages`' height targets
        -- see the module docstring's "CONDITIONS AND GENERAL ARE TWO TABS"
        and "SWITCHING TABS SLIDES THE HEIGHT, NOT JUST THE PAGE". Snaps to
        whichever tab is current rather than animating: a fresh project is
        not a tab switch."""
        kept = {key: row.own_models() for key, row in self.rows.items()}
        while self._cond_lay.count() > 1:   # keep the trailing stretch
            item = self._cond_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self.rows = {}
        for key, name in conditions:
            r = ConditionRow(self.t, key, name, models=kept.get(key, (DEFAULT_MODEL,)))
            r.toggled.connect(self._on_row_toggled)
            r.model_changed.connect(self.condition_model_changed.emit)
            r.models_changed.connect(self._on_row_models)
            r.shared_d_changed.connect(self._on_row_shared_d)
            self._cond_lay.insertWidget(self._cond_lay.count() - 1, r)
            self.rows[key] = r
        self._lay_columns()
        self._show_every_models()

        self._rebuild_grid()
        # every row's Force same D across: its own where it was set, else the General page's just rebuilt
        # (`_sync_shared_d`)
        self._sync_shared_d()

        # `General`'s own sizeHint -- a plain QWidget, not a QScrollArea, so
        # (unlike the bug this replaced) it correctly reflects what its
        # content actually needs.
        self._general_h = self._general_page.sizeHint().height()

        # `Conditions`' own chrome -- the titles, the Every condition line and
        # its rule, the four-populations line, and the gaps between them -- is
        # FIXED height (`cond_page_lay`'s stretch-0 items); only `_cond_scroll`
        # (stretch 1) ever shrinks below its natural height. So the row list's
        # own budget is `_conditions_h` MINUS that chrome -- measured from
        # `_cond_host` (the rows) directly, not `_cond_scroll` -- see the
        # width comment below for why the two are not interchangeable.
        self._fit_conditions_h()

        current = (self._conditions_h if self.tabs.current() == "fits"
                   else self._general_h)
        self._slide_pages(current, animate=False)

    def _cond_chrome_h(self) -> int:
        """The Fits page's fixed height: every item of its layout but the scrolling rows, the gaps included."""
        lay = self._cond_page.layout()
        total = lay.contentsMargins().top() + lay.contentsMargins().bottom()
        for i in range(lay.count()):
            item = lay.itemAt(i)
            if item.widget() is self._cond_scroll:
                continue
            total += item.sizeHint().height()
        return total

    def _fit_conditions_h(self) -> None:
        """The Fits page's height target (`_conditions_h`): its fixed chrome -- every line but the rows, the gaps
        included (`_cond_chrome_h`) -- and as many rows as it shows before they scroll: every row up to
        `MAX_VISIBLE_ROWS`, fewer where the window the dialog is open in has no room for them (`set_pages_room`), never
        fewer than `MIN_VISIBLE_ROWS`. Called by `set_conditions`, by `set_pages_room` and when the line under the Force
        same D across on top comes or goes (`_show_wide_note`).

        ITS OWN BUDGET, NOT GENERAL'S. This was `min(chrome + rows, _general_h)`,
        written while `Lock Model` still lived on General and made that page tall.
        Once it moved here, a project with long condition names got the WIDE General
        grid (two short rows), and that cap left Conditions less height than its own
        fixed chrome: MEASURED with nine conditions named like one user's, one row showed,
        the two Lock Model rows were drawn over each other and the scrollbar was a
        sliver -- reported as "so distorted that u cant see", and the reason a "2"
        nobody could see stayed locked. The rows now scroll past `MAX_VISIBLE_ROWS`,
        and the chrome is never squeezed.

        `_cond_lay`'s own right margin permanently reserves
        `_scrollbar_w` INSIDE `_cond_host` (see its own comment) --
        correct once a scrollbar actually appears, but reported
        directly as a defect when it never does: `_cond_host.sizeHint()`
        still carried the reservation regardless, widening the card
        past what the rows actually render, and reading as a lopsided
        right margin nothing else in the dialog had. The margin is now
        conditional on `needs_scrollbar`, computed the same way
        `_conditions_h` already is, so a short project's card matches
        its rows exactly and the reservation returns the moment a
        longer one needs it.

        `QScrollArea.sizeHint()` IS broken once `_cond_scroll` is not
        `self._pages`' DIRECT child -- it sits inside `_cond_page`, between
        the header lines and the four-populations line, and answers `(14, 0)`.
        `setMinimumWidth` (`_hold_page_width`) is what actually worked for `_HelpButton.SIZE` and
        does here too, for the same reason: a `QWidgetItem`'s effective
        sizeHint clamps UP to a declared minimum even when the widget's own
        `sizeHint()` lies about needing less."""
        chrome_h = self._cond_chrome_h()
        cond_rows_h = self._cond_host.sizeHint().height()
        n_rows = max(len(self.rows), 1)
        visible = min(n_rows, self.MAX_VISIBLE_ROWS)
        room = getattr(self, "_pages_room", None)
        if room is not None and cond_rows_h > 0:
            # THE WINDOW'S ROOM decides, as the apparent-D and HMM dialogs' (`run_dialog.FitsPage._fit`): the rows that
            # fit under the chrome, two at least. `MAX_VISIBLE_ROWS` only caps a dialog with no window to measure
            # (4 Nov 2026: it held ten conditions at five in a window that held nine)
            fit = int((room - chrome_h) * n_rows // cond_rows_h)
            visible = max(min(n_rows, fit), min(n_rows, self.MIN_VISIBLE_ROWS))
        rows_budget = (cond_rows_h if n_rows <= visible
                       else int(round(cond_rows_h * visible / n_rows)))
        #: the rows' share of the Fits page's height
        self._rows_budget = rows_budget
        self._conditions_h = chrome_h + rows_budget
        needs_scrollbar = cond_rows_h > rows_budget
        self._cond_lay.setContentsMargins(
            0, 0, self._scrollbar_w if needs_scrollbar else 0, 0)
        self._hold_page_width()

    def set_pages_room(self, room) -> None:
        """THE CARD FITS ITS WINDOW (1 Nov 2026, the night): ``room`` is the most height the pages may take in the
        window the dialog is open in (`ComputeDialog._fit_window`), None for no limit. The Fits page then shows as many
        rows as that holds (`_fit_conditions_h`) and scrolls the rest: all five of five conditions at 1920 x 1080, fewer
        on a laptop at 125 %, where the Force same D across on top took a line the card did not have."""
        self._pages_room = None if room is None else int(room)
        if getattr(self, "_conditions_h", None) is None:
            return
        self._fit_conditions_h()
        if self.tabs.current() == "fits":
            self._slide_pages(self._conditions_h, animate=False)

    def _lay_columns(self) -> None:
        """THE COLUMNS LINED UP: the lead -- a row's include box and name, the "Every condition" label at the end of
        its line's lead, the titles line's empty lead -- one width for every line, the longest of them, by widening
        every row's name to fill it. The model and Force same D across columns then start at the same x on every
        line, since each line lays them out with `ConditionRow`'s gaps and the same widgets (`_ModelPills`, the same
        labels in every tray; `_SharedDChoice`): widths every line shares by construction, never a measured offset.
        The "Models" title is held to the trays' width, so the "Force same D across" title starts over the rows'
        pills."""
        if not self.rows:
            return
        box = next(iter(self.rows.values())).box.maximumWidth() + ConditionRow.BOX_GAP
        lead_w = max([box + r.title.sizeHint().width() for r in self.rows.values()]
                     + [self._every_label.sizeHint().width()])
        for r in self.rows.values():
            r.title.setFixedWidth(lead_w - box)
        self._every_lead.setFixedWidth(lead_w)
        self._titles_lead.setFixedWidth(lead_w)
        trays = self.every_models.sizeHint().width()
        self._models_title.setFixedWidth(max(trays, self._models_title.sizeHint().width()))

    def _hold_page_width(self) -> None:
        """The Fits page at least as wide as its rows and its chrome (`set_conditions`' measurement, again whenever a
        row's width moves: a set-wide Force same D across on one pill)."""
        self._cond_page.setMinimumWidth(max(
            self._cond_host.sizeHint().width(),
            self._wide_host.sizeHint().width(),
            self._titles.sizeHint().width(),
            self._every_row.sizeHint().width(),
            self._four_row.sizeHint().width()))

    # -- the models each row lights (1 Nov 2026) ---------------------------------------------------------------- #
    def _included(self) -> list:
        """The rows in the Run, in row order."""
        return [row for row in self.rows.values() if row.box.checked]

    def _on_row_toggled(self, key: str, on: bool) -> None:
        """A row's include box moved: the Every condition pills are read off the included rows."""
        self._show_every_models()
        self.condition_toggled.emit(key, on)

    def _on_row_models(self, key: str, _models: list) -> None:
        """A row's model pills moved (a user's click)."""
        self._show_every_models()
        self.changed.emit("fit_models", key)

    def _set_every_model(self, model: str) -> None:
        """A click on ``model``'s pill on the Every condition line: every included row's models as
        `workspace_view.toggle_every_model` says -- lit (every included row has it), it comes off every row that
        keeps another; unlit, it goes on every included row. While Force same D across is set-wide it is THE models of
        the one fit (`_wide_models`): the click toggles them (`workspace_view.toggle_model`, the last stays lit), and
        every row shows them."""
        if self._wide_models is not None:
            models = toggle_model(self._wide_models, model, MODEL_ORDER)
            if models == self._wide_models:
                return
            self._wide_models = models
            for row in self.rows.values():
                row.set_wide_models(models)
            self._show_every_models()
            self.changed.emit("fit_models", model)
            return
        rows = self._included()
        for row, models in zip(rows, toggle_every_model([row.models() for row in rows], model, MODEL_ORDER)):
            if models != row.models():
                row.set_models(models)
        self._show_every_models()
        self.changed.emit("fit_models", model)

    def _show_every_models(self) -> None:
        """The Every condition line lights the models every included row has (none while no row is included,
        `workspace_view.every_models`) -- or, while Force same D across is set-wide, the set's."""
        if self._wide_models is not None:
            self.every_models.set_models(self._wide_models)
            return
        self.every_models.set_models(models_every_row_has([row.models() for row in self._included()]))

    @property
    def _own_model(self) -> dict:
        """``{condition key: the models its row keeps for later}`` while Force same D across is set-wide and the rows
        show the set's ({} otherwise): what the window carries through a theme switch and a session
        (`apply_values(own_model=)`), so each row's own come back when the set-wide value lifts. (Always empty from 1
        Nov 2026 until that night: no lock hid a row's pick; before, the picks "Fit every condition at" hid.)"""
        if self._wide_models is None:
            return {}
        return {key: row.own_models() for key, row in self.rows.items()}

    def side_by_side(self) -> list:
        """THE MODELS EVERY CONDITION IS FITTED AT, side by side: the union of the included rows' models when any of
        them ticks two or more, in `MODEL_CHOICES` order; [] when each ticks one (each its own model)."""
        rows = [row.models() for row in self._included()]
        if not any(len(models) >= 2 for models in rows):
            return []
        return _models_in_order({m for models in rows for m in models})

    def shown_fits(self) -> dict:
        """``{condition key: the models its row ticks}`` for every row that ticks otherwise than `side_by_side` --
        the rows that draw fewer of the models side by side (`workspace_view.shown_fits` reads them, the included
        ones only); with no fits side by side, a row left out of the Run that ticks two, so `apply_values` puts every
        row back as it was."""
        fit = self.side_by_side()
        out = {}
        for key, row in self.rows.items():
            mine = row.models()
            if (mine != fit) if fit else len(mine) >= 2:
                out[key] = mine
        return out

    def set_every_models(self, models) -> None:
        """Every row lights exactly ``models`` (one at least), as clicking them on each would -- an older staging's
        lock, or "Counts", put back (`apply_values`)."""
        models = _models_in_order(models)
        for row in self.rows.values():
            row.set_models(models)
        if self._wide_models is not None:
            # set-wide: they are the one fit's models too
            self._wide_models = models
            for row in self.rows.values():
                row.set_wide_models(models)
        self._show_every_models()
        self.changed.emit("fit_models", ",".join(models))

    # -- a Force same D across per condition (1 Nov 2026) ------------------------------------------------------ #
    def _shared_d_options(self) -> tuple:
        """The General page's Force same D across options, a folder design's Conditions pills included
        (`_controls_table`)."""
        return dict((k, o) for k, _n, o in self._controls_table())["shared_d"]

    def general_shared_d(self) -> str:
        """The General page's Force same D across value: one of `ROW_SHARED_D`, "global", or
        ``GROUP_PREFIX + factor``."""
        seg = self.controls.get("shared_d")
        options = self._shared_d_options()
        if seg is None or seg.current() == -1:
            return DEFAULT_COMPUTE["shared_d"]
        return options[seg.current()][0]

    def _sync_shared_d(self) -> None:
        """Every row's Force same D across as the General page and the rows' own picks say: a row SET on its own
        (`_own_shared_d`) shows that, every other row the General page's value -- or, while the General page forces
        one D across conditions (Conditions, Whole data set), every row (and the Every condition line) shows that
        setting on one disabled pill, keeping its own pick for when Replicates or Don't force anywhere comes back
        (`_SharedDChoice.set_wide`)."""
        if self._restoring:
            return
        general = self.general_shared_d()
        options = [v for v, _l, _t in self._shared_d_options()]
        # ONE VALUE IN TWO PLACES: the Fits page's pills on top show the General page's
        if self.fits_shared_d is not None:
            index = options.index(general) if general in options else -1
            if self.fits_shared_d.current() != index:
                self.fits_shared_d.blockSignals(True)
                try:
                    self.fits_shared_d.select(index)
                finally:
                    self.fits_shared_d.blockSignals(False)
        wide = None
        if is_set_wide(general):
            wide = next((lab for v, lab, _t in self._shared_d_options() if v == general), str(general))
        # ONE D ACROSS CONDITIONS, ONE SET OF MODELS: picked, it starts from what every included row has (else Best,
        # `workspace_view.set_wide_models`); lifted, every row shows its own again
        if wide is None:
            self._wide_models = None
        elif self._wide_models is None:
            self._wide_models = set_wide_models([row.own_models() for row in self._included()], DEFAULT_MODEL)
        for key, row in self.rows.items():
            row.set_wide_models(self._wide_models)
            row.shared_d.set_wide(wide)
            if wide is None:
                row.shared_d.set_value(self._own_shared_d.get(key, general))
        self.every_shared_d.set_wide(wide)
        if wide is None:
            self._show_every_shared_d()
        self._show_every_models()
        self._show_wide_note(set_wide_note(general))
        if self.rows:
            self._hold_page_width()

    def _on_fits_shared_d(self, index: int) -> None:
        """The Fits page's Force same D across was clicked: it IS the General page's, so that one is selected (its
        own signal then stages the value and syncs the rows, `_sync_shared_d`)."""
        seg = self.controls.get("shared_d")
        if seg is not None and index != -1:
            seg.select(index)

    def _show_wide_note(self, text: str) -> None:
        """The line under the Fits page's Force same D across: ``text`` (`workspace_view.set_wide_note`), hidden when
        empty -- and the page's height target moved with it, slid to when the Fits page is in front."""
        self.wide_note.setText(text)
        if self.wide_note.isHidden() != bool(text):
            return
        self.wide_note.setVisible(bool(text))
        if getattr(self, "_conditions_h", None) is None:
            return
        # MEASURED AGAIN, NOT READ FROM A CACHE: hiding a widget that was never shown (the dialog closed, or the
        # General page in front) invalidates no layout, so the host kept the note's height -- 295 px after the note
        # went, against 272 before it came
        self._wide_host.layout().invalidate()
        self._wide_host.updateGeometry()
        self._cond_page.layout().invalidate()
        self._fit_conditions_h()
        if self.tabs.current() == "fits":
            self._slide_pages(self._conditions_h)

    def _show_every_shared_d(self) -> None:
        """The Every condition pills raise the rows' common value, none while they differ."""
        found = {row.shared_d.value() for row in self.rows.values()}
        self.every_shared_d.set_value(found.pop() if len(found) == 1 else None)

    def _on_row_shared_d(self, key: str, value: str) -> None:
        """A row's Force same D across was picked: it is that row's own from now on, whatever the General page
        says."""
        self._own_shared_d[key] = value
        self._show_every_shared_d()
        self.changed.emit("shared_d_rows", key)

    def _set_every_shared_d(self, value: str) -> None:
        """The Every condition pills were clicked: every row's own Force same D across is ``value``, as clicking each
        row would make it."""
        for key, row in self.rows.items():
            self._own_shared_d[key] = value
            row.shared_d.set_value(value)
        self.changed.emit("shared_d_rows", value)

    def shared_d_rows(self) -> dict:
        """``{condition key: its Force same D across}`` for every row fitted otherwise than the General page says
        (`workspace_view.shared_d_rows` reads it); {} while the General page forces one D across conditions, whatever
        the rows keep for later."""
        general = self.general_shared_d()
        if general not in ROW_SHARED_D:
            return {}
        return {key: value for key, row in self.rows.items()
                if (value := row.shared_d.value()) is not None and value != general}

    def values(self) -> dict:
        """The currently staged settings, `{key: value}` -- everything but
        the per-condition rows' include box and first model, which are read
        separately (`condition_values`) since their shape depends on how many
        conditions are loaded.

        THE MODELS IN THE ENCODING THE RUN HAS READ SINCE 30-31 OCT (`workspace_view.spec_from_values`, unchanged):
        ``fit_models`` the models side by side (`side_by_side`, [] when each row lights one), ``shown_fits`` the
        rows that light otherwise, ``lock_model`` always `DEFAULT_LOCK` (no lock any more)."""
        result = {key: options[self.controls[key].current()][0]
                  for key, _name, options in self._controls_table()}
        result["ecdf_points"] = self.ecdf_points.value()
        result["chunks"] = self.chunks.value()
        result["best_max_pop"] = 4 if self.best_tries_four.checked else 3
        result["fit_models"] = self.side_by_side()
        result["lock_model"] = DEFAULT_LOCK
        result["shown_fits"] = self.shown_fits()
        # each row fitted at another Force same D across than the General page's; none under Whole data set
        result["shared_d_rows"] = self.shared_d_rows()
        return result

    def apply_values(self, values: dict, conditions: dict, own_model=None, own_shared_d=None) -> None:
        """Put a staged configuration back, through the same pill `select` calls a
        click makes -- the theme switch rebuilds the window, and without this every
        setting went back to its default.

        THE MODELS (1 Nov 2026): each row in ``conditions`` takes its include box and its model; then, where the
        staging names models -- ``fit_models``, or an older one's lock or Overlay "Counts", read through
        `workspace_view.ticked_models` -- a row in ``shown_fits`` lights those, every other row the models named. A
        staging that names none (the Options screen's standards) leaves every row's models as they are.
        ``own_model`` held the picks a lock hid (30 Oct - 1 Nov); now the models a set-wide value keeps (below).

        THE ROWS' OWN FORCE SAME D ACROSS: ``own_shared_d`` (`_own_shared_d` as the window held it: the theme
        switch, a session) is put back as it was, every row SET on its own included -- also those it keeps, unshown,
        under Whole data set; without it a staging's ``shared_d_rows`` names the rows set, and without either (the
        Options screen's standards) the rows keep what they have. A value no row offers is refused by name.

        ONE D ACROSS CONDITIONS (1 Nov 2026, the night): a staging under a set-wide Force same D across names the SET's
        models (every row shows them, `values()`), so the rows it touches give the set its models, and ``own_model``
        (`_own_model` as the window held it) puts back the models each row keeps for later -- a one-model string, as
        a session from 30-31 Oct holds, read as that model. Under any other value ``own_model`` is not read. Nothing
        is synced until all of it is in (`_restoring`), so a General page value put back first does not take the
        rows' models before the staging's own arrive."""
        before = self._wide_models
        touched = set()
        self._restoring = True
        try:
            for key, _name, options in self._controls_table():
                offered = [v for v, _l, _t in options]
                # a value this project does not offer (Conditions per a factor it has no design for) leaves the pill
                if key in values and values[key] in offered:
                    self.controls[key].select(offered.index(values[key]))
            if "ecdf_points" in values:
                self.ecdf_points.setValue(int(values["ecdf_points"]))
            if "chunks" in values:
                self.chunks.setValue(int(values["chunks"]))
            if "best_max_pop" in values:
                self.best_tries_four.set_checked(int(values["best_max_pop"]) >= 4,
                                                 animate=False)
            for key, (included, model) in conditions.items():
                row = self.rows.get(key)
                if row is None:
                    continue
                row.box.set_checked(bool(included), animate=False)
                if _choice_index(MODEL_CHOICES, model) != -1:
                    row.set_models([model])
                    touched.add(key)
            if any(k in values for k in ("fit_models", "lock_model", "side_by_side")) \
                    or values.get("overlay") == "counts":
                # THE MODELS NAMED: `fit_models`, else what an older staging meant (a lock, or Overlay "Counts"' counts)
                named = ticked_models(values)
                shown = {str(k): _models_in_order(v) for k, v in dict(values.get("shown_fits") or {}).items() if v}
                for key, row in self.rows.items():
                    if key in shown:
                        row.set_models(shown[key])
                        touched.add(key)
                    elif named:
                        row.set_models(named)
                        touched.add(key)
            own = own_shared_d if own_shared_d is not None else values.get("shared_d_rows")
            if own is not None:
                own = {str(k): v for k, v in dict(own).items()}
                bad = {k: v for k, v in own.items() if v not in ROW_SHARED_D}
                if bad:
                    raise ValueError(f"a condition's own {SHARED_D_NAME} is one of {ROW_SHARED_D}, got {bad}")
                self._own_shared_d = own
        finally:
            self._restoring = False
        if is_set_wide(self.general_shared_d()):
            if touched:
                # the staging's rows carry the set's models
                self._wide_models = set_wide_models(
                    [row.own_models() for key, row in self.rows.items() if key in touched], DEFAULT_MODEL)
            else:
                self._wide_models = before
            for key, models in dict(own_model or {}).items():
                row = self.rows.get(str(key))
                if row is not None and models:
                    row.set_models([models] if isinstance(models, str) else models)
        else:
            self._wide_models = None
        self._sync_shared_d()

    def show_general(self) -> None:
        """Put `General` in front, without the slide -- asked for directly: "general
        tab should be first when I click on run". Called every time the dialog
        opens, so it opens on General even if Fits was the last tab used."""
        if self.tabs.current() != "general":
            self.tabs.blockSignals(True)
            self.tabs.select("general")
            self.tabs.blockSignals(False)
        self._pages.setCurrentIndex(1)
        if getattr(self, "_general_h", None) is not None:
            self._slide_pages(self._general_h, animate=False)

    def reorder(self, keys) -> None:
        """Put the condition rows in ``keys``' order (the nesting chosen, `factor_layout`), keeping each row as it is
        -- included or not, its models -- since `condition_values`, read in row order, is the Run's condition order."""
        keys = [k for k in keys if k in self.rows] + [k for k in self.rows if k not in keys]
        for index, key in enumerate(keys):
            row = self.rows[key]
            self._cond_lay.removeWidget(row)
            self._cond_lay.insertWidget(index, row)
        self.rows = {key: self.rows[key] for key in keys}

    def condition_values(self) -> dict:
        """`{condition: (included, model)}`, read fresh from the rows: the model is the row's first lit (its
        only one unless fits are side by side, when `values()` names them all)."""
        return {key: (row.box.checked, row.models()[0]) for key, row in self.rows.items()}

    def set_stale(self, changed: list) -> None:
        """Say which settings have moved since the figures were computed."""
        if not changed:
            self.note.hide()
            return
        names = {k: n for k, n, _o in self._controls_table()}
        names["ecdf_points"] = "ECDF points"
        names["chunks"] = "Chunks per condition"
        names["fit_models"] = "Models"
        names["shown_fits"] = "Models"
        names["shared_d_rows"] = f"{SHARED_D_NAME} per condition"
        names["best_max_pop"] = "Best (BIC)'s range"
        names["lock_model"] = "Models"

        def in_sentence(name):
            # the first letter lowered, an acronym ("ECDF") and the D of "Force same D across" kept
            return name if name[1:2].isupper() else name[:1].lower() + name[1:]

        which = ", ".join(dict.fromkeys(in_sentence(names[k]) for k in changed))
        self.note.setText(
            f"{which} changed after this was computed; the figures still "
            f"show the previous fit. Run again to update them.")
        self.note.show()


class ComputeDialog(OverlayDialog):
    """shadcn's `Dialog`, holding `ComputePanel` -- see `widgets.dialogs.
    OverlayDialog` for the shared overlay mechanics (blur, disable, the
    fade+zoom open/close animation, Escape) and the module docstring above
    for what THIS dialog's own body is for.
    """

    run = Signal()
    cancelled = Signal()

    def __init__(self, t, conditions=(), *, content=None, parent=None, factor_names=()):
        super().__init__(t, content, parent=parent)
        self.t = t

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        # NOT `ty.ui("lg", "semibold")` -- the registry's own `DialogTitle`
        # class (`text-lg leading-none font-semibold`, 18px/600) is what
        # that would match literally, but MEASURED against a real render of
        # ui.shadcn.com's own Dialog, that read ~26% taller than the
        # reference (24px against 19px of glyph span) and visibly heavier --
        # more than the ~19% gap this project's own subtitle already carries
        # against the reference's description text at the identical nominal
        # `text-sm`, which nobody has flagged and is left alone (likely just
        # Inter's own cap-height running larger than whatever face the
        # reference actually uses, not a bug). `base` (16px) targets the
        # title-specific remainder of that gap.
        # "Configure SQD", as its siblings are titled ("Configure HMM", "Configure kinetics", "Configure jump
        # distances"), in sentence case like every label under it: "Compute Configuration" until 1 Nov 2026, the one
        # Title Case line in the dialog (the third look at the whole dialog)
        head.addWidget(label("Configure SQD", ty.ui("base", "semibold"),
                             t.foreground))
        head.addStretch(1)
        self._close_btn = CloseButton(t)
        self._close_btn.clicked.connect(self._on_cancel)
        head.addWidget(self._close_btn)
        self.card_lay.addLayout(head)
        # WORD-WRAPPED: unwrapped a longer sentence measured 414px wide, which
        # would itself become the card's width floor for a short-named
        # project -- exactly the fixed-width trap the conditions table just
        # got out of. The siblings' sentence since 1 Nov 2026 (it read
        # "Configure the SQD fit, then Run. Nothing changes until you do.").
        subtitle = label(
            "Nothing runs until you press Run.",
            ty.ui("sm"), t.muted_foreground)
        subtitle.setWordWrap(True)
        self.card_lay.addWidget(subtitle)

        self.compute = ComputePanel(t, conditions, factor_names=factor_names)
        self.compute.pages_resizing.connect(self._resync_after_resize)
        self.card_lay.addWidget(self.compute)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        cancel_btn = Button(t, "Cancel", variant="outline")
        cancel_btn.clicked.connect(self._on_cancel)
        self.run_btn = Button(t, "Run", variant="default")
        self.run_btn.clicked.connect(self._on_run)
        footer.addWidget(cancel_btn)
        footer.addWidget(self.run_btn)
        self.card_lay.addLayout(footer)

    def set_conditions(self, conditions) -> None:
        self.compute.set_conditions(conditions)

    def open(self) -> None:
        """Every open starts on `General` -- see `ComputePanel.show_general` -- with the card fitted to its window
        (`_fit_window`)."""
        self.compute.show_general()
        self._fit_window()
        super().open()

    #: The room kept between the card and its window's top and bottom edges
    _WINDOW_ROOM = 8

    def _fit_window(self) -> None:
        """THE CARD FITS THE WINDOW IT OPENS IN (1 Nov 2026, the night): the pages may take the window's height less
        the card's own chrome (title, subtitle, tabs, footer, margins) and `_WINDOW_ROOM` above and below
        (`ComputePanel.set_pages_room`), so the Fits page scrolls its rows rather than push the card past the window
        -- the Force same D across on top took a line the laptop's card did not have (575 px on Fits, 598 with its
        line, in a maximised window of about 550). Without a parent window nothing is limited."""
        window = self.parentWidget()
        if window is None or window.height() <= 0:
            self.compute.set_pages_room(None)
            return
        self._card.layout().activate()
        chrome = self._card.sizeHint().height() - self.compute._pages.height()
        self.compute.set_pages_room(window.height() - 2 * self._WINDOW_ROOM - chrome)

    def eventFilter(self, obj, event):
        """The window resized while the card is open: the card fits it again (`_fit_window`)."""
        handled = super().eventFilter(obj, event)
        if obj is self.parentWidget() and event.type() == QEvent.Resize and self.isVisible():
            self._fit_window()
        return handled

    def _activate_content_layout(self) -> None:
        self.compute.layout().activate()

    def _on_run(self) -> None:
        self._acted = True
        self.run.emit()
        self._dismiss()

    def _on_cancel(self) -> None:
        self._dismiss()

    def _on_dismiss_without_action(self) -> None:
        self.cancelled.emit()
