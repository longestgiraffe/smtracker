"""The Clustering section's Run dialog (7 Nov 2026; the section had two switches on its page and no dialog, unlike every
other analysis section). SQD's frame (`run_dialog.RunDialog`, its General page only): lags, feature scale, each condition
alone or shared clusters, k (the data chooses up to a largest, or set by hand for shared clusters), the chance the index is
scored against, the SQD-mixture null's localisation error and populations, and the chance draws. THIN QT: every rule is
`clustering_view.section_config` / `run_problem`; a refusal is said above Run and nothing runs. Transactional like every Run
dialog here: Cancel and the corner X discard the draft. Seeded from the last Run."""
from __future__ import annotations

from ...clustering_null import NULL_LABELS, NULL_QUESTIONS, NULLS
from ...clustering_view import (CLUSTER_FEATURE_LABELS, CLUSTER_FEATURES, CLUSTER_LAGS, DRAW_CHOICES, K_MAX_CHOICES,
                                MODE_LABELS, MODES, SHARED_K_CHOICES, config_args, run_problem, section_config)
from ..theme import typography as ty
from ..widgets.controls import NumberField, Segmented
from ..widgets.primitives import label
from .run_dialog import RunDialog
from .toolbar import rem

__all__ = ["ClusteringConfigDialog"]

_POPULATIONS = (None, 1, 2, 3)


class ClusteringConfigDialog(RunDialog):
    """Every Clustering setting, one line each with its help under it."""

    def __init__(self, t, n_conditions: int, *, seed: dict | None = None, content=None, parent=None):
        super().__init__(t, "Configure clustering", "Nothing runs until you press Run.", content=content,
                         parent=parent)
        self.tabs.hide()
        self._n_conditions = int(n_conditions)
        try:
            seed = section_config(**config_args(seed or {}))
        except ValueError:
            seed = section_config()
        self._hint = self.hint
        lag = CLUSTER_LAGS.index(seed["n_lags"])
        self._lags = Segmented(t, [str(n) for n in CLUSTER_LAGS], current=lag, size="sm")
        self._feature = Segmented(t, [CLUSTER_FEATURE_LABELS[f] for f in CLUSTER_FEATURES],
                                  current=CLUSTER_FEATURES.index(seed["feature"]), size="sm")
        self._mode = Segmented(t, [MODE_LABELS[m] for m in MODES], current=MODES.index(seed.get("mode", "condition")),
                               size="sm")
        k = seed.get("k")
        self._k = Segmented(t, ["Data chooses"] + [str(v) for v in SHARED_K_CHOICES],
                            current=0 if k is None else SHARED_K_CHOICES.index(k) + 1, size="sm")
        self._k_max = Segmented(t, [str(v) for v in K_MAX_CHOICES],
                                current=K_MAX_CHOICES.index(seed.get("k_max", 4)), size="sm")
        self._null = Segmented(t, [NULL_LABELS[n] for n in NULLS], current=NULLS.index(seed.get("null", "free")),
                               size="sm", tips=[NULL_QUESTIONS[n] for n in NULLS])
        sigma = seed.get("sigma_nm")
        # 7 Nov 2026: Per spot first, the standard (each track the error of its own spots, calibrated)
        sigma_mode = 2 if sigma is not None else (1 if seed.get("per_spot") is False else 0)
        self._sigma_mode = Segmented(t, ["Per spot", "The acquisition's", "Set"], current=sigma_mode, size="sm")
        self._sigma = NumberField(t, 0, 500, int(30 if sigma is None else sigma), width=rem(7))
        self._sigma_unit = label("nm", ty.ui("xs"), t.muted_foreground)
        self._pop = Segmented(t, ["Best (BIC)", "1", "2", "3"], current=_POPULATIONS.index(seed.get("sqd_pop")),
                              size="sm")
        self._draws = Segmented(t, [str(v) for v in DRAW_CHOICES], current=DRAW_CHOICES.index(seed.get("draws", 20)),
                                size="sm")
        self.helpers = {
            "lags": self.add_setting(
                "Lags", self._lags, "How many lags each track's MSD curve has; a track needs all of them.",
                "More lags keep fewer tracks (a track must reach the last one) and describe the curve's shape "
                "better. 4 is SMTracker 2's."),
            "feature": self.add_setting(
                "Feature scale", self._feature, "MSD (SMTracker 2's), or log MSD, which damps the noisy top.",
                "A per-track MSD from a handful of steps is very noisy, with a long upper tail. On the raw scale the "
                "tail dominates the index; the log scale carries the shape. The scale used is shown."),
            "mode": self.add_setting(
                "Clusters", self._mode,
                "Each condition alone, or shared clusters: one set of clusters for every condition.",
                "Alone, every condition gets its own clusters, so their shares cannot be compared. Shared clusters "
                "are fitted once on the pooled tracks, numbered from the slowest, so shares can be compared across "
                "conditions (Welch on the replicates)."),
            "k": self.add_setting(
                "Number of clusters (k)", self._k,
                "The data chooses k, or you set it (shared clusters only).",
                "Chosen: the k with the largest excess over chance in chance SDs. Set by hand, no test is made of "
                "whether the curves hold structure beyond chance."),
            "k_max": self.add_setting(
                "Largest k the data may choose", self._k_max, "k is tried from 2 up to this.",
                "More k costs more chance draws; a larger k also needs more tracks per cluster."),
            "null": self.add_setting(
                "Chance", self._null,
                "What the index is scored against: tracks with no clusters in them.",
                "The SQD mixture: tracks drawn from the condition's fitted SQD populations; beating it means "
                "clustering finds more than they explain. One free population: a single D; beating it only means "
                "more than one D."),
            "sigma": self.add_setting(
                "Localisation error", self._sigma_mode,
                "The error the SQD mixture is fitted and drawn with.",
                "Per spot: the mixture is fitted with each step's own error (as SQD's Per spot), and each chance "
                "track copies a real track's length and spot errors. A condition without per-spot precision uses "
                "the acquisition's stated precision, or none. Set: one value for every condition.",
                extra=(self._sigma, self._sigma_unit)),
            "pop": self.add_setting(
                "SQD populations", self._pop, "How many populations the mixture has.",
                "Best (BIC) fits 1 to 3 populations at lag 1 and takes the lowest BIC; or set a count."),
            "draws": self.add_setting(
                "Chance draws", self._draws, "Sets of chance tracks the index is compared with.",
                "20 is the fewest whose spread can be quoted; more draws cost time and tighten the spread."),
        }
        for control in (self._lags, self._feature, self._mode, self._k, self._k_max, self._null, self._sigma_mode,
                        self._pop, self._draws):
            control.changed.connect(lambda _i: self._sync())
        self._sigma.changed.connect(lambda _v: self._sync())
        self._sync()

    # -- the rules (headless in `clustering_view`) ---------------------------------------------------------------- #
    def config(self) -> dict:
        """The draft as `clustering_view.section_config` makes it. RAISES ValueError on a draft it refuses."""
        mode = MODES[self._mode.current()]
        shared = mode == "shared"
        index = self._k.current()
        return section_config(
            CLUSTER_LAGS[self._lags.current()], CLUSTER_FEATURES[self._feature.current()], mode,
            None if (index <= 0 or not shared) else SHARED_K_CHOICES[index - 1], NULLS[self._null.current()],
            None if self._sigma_mode.current() != 2 else float(self._sigma.value()),
            _POPULATIONS[self._pop.current()], K_MAX_CHOICES[self._k_max.current()],
            DRAW_CHOICES[self._draws.current()], per_spot=self._sigma_mode.current() == 0)

    def _sync(self) -> None:
        shared = MODES[self._mode.current()] == "shared"
        sqd = NULLS[self._null.current()] == "sqd"
        fixed = shared and self._k.current() > 0
        self._k.setEnabled(shared)
        self._k_max.setEnabled(not fixed)
        for key in ("sigma", "pop"):
            self.helpers[key].setVisible(sqd)
        self._sigma_mode.setVisible(sqd)
        self._pop.setVisible(sqd)
        self._sigma.setVisible(sqd and self._sigma_mode.current() == 2)
        self._sigma_unit.setVisible(sqd and self._sigma_mode.current() == 2)
        self._draws.setEnabled(not fixed)
        try:
            problem = run_problem(self.config(), self._n_conditions)
        except ValueError as exc:
            problem = str(exc)
        self._hint.setText(problem)
        self._run_btn.setEnabled(not problem)

    def _on_run(self) -> None:
        try:
            config = self.config()
        except ValueError:
            return
        if run_problem(config, self._n_conditions):
            return
        self._acted = True
        self.run.emit(config)
        self._dismiss()

