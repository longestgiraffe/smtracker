# SMTracker 3

Single-molecule tracking analysis for bacteria: from raw microscope movies to the fractions of molecules in each
diffusive state, compared across conditions with error bars over independent replicates.

SMTracker 3 is a desktop program in four stages, each a window of its own:

- **Image preparation and tracking**: raw Nikon `.nd2` and MetaMorph `.stk` movies, cropped and calibrated, then
  tracked with TrackMate (Fiji) or u-track (MATLAB), one condition at a time or as an overnight batch.
- **Cell outlines**: a classical segmenter, Cellpose-SAM on request, and an editor to correct outlines by hand.
- **Analysis**: quality control, square-displacement (SQD) fractions, jump distances, apparent diffusion, MSD, a
  hidden Markov model with switching rates, spatial maps, kinetics and clustering, each condition beside the others,
  with figures and their source data for export. An **SMTracker 2.0 view** runs the earlier program's methods on the
  same data.
- **Simulated data**: movies and tracks with a known truth, to check what an analysis can recover.

## Download

**Windows**: the installer `SMTracker-<version>-<commit>-setup.exe` on the
[Releases](https://github.com/longestgiraffe/smtracker/releases) page. It installs for one user without
administrator rights, and offers the segmentation pack (PyTorch and Cellpose, downloaded on request) and the Fiji
download page. A zip of the same program is there for computers where nothing may be installed. The program is not
signed, so Windows first says "Windows protected your PC": choose More info, then Run anyway.

**macOS**: build it on the Mac from source (below) until a disk image is published.

Once installed, SMTracker looks for a newer release once a day and asks before it installs anything. Options turns
the check off.

**Tracking** needs Fiji with TrackMate, or MATLAB with u-track, installed separately.

## Manual

[longestgiraffe.github.io/smtracker-site](https://longestgiraffe.github.io/smtracker-site/docs). The same text ships
with the program ([docs/MANUAL.md](docs/MANUAL.md)): every window's **Help** opens it, online or offline.

## Built on SMTracker and SMTracker 2.0

SMTracker 3 reimplements in Python the analyses of SMTracker and SMTracker 2.0, MATLAB programs from Marburg, and goes
beyond them. On SMTracker 2.0's reference data it reproduces the SQD fit to three significant figures, and the
fractions of its apparent-diffusion (gamma) fit to within 1e-4 percentage points. Where the earlier programs have a
known defect, SMTracker 3 does not repeat it.

- Rösch TC, Oviedo-Bocanegra LM, Fritz G, Graumann PL (2018). SMTracker: a tool for quantitative analysis,
  exploration and visualization of single-molecule tracking data reveals highly dynamic binding of *B. subtilis*
  global repressor AbrB throughout the genome. *Sci Rep* 8:15747.
  [doi:10.1038/s41598-018-33842-9](https://doi.org/10.1038/s41598-018-33842-9)
- Oviedo-Bocanegra LM, Hinrichs R, Rotter DAO, Dersch S, Graumann PL (2021). Single molecule/particle tracking
  analysis program SMTracker 2.0 reveals different dynamics of proteins within the RNA degradosome complex in
  *Bacillus subtilis*. *Nucleic Acids Res* 49:e112.
  [doi:10.1093/nar/gkab696](https://doi.org/10.1093/nar/gkab696)

## Citing SMTracker 3

A paper describing SMTracker 3 is in preparation. Until it appears, please cite this repository with the version you
used, which the title bar shows and `smtracker --version` prints, with the build.

## Building from source

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate            # on macOS: source .venv/bin/activate
pip install -e ".[ui,raw,engines]"
smtracker
```

The standalone program and its installer are made with `packaging/build.py` (PyInstaller; on Windows the installer
needs Inno Setup 6):

```
pip install -e ".[ui,raw,engines,packaging]"
python packaging/build.py --zip --installer
```

The build checks itself before it is zipped: a self-test against stored numbers, the bundled readers, the licences
and the source beside the program.

## Licence

GPL-3.0-or-later (`GPL-License.txt`), as SMTracker 2.0, from which parts of the code are ported. The packaged program
carries this licence, the licences of every library inside it (Qt under the LGPL-3.0 through PySide6) and the source
of the commit it was built from.
