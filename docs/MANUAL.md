# SMTracker 3 user manual

This is the manual on the SMTracker website, https://longestgiraffe.github.io/smtracker-site/docs, copied into the program so that **Help** works without a network. The website may be newer. From the website, 27 Nov 2026 (version 7c68068), copied by `scripts/manual_from_site.py`.

## 1. Get started

SMTracker takes you from raw single-molecule movies to diffusive-state fractions for each of your conditions. A *condition* is one folder of movies that you compare against the others, usually one strain under one treatment. You bring the movies; SMTracker helps you crop and calibrate them, track the molecules, outline the cells, and compare the conditions, with error bars over your replicates.

This manual assumes you already know single-particle tracking. It explains how SMTracker does each step, which settings matter, and what to check before you trust a number.

#### What SMTracker measures

The main result is a *fraction*: the share of molecules in each diffusive state, such as a slow, bound-like population and a faster, freely diffusing one. SMTracker shows how these fractions shift between your conditions, for example between two carbon sources or a wild type and a mutant.

Each fitted state also has a diffusion coefficient (D). Treat it as a bound rather than a physical value:

- The slow state sits near the *localization floor*: the apparent D that an immobile molecule shows from the uncertainty in its position alone.
- The fast state sits near the *detection ceiling*: molecules that move farther than the tracker's linking radius in one frame are lost, so the fastest D you can measure is capped.

> **What to report.** Report the fractions. If you quote a D, give the frame interval, the pixel size and the linking radius with it, and don't present it as a precise coefficient.

#### Three stages, each in its own window

When SMTracker opens, it asks **Where would you like to start?** Click a card to open that stage in its own window. You'll usually go through the three stages in order, but you can start at any of them.

*A diagram here is on the website: https://longestgiraffe.github.io/smtracker-site/docs*

| Stage | What it does | What it needs |
| --- | --- | --- |
| [Image preparation + Tracking](#2-image-preparation-and-tracking) | Crops each movie to the frames where single molecules are visible, saves the calibration, then detects and links spots into tracks | Fiji with TrackMate, or MATLAB with u-track |
| [Cell outlines](#3-cell-outlines) | Finds the cells automatically, then lets you correct the outlines by hand | Nothing else for the classical segmenter and the editor. The Omnipose models need SMTracker run from source ([Install and open SMTracker](#install-and-open-smtracker)). |
| [Analysis](#4-analysis) | Loads the tracks, checks them for bias, fits the populations and makes figures | Tracks and cell outlines for every movie |

The analysis keeps only tracks that lie inside a cell outline. That's how SMTracker knows which molecules belong to which cell. So even if your movies are already tracked, outline the cells before you open the analysis: a movie without outlines is left out.

The first window has a fourth card, **Simulated data**. It builds a data set with known populations, diffusion and switching, either as tracks for the analysis or as movies to track, so you can compare SMTracker's results with the truth. Use it to try SMTracker before you have movies of your own.

#### Controls you'll see in every stage

The three stage windows share the same three controls. The first window, where you only pick a stage, doesn't have them, and neither does Simulated data, apart from **Back to menu**. The first window's moon icon switches SMTracker between light and dark, and every window opens in the one chosen.

| Control | What it does |
| --- | --- |
| **Help** | Offers the manual on the website or the copy that came with SMTracker (no network needed), at the chapter for the stage you're in, and **Check for updates** |
| **Options** | Sets the *standards*: the values a new condition starts from |
| **Back to menu** | Returns to the first window |

A value you change on a page always wins over its standard. When you save new options, SMTracker updates only the values you haven't changed by hand.

> **In the app.** In the preparation and outlines windows, and on the analysis's start screen, these sit in the header. Once a project is open in the analysis, **Help** is the info icon at the bottom of the right-hand rail, and **Back to menu** is in the project menu at the top of the sidebar.

The app starts in the light theme. To switch, click the moon icon at the bottom of the analysis window; SMTracker remembers your choice.

#### Where to go next

If this is your first time, read these two pages in order:

1. [Install and open SMTracker](#install-and-open-smtracker): get the app running, and connect Fiji or MATLAB for tracking.
2. [Your first project](#your-first-project): one folder from raw movies to fractions, step by step.

To prepare your data, read [Organizing your folders](#organizing-your-folders) and [How replicates are counted](#how-replicates-are-counted). If your tracks come from another program, see [Tracks from other tools](#tracks-from-other-tools) and [Frame interval and pixel size](#frame-interval-and-pixel-size).

Each stage has its own chapter:

- [Image preparation + Tracking](#2-image-preparation-and-tracking): Crop, calibrate, then detect and link in TrackMate or u-track
- [Cell outlines](#3-cell-outlines): Find the cells, then correct them by hand
- [Analysis](#4-analysis): QC, fractions and figures from the same tracks

### Install and open SMTracker

*Get SMTracker running on Windows or macOS, check that it works, and connect Fiji or MATLAB for tracking.*

SMTracker runs on Windows and macOS, and you don't need Python. It comes with an installer for Windows and for Macs with Apple silicon, and a zip of the same program is there for computers where nothing may be installed. The only extra programs you need are the ones for tracking, Fiji or MATLAB.

> **On Linux?** There's no packaged version for Linux yet. Run SMTracker from its source code instead: see [Run from source](#run-from-source) at the end of this page.

#### What your computer needs

| | Needed | Better |
| --- | --- | --- |
| System | Windows 10 or 11 (64-bit), or macOS 12.3 or later on Apple silicon | |
| Processor | 2 cores | 4 or more: the analyses use every core but one |
| Memory | 8 GB | 16 GB |
| Disk | 1 GB for the program, plus room for each project's cache | |
| Screen | 1366 × 768 | 1920 × 1080 |
| Tracking | Fiji for TrackMate, or MATLAB with u-track (each u-track session takes up to about 1.5 GB of memory) | |
| Model segmentation (optional) | The [segmentation pack](#automatic-cell-outlines): 2.5 GB to download and 4.1 GB on disk with GPU support, 0.2 GB and 0.7 GB without | An NVIDIA graphics card. Training Cellpose-SAM needs one with about 6 GB of memory or more |

#### Get SMTracker running

Download it from the [newest release](https://github.com/longestgiraffe/smtracker/releases/latest) on GitHub.

**Windows**

##### Run the installer

Download `SMTracker-<version>-<build>-setup.exe` from the release and double-click it. Windows may first show **Windows protected your PC**: SMTracker isn't signed with a paid certificate, so Windows doesn't recognize it yet. Click **More info**, then **Run anyway**.

The installer needs no administrator rights: it installs SMTracker for you alone, in `%LOCALAPPDATA%\Programs\SMTracker`.

##### Choose what else to get

After the license, the installer offers three things, none of them ticked: a desktop shortcut, the [segmentation pack](#automatic-cell-outlines) for Cellpose-SAM (downloaded in a window that stays open until you close it), and Fiji's download page, for tracking with TrackMate. You can get both of the last two later as well.

##### Start it

Start SMTracker from the installer's last page or from the Start menu. **SMTracker (with console)** in the Start menu opens the same program with a console window beside it that shows its messages. Closing that console closes SMTracker.

##### Check that it opened

The first window asks **Where would you like to start?** When you see it, SMTracker is running. To go on, follow [Your first project](#your-first-project).

> **Where nothing may be installed** The release also has a zip of the same program. Unzip it to a short path on your own disk, for example `C:\SMTracker`: Windows limits a file's full path to 260 characters, and the longest file inside uses about 140 of them. Then double-click `smtracker.exe`, or `smtracker-console.exe` to see its messages.

**macOS**

The Mac package is for Macs with Apple silicon (M1 or later) and macOS 12.3 or later. On an Intel Mac, [run SMTracker from source](#run-from-source).

##### Open the installer

Download `SMTracker-<version>-<build>-macos-arm64.pkg` from the release and double-click it. SMTracker isn't signed with a paid Apple certificate, so macOS first says it can't check it for malicious software. Click **Done**, open **System Settings**, then **Privacy & Security**, scroll down to the message about SMTracker and click **Open Anyway**. You only need to do this once.

##### Install it

The installer shows a welcome page, a short read-me and the license. It puts SMTracker in **Applications** for all users (it asks for your password), or in the **Applications** folder of your home folder for you alone (no password).

Under **Customize** it offers three things, none of them ticked: the [segmentation pack](#automatic-cell-outlines) for Cellpose-SAM (downloaded in a Terminal window that stays open until you press `Return`; it needs macOS 15), Fiji's download page, for tracking with TrackMate, and opening SMTracker at the end. You can get the first two later as well.

##### Start it

Start SMTracker from Applications. It opens no Terminal window.

If macOS refuses it again, or asks about parts of the program, remove the download mark once: open **Terminal**, type `xattr -cr /Applications/SMTracker.app` (or `~/Applications/SMTracker.app` if you installed it for you alone), press `Return`, and start SMTracker again.

##### Check that it opened

The first window asks **Where would you like to start?** When you see it, SMTracker is running. To go on, follow [Your first project](#your-first-project).

##### Where SMTracker keeps its files

The installer puts the program in `%LOCALAPPDATA%\Programs\SMTracker` on Windows, and on a Mac in **Applications** or your home folder's **Applications**; a zip copy is wherever you unzipped it. Besides the program, SMTracker keeps:

| What | Windows | macOS |
| --- | --- | --- |
| Your options, the Fiji, MATLAB and u-track folders it found, saved folder designs, presets, trained segmentation models, and a log | `%APPDATA%\SMTracker` | `~/.config/smtracker` |
| Caches, so projects reopen quickly, and the list of recent projects | `%LOCALAPPDATA%\smtracker` | `~/Library/Caches/smtracker` |
| Everything it makes for a project: ranges, projections, tracks, outlines, groupings and caches | In each condition folder, beside your data ([What SMTracker writes](#what-smtracker-writes)) | The same |

To remove SMTracker, uninstall it in Windows' **Installed apps**, drag SMTracker from Applications to the Trash on a Mac, or delete the folder of a zip copy. The uninstaller leaves your settings, caches and a downloaded segmentation pack: remove the pack first with **Segmentation pack** in the outlines window, then delete the two folders in the table. Deleting the caches also clears the list of recent projects.

##### If something seems wrong

The first window has **Report a problem…** at the bottom left. It packs the log, the version and your system details into one zip. **Report on GitHub** then opens a new public report on SMTracker's GitHub page with the version and the end of the log filled in: read it, delete any folder names you'd rather not show, drag the zip in, and submit it with your GitHub account. Nothing is sent until you do. You can also send the zip to whoever maintains SMTracker in your lab.

You can also run a quick self-test:

##### Open a terminal

On Windows, click **Start**, type `cmd` and press `Enter`. On a Mac, open **Terminal** from Applications › Utilities.

##### Run the self-test

Go to the SMTracker folder: type `cd %LOCALAPPDATA%\Programs\SMTracker` on Windows (or the folder of a zip copy) or `cd /Applications/SMTracker.app/Contents/MacOS` on a Mac (`~/Applications/...` if you installed it for you alone), and press `Enter`. Then type `smtracker-console.exe --selftest` on Windows or `./smtracker --selftest` on a Mac, and press `Enter`. The output ends with `analysis self-test: OK` when all is well.

#### Connect a tracker

You only need this if you track your movies in SMTracker. If you already have tracks, skip to [Automatic cell outlines](#automatic-cell-outlines): the analysis reads u-track and TrackMate output directly, and other trackers' tables as described in [Tracks from other tools](#tracks-from-other-tools).

SMTracker doesn't include a tracker. It runs TrackMate or u-track in the background, and you choose which with **Tracker** on Detect & link, Track or Batch.

**TrackMate (Fiji)**

TrackMate is free and comes with Fiji.

1. Download Fiji from [the Fiji download page](https://imagej.net/software/fiji/downloads) and unzip it, for example to `C:\Fiji`.
2. In SMTracker's preparation window, click **Track** in the sidebar. The **Fiji** card at the top says whether SMTracker found Fiji in the usual folders. If it didn't, click **Choose Fiji folder…** and choose the Fiji folder.

SMTracker remembers the folder. Both the current Fiji (which starts with `fiji-windows-x64.exe`) and older versions (with `ImageJ-win64.exe`) work.

**u-track (MATLAB)**

u-track needs MATLAB with its Statistics, Optimization and Image Processing toolboxes, plus the u-track package from the Danuser lab.

1. Install MATLAB with those toolboxes, and download and unzip u-track.
2. In SMTracker's preparation window, choose *u-track (MATLAB)* in **Tracker**, then click **Track** in the sidebar. SMTracker starts looking for MATLAB and u-track once u-track is the chosen tracker, and the **MATLAB and u-track** card says what it found. If it found nothing, click **Choose matlab.exe…** and choose `matlab.exe` in a MATLAB release's `bin` folder (on a Mac, the `matlab` program in the MATLAB app's `bin` folder), then **Choose u-track folder…** and choose the folder that holds `software/@UTrackPackage`.

SMTracker remembers what you choose. The first u-track run starts MATLAB once to check the toolboxes, which takes about 30 seconds.

With **Movies at once**, SMTracker can track up to four movies side by side, with either tracker. With u-track, each one is a MATLAB session of its own, with its own share of memory.

Each u-track session holds its own MATLAB, which takes up to about 1.5 GB of memory.

> **Check this.** Track every condition you want to compare with the same tracker. On the same movies, TrackMate and u-track differ by about 1 percentage point in the fractions and by 10 to 18 % in D.

#### Automatic cell outlines

SMTracker comes with the classical segmenter, which is fast but weaker on some data, and with **Edit outlines** to draw and correct outlines by hand. Segmenting with the Cellpose-SAM model, and training your own, need PyTorch and Cellpose. They're not part of the program, so most people never download them; SMTracker downloads them for you the first time you want them.

1. In the **Cell outlines** window, click **Get segmentation models…** on **Segment**, or **Segmentation pack** in the header.
2. Choose *With GPU support (CUDA)* for an NVIDIA graphics card (2.5 GB to download, 4.1 GB on disk), or *CPU only* (0.2 GB and 0.7 GB). SMTracker picks the one that fits this computer and says why.
3. Pick a folder, then click **Download**. If the program's drive is too small, SMTracker proposes the next drive with room. Click **No** to go on without them.

The files come from pytorch.org and pypi.org, and each one is checked against a fixed checksum before it's used. Cellpose-SAM's own model (1.2 GB) downloads the first time you segment with it. On a GeForce GTX 1650, Cellpose-SAM outlined a field in 1.5 s; on a laptop's processor alone, in about 3.5 minutes.

The Omnipose models run only when you run SMTracker from source with Omnipose installed: Omnipose's license doesn't allow it to be shipped with SMTracker.

#### Update SMTracker

SMTracker looks for a newer release once a day when it starts, and shows it on the first window. **Download and install** downloads the installer, checks it against the release's checksums, closes SMTracker and updates it in place; your settings and the segmentation pack stay. Nothing is installed until you click it. On a Mac it opens the release page instead, where you download the new installer. **Help** › **Check for updates** looks at once, and **Check for updates at start** in Options turns the daily look off.

With a zip copy, unzip the new version into a new folder, for example `C:\SMTracker-new` or `~/SMTracker-new`, and start it from there. Your settings live outside the program folder (`%APPDATA%\SMTracker` on Windows, `~/.config/smtracker` on a Mac), so the new version finds them. Once it runs well, delete the old folder.

#### Run from source

On Linux, or to use the Omnipose models, run SMTracker from its source code. You need Python 3.10 or newer and the SMTracker source: clone it from [GitHub](https://github.com/longestgiraffe/smtracker) with `git clone https://github.com/longestgiraffe/smtracker`, or unzip the copy in a packaged SMTracker's `source` folder. Open a terminal in the source folder, and run:

`Terminal (macOS or Linux)`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui,raw]"
smtracker
```

On Windows, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\activate`; the other two commands are the same. The `raw` part adds the readers for `.nd2`, `.stk` and TIFF movies, and `ui` the rest of the interface.

### Your first project

*One walk through all three stages, from a folder of raw movies to the fractions of each condition. Plan for an afternoon, most of it waiting for tracking.*

Take one project from raw movies to fractions, through all three stages, to see how they fit together. Each step links to the chapter that explains it. You don't need those chapters first: the defaults give sensible results on a first run.

#### Before you start

You'll need:

- **SMTracker running**, and Fiji or MATLAB connected ([Install and open SMTracker](#install-and-open-smtracker)).
- **At least two conditions** to compare. A *condition* is one folder of movies (`.nd2`, `.stk` or TIFF stacks) with their phase-contrast snapshots (`.tif`) inside. [Organizing your folders](#organizing-your-folders) shows the layouts SMTracker recognizes.
- **Three or more replicates** per condition, if you want error bars you can trust. A *replicate* is an independent culture, and the error bars are computed over replicates. Fewer still work, but QC flags them.
- **The pixel size** of your camera setup, in nm. SMTracker never reads it from the `.nd2`, because the value stored there can be wrong.

```
ProteinA/
  PdhA_glucose/     a condition
    series001.nd2   a movie
    series001.tif   its phase snapshot
    …
  PdhA_malate/      a condition
    …
```

*A small project: two conditions, each with movies and snapshots side by side.*

The preparation window never moves, renames or changes your movies. Everything SMTracker creates goes into new files and folders beside them ([What SMTracker writes](#what-smtracker-writes)).

#### Part 1: Prepare and track

First, turn the raw movies into tracks. Work through the pages from top to bottom. Each of the first three ends with a button naming the next page, which lights up when the page's work is done.

##### Open the folder

In the first window, click the **Image preparation + Tracking** card. Click **Open folder…** and choose the folder above your conditions (`ProteinA` in the example). The sidebar lists the pages and, below them, your conditions with their progress.

##### Crop the movies

On **Movies & crop**, wait until the **Reading movies** card finishes, then click **Propose for all conditions**. Each movie gets a *range*: the frames it keeps for tracking. SMTracker leaves out the early frames, where the field is still bleaching down to single molecules.

Click through a few movies. The kept range should start where the frame mean has flattened out. If one doesn't, drag the range's edges on the chart. Nothing is written yet.

For the start rules and how to fix one movie, see [Movies & crop](#movies-crop).

##### Enter the calibration and save

On **Calibration & save**, each condition has one row. Type the **Pixel size** in nm; the **Frame interval** comes from the movie file. Choose your **Fluorophore** from the list: one from SMTracker's list fills in its emission, which sets the expected spot size. If every condition was imaged the same way, fill in the first row and click **Copy first row to all**.

Then click **Save all**. This writes the ranges, the calibration and a projection of each movie (in `sum/`, used later for the outlines).

> **Check this.** Every D scales with the square of the pixel size, and inversely with the frame interval, so a wrong value moves every D. Tracking always uses what was saved, so save again after any change.

For each field and the pixel-size helper, see [Calibration & save](#calibration-save).

##### Set up detection and linking

On **Detect & link**, choose *TrackMate (Fiji)* or *u-track (MATLAB)* in **Tracker**. For TrackMate, click **Measure all thresholds**: SMTracker measures each movie's noise and sets the detection threshold from it.

The **Linking radius** is the farthest a molecule may move between two frames, in px. It defaults to 5 px and caps the fastest D you can measure; the page shows that D under the table. Leave it, and the other defaults, for a first run.

For what each setting does and when to change it, see [Detect & link](#detect-link).

##### Track

On **Track**, check that the card at the top found Fiji (or MATLAB and u-track). SMTracker looks for them on its own; if it didn't find them, choose them there. Then click **Track all ready conditions**.

This is the slow part: TrackMate needs about 30 seconds per movie. To track several movies side by side, choose a higher number in **Movies at once**. You can use other programs meanwhile; SMTracker tells you when the run ends.

For the tracker card, stopping a run and what it writes, see [Track](#track). To track many folders overnight, see [Batch](#batch).

#### Part 2: Outline the cells

Next, outline the cells, because the analysis only keeps tracks that lie inside an outline. Click **Back to menu**, click the **Cell outlines: Segmentation + Mesh editor** card, then click **Open folder…** and choose the same folder.

##### Segment

On **Segment**, each condition has a row with a **Model**. In the packaged app, choose *Classical, sum image (no GPU, no training)*: the packaged app can't run the Omnipose models, and the classical segmenter outlines the cells on the sum image saved on Calibration & save ([Install and open SMTracker](#install-and-open-smtracker)). If you run SMTracker from source with Omnipose installed, keep the default, *bact_phase_omni*, which reads the phase snapshots and suits most data.

Click **Segment** on a row, or **Segment all ready** to do every condition.

For the models and their settings, see [Segment](#segment).

##### Check and correct the outlines

Click **Edit outlines** in the sidebar and choose a condition. Step through the *fields*, the imaged positions, with the arrow keys, and fix what the model got wrong: `X` splits two cells drawn as one, `M` merges a cell cut in two, `D` draws a missing one, and `Del` removes a wrong one. `Ctrl+S` saves the field on screen; click **Save all** to save every field you've edited.

> **Check this.** The analysis keeps only tracks that lie inside an outline. Outlines drawn too tight drop the free molecules near the membrane and raise the slow fraction, so correct them before you analyze.

For every tool and shortcut, see [Edit outlines](#edit-outlines).

#### Part 3: Analyze

Last, compare the conditions. Click **Back to menu** and click the **Analysis** card.

##### Open the project

Click **Choose folder…** (or drop the folder onto the window) and choose the folder above your conditions again.

If your movies sit two or more folders below the one you opened, **What do the folders mean?** asks you to name each level: a factor (such as protein or medium), a replicate, or a level to skip. **Confirm project** then shows, for each condition, the pixel size, the frame interval, and the movies, cells and tracks kept in each replicate. Check that the replicates are grouped the way you expect, then click **Open workspace**.

For each page of opening a project, see [Opening a project](#opening-a-project) and [How replicates are counted](#how-replicates-are-counted).

##### Run QC before anything else

Click **QC** in the sidebar, then **Run analysis**. The **Is the comparison fair?** table puts your conditions side by side on what can bias a fraction: the tracker, the replicates, the frame interval, track length, yield, how many tracks lie inside cells and near their edges, how well tracks and outlines line up, and how crowded the movies are.

An orange value means "look at this", not "this is wrong". Hover over the column heading to read the reason, and decide whether it matters for your comparison.

For every column, see [QC](#qc).

##### Fit the populations

Click **SQD** in the sidebar, then the Compute mark at the top of the right rail (its hover reads *Run SQD…*). The **Configure SQD** dialog opens; leave the defaults and click **Run**.

SMTracker fits the distribution of squared displacements and recommends a number of populations for each condition, by BIC. If an extra population isn't plausible, for example one sitting at the localization floor, choose another number on the **Fits** tab of the dialog and run again.

The result is a bubble chart: one bubble per population and condition, at its D, sized by its fraction, with error bars over your replicates.

For the settings and how to read the chart, see [SQD](#sqd-populations) and [Reading the numbers](#reading-the-numbers).

##### Export a figure

Each chart has an export icon (its hover reads *Export figure*). Click it to save the chart as a publication figure, with its source data beside it. To combine several charts into one figure, use the Composer.

When you share a result, report the fractions, and give the frame interval, the pixel size and the linking radius with any D. Treat each D as a bound, not a precise coefficient ([Reading the numbers](#reading-the-numbers)).

For the export options and the Composer, see [Figures](#figures).

#### Next

Your first project is done. From here:

- If something was refused or a movie was left out, see [When something is refused](#when-something-is-refused).
- If your folders look different from the example, see [Organizing your folders](#organizing-your-folders) and [How replicates are counted](#how-replicates-are-counted).
- If you tracked your movies in another program, see [Tracks from other tools](#tracks-from-other-tools).

### Organizing your folders

*Arrange your movies so SMTracker finds every condition and pairs each movie with its phase snapshot.*

Arrange your folders so SMTracker finds every condition. SMTracker doesn't ask you to import or register anything. It reads your data straight from the folders on disk, so the way you arrange them is how SMTracker understands your experiment. The rules are short.

#### A condition is a folder

A *condition* is one folder of movies that you compare against the others, for example protein A in glucose. It's usually one strain under one treatment, but SMTracker only knows it as a folder. The folder holds all of that condition's movies, from every replicate:

```
PdhA_LB/
  series001.nd2   a movie
  series001.tif   its phase snapshot
  series002.nd2
  series002.tif
  …
```

*One condition: all its movies, from every replicate.*

When you open a folder in SMTracker, you can choose either:

- one condition folder, to work on that condition alone, or
- the folder above several conditions, to see all of them at once. This is what you'll usually want, since the analysis compares conditions.

To keep each replicate in its own sub-folder, see [How replicates are counted](#how-replicates-are-counted).

#### Movies and their phase snapshots

Each movie usually comes with a *phase-contrast snapshot* of the same field, which SMTracker uses to find the cells. SMTracker reads two layouts: movie and snapshot side by side, or movies in `raw/` and snapshots in `phase/`.

```
PdhA_LB/
  series001.nd2   or .stk, or a TIFF stack
  series001.tif   phase snapshot
```

*Straight off the microscope: movie and snapshot side by side.*
```
PdhA_LB/
  raw/
    series001.nd2
  phase/
    series001.tif
```

*Filed: movies in raw/, snapshots in phase/.*

A TIFF stack works as a movie too. A single-page `.tif` with the same name as a movie is read as its snapshot, not as a movie.

A snapshot is paired with its movie by name, so keep the two names identical apart from the extension: `series001.nd2` goes with `series001.tif`.

> **Check this.** Don't mix the two layouts in one folder. If some movies sit in `raw/` and newer ones lie loose beside it, the preparation window reads only the loose ones. The condition's line in the sidebar says *half-organised*, and the note on Calibration & save says which movies are read. Move all of the condition's movies into `raw/`, or all of them out of it.

The preparation window never moves or renames your files. The older classic tracking window does: when you save ranges there, it files loose movies into `raw/` and snapshots into `phase/`, without keeping copies.

#### Grouping conditions

As projects grow, you'll probably group conditions, for example one folder per protein:

```
experiments/
  ProteinA/
    PdhA_LB/        a condition
    PdhA_glucose/   a condition
  ProteinB/
    …/              its conditions
```

SMTracker's three windows don't read groups the same way:

| You open | Preparation | Cell outlines | Analysis |
| --- | --- | --- | --- |
| A condition (`PdhA_LB`) | Yes | Yes | Yes |
| A group of conditions (`ProteinA`) | Yes | Yes | Yes |
| A folder of groups (`experiments`) | No: open each group | Yes, unless two groups hold conditions of the same name | Yes |

So in preparation, open one protein's folder at a time. In the analysis, open the top folder to compare every condition.

> **Check this.** The analysis treats `ProteinA` as a group of conditions only when its conditions reuse the same movie names, each starting again at `series001`. If the series run on from one condition to the next (001 to 030, then 031 to 060), the analysis opens `ProteinA` as one condition holding both media. Check the conditions it lists before you go on.

When your movies sit two or more folders below the one you open, the analysis asks **What do the folders mean?**: whether each level is a *factor* that describes the condition (such as protein or medium), a *replicate*, or a level to skip. See [Opening a project](#opening-a-project).

#### Two experiments in one folder

Sometimes one folder holds two experiments with the same conditions, for example exponential and stationary phase. The analysis names each condition after its experiment folder, so they don't clash:

| Folder | Opens as condition |
| --- | --- |
| `exp/free` | `exp free` |
| `exp/yloc` | `exp yloc` |
| `stat/free` | `stat free` |
| `stat/yloc` | `stat yloc` |

The same goes for groups: `ProteinA/LB` and `ProteinB/LB` open in the analysis as `ProteinA LB` and `ProteinB LB`. Renaming a group folder therefore renames its conditions in the analysis.

The Cell outlines window can't tell such conditions apart, and finds nothing to segment. Open each experiment's folder (`exp`, then `stat`) there instead, or give the conditions different names, such as `PdhA_LB` and `PdhB_LB`.

#### Next

- To keep each replicate in its own folder, see [How replicates are counted](#how-replicates-are-counted).
- If you tracked your movies with another program, see [Tracks from other tools](#tracks-from-other-tools).
- To see what SMTracker adds to these folders, see [What SMTracker writes](#what-smtracker-writes).

### How replicates are counted

*Check that SMTracker groups your movies into the right replicates, the independent units behind every error bar, and regroup them when it doesn't.*

A *replicate* is an independent measurement of a condition, usually a separate culture imaged on its own day. By default, the analysis computes its error bars and significance tests over replicates, not over movies or tracks, because movies from the same culture aren't independent of each other. That's why it matters that SMTracker groups your movies into the right replicates.

SMTracker proposes the replicates for you, and **Confirm project** shows them before anything is computed. Check them there: when you open the workspace, they're saved and used from then on.

These rules belong to the analysis. The preparation and outlines windows see each folder that holds movies as a condition of its own.

#### One replicate per imaging day (the default)

The first time you open a condition, SMTracker puts each imaging day's movies into one replicate. It reads the days in one of two ways:

- **By acquisition time**, when every movie records when it was taken, in its raw `.nd2` or `.stk` file in `raw/` or in u-track's movie data. A gap of more than an hour between movies starts a new day, so two sessions on one date are two replicates.
- **By series number** otherwise: the first number in the movie's name, such as 1 in `series001`. Each block of ten is one day: series 001 to 010 is the first, 011 to 020 the second, and so on.

*A diagram here is on the website: https://longestgiraffe.github.io/smtracker-site/docs/replicates*

The time u-track imported a movie doesn't count, because one import can hold movies imaged months apart. The replicates are numbered by their lowest series number, and gaps are fine: series 001 to 008 and 021 to 030 make two replicates.

##### When the proposal is wrong

If you didn't image ten movies per replicate, or a replicate spans two sessions, regroup the movies when you open the project. On **Confirm project**, click the *Groups* view (the layers icon beside the condition tabs). On the **Analysis groups** board, click **Regroup…** to set groups of any number of series at once, with a preview, or drag single series between replicates. See [Opening a project](#opening-a-project).

When you open the workspace, SMTracker saves each condition's grouping as `day_map.json` in its folder, and uses it from then on. So you only regroup once per condition. **Reset to replicates** on the board goes back to the imaging days. Movies you add later are placed by their series block: into that block's replicate, or as a new one. Check the board before you open the workspace again, which saves the new grouping.

#### Replicates in folders

You can also keep each replicate in a folder of its own. The two arrangements below tell SMTracker that the folders are one condition, not several.

On a first open, the board still proposes one replicate per imaging day, as above, so check that its groups match your folders.

##### Replicates as sub-folders

Each replicate sits in a folder of its own, inside the condition's folder. SMTracker reads them as one condition. Check on **Confirm project** that the board's groups match your folders.

```
ProteinA-mNG/
  Glc/          one condition
    Glc1/       replicate 1: series001, series002, …
    Glc2/       replicate 2: series001, series002, …
    Glc3/       replicate 3: series001, series002, …
```

Here every replicate starts its series again at `series001`. The replicates are ordered by their numbers, and the movies are listed as `series001 (Glc1)`, `series001 (Glc2)`, and so on, so you can still tell them apart.

For this to work, the folders must be named as replicates:

- a number, after the condition's name if you like: `1`, `Glc1`, `M2c`, `day 2`, `replicate 2`; or
- the full date they were imaged, which orders them by date: `10.03.2025`, `2025-03-10`.

> **Check this.** If even one folder that holds movies isn't named as a replicate (say `Glc/maltose` beside `Glc/Glc1`), the analysis can't tell that they belong together, and opens them as separate conditions. Rename the odd one out.

If you open the folder above the condition (`ProteinA-mNG` here), the movies sit two levels down, so the analysis first asks **What do the folders mean?**. That's expected: mark the level as the replicate there, or click **Load without a design**. The same goes for numbered folders opened from above.

##### Replicates in numbered folders

Each replicate has its own folder under one shared name, but the series numbers continue across the folders instead of starting again.

```
LB/       one condition
  LB-1/   replicate 1: series001–010
  LB-2/   replicate 2: series011–020
  LB-3/   replicate 3: series021–030
```

This opens as one condition, as if all the movies sat in one folder. Check on **Confirm project** that the board's groups match your folders; they do when each folder holds one block of ten series. Longer names work too, such as `GapA-LB-rif-1`, `-2`, `-3`.

- Extra folders that only wrap the replicates are fine, for example `Glu Exhaustion/20240618/…-1`, `-2`, `-3`.
- The numbers must run 1, 2, 3, and so on. If they don't (only `LB-2` and `LB-3`, or `IPTG-10` and `IPTG-100`), the folders aren't read as replicates. Opened from the folder above, `LB` is then one condition, with a replicate per imaging day. Opened on its own, each folder becomes a condition of its own.

#### Which arrangement the analysis uses

The folder names and series numbers decide how the folders are read:

| Series numbers | Arrangement | Example |
| --- | --- | --- |
| Everything in one folder | [one replicate per imaging day](#one-replicate-per-imaging-day-the-default) | `LB/series001` to `series030` |
| Start again at `series001` in every folder | [replicates as sub-folders](#replicates-as-sub-folders) | `Glc/Glc1`, `Glc/Glc2`, `Glc/Glc3` |
| Continue across the folders | [replicates in numbered folders](#replicates-in-numbered-folders) | `LB/LB-1` (001–010), `LB-2` (011–020) |

If your replicate folders fit none of these, for example dates written as `20240101`, open the folder above your conditions, describe your folders on **What do the folders mean?**, and mark that level as the replicate. See [Opening a project](#opening-a-project).

> **Note.** Replicates are never matched across conditions by their number: replicate 1 of glucose has nothing to do with replicate 1 of malate. If your replicates of different conditions really belong together (for example one culture split into two media), select **Pair replicates** when you open the project. See [Opening a project](#opening-a-project).

### Tracks from other tools

*Analyze tracks from another tracker: put their tables in the condition folder and, when their units aren't stated, describe them in a small file.*

You don't have to track in SMTracker to use its analysis. Besides tracks from u-track and TrackMate, the analysis reads *track tables*: one file per movie, listing each localization with its track, frame and position. Use this page if you've already tracked your movies with another program.

#### Trackers it reads

The analysis recognizes the tables written by:

- trackpy
- MOSAIC ParticleTracker
- TrackIt
- Spot-On
- evalSPT / SLIMfast
- TrackMate's CSV export

The format is recognized from the column names (evalSPT tables have none, and are recognized by their four tab-separated columns), so you don't need to rename or convert anything. The analysis also reads any other table, as long as you describe its columns and units in a small file: see [Describing a table of your own](#describing-a-table-of-your-own).

#### Where to put the tables

Put the tables in a `track_tables/` folder inside the condition's folder, one file per movie, named after the movie. Tables can be `.csv`, `.tsv` or `.txt`.

```
PdhA_LB/
  series001.nd2     the movie (optional)
  track_tables/
    series001.csv
    series002.csv
    format.json     only when the units aren't stated (below)
```

*One table per movie, named after the movie, in the condition's track_tables/ folder.*

Use one tracker per condition. If a condition holds track tables and u-track or TrackMate output, the analysis asks which to use.

#### When the units aren't stated

SMTracker has to know the units of the positions, and how long a frame is, because every D scales with both. Most formats say so; some don't. Add a `format.json` to the `track_tables/` folder when:

- the format doesn't state its units (evalSPT, or TrackMate's CSV without its units row),
- the table is one of your own, or
- SMTracker hasn't saved a calibration for this folder, because you never saved it on [Calibration & save](#calibration-save), for example when the folder holds no movies.

`track_tables/format.json`:

```json
{"units": "um", "pixel_size_nm": 106, "frame_interval_ms": 20}
```

| Key | What it means |
| --- | --- |
| `units` | The units of the positions in the table: exactly `px` (pixels), `nm` or `um` (µm) |
| `pixel_size_nm` | The size of one camera pixel in the sample, in nm. Needed when SMTracker hasn't saved a calibration for this folder, whatever the table's units |
| `frame_interval_ms` | The time between two frames, in ms: the camera's frame interval, not the exposure. Needed in the same case |

A calibration saved on Calibration & save wins: a number it has replaces the one in `format.json`. If the two differ, SMTracker names both in its log (**Report a problem…** sends it) and, when you start it with `smtracker-console.exe`, in the console window, because every D depends on which is right. Correct the one that's wrong. For why the frame interval and exposure differ, see [Frame interval and pixel size](#frame-interval-and-pixel-size).

##### Describing a table of your own

For a format SMTracker doesn't know, also tell it which column holds what:

`track_tables/format.json`:

```json
{"columns": {"track": "TrackID", "frame": "Frame", "x": "X", "y": "Y"}, "units": "nm", "frame_origin": 1}
```

`columns` maps the track number, frame, x and y to your column names. `frame_origin` is the number of your first frame, 0 or 1. It defaults to 1, so set it to 0 if your frames start at 0: a wrong origin shifts every frame by one.

> **Refused.** A table whose units are unknown is refused, not guessed, because a guess would silently shift every D. Add `track_tables/format.json` with its units, and open the folder again. One table that can't be read stops the whole folder from opening, so fix it or move it aside.

#### Open them in the analysis

Open the folder above your conditions in the analysis, as in [Your first project](#your-first-project). The analysis keeps only tracks inside a cell outline, so [outline the cells](#3-cell-outlines) of each movie first: a table whose movie has no outlines is left out, and a folder where no movie has them doesn't open.

### Frame interval and pixel size

*Every D scales with the frame interval and the pixel size. See where SMTracker reads them for tracks made elsewhere, and how to fix a movie that lacks one.*

Make sure SMTracker has the right frame interval and pixel size for tracks made outside it, mostly with u-track. For movies you prepare in SMTracker, you set both on [Calibration & save](#calibration-save), and for TrackMate tracks and track tables you can skip this page.

Two numbers set the scale of every diffusion coefficient:

- the *frame interval*, the time between two frames. Every D is proportional to 1 / frame interval.
- the *pixel size*, the size of one camera pixel in the sample. Every D is proportional to its square.

A 10 % error in the pixel size moves every D by about 21 %. SMTracker never guesses either number.

A wrong pixel size leaves the fractions exactly as they are with the standard settings, because every squared jump is off by the same factor. With **Localisation error** set to *Per spot*, the fractions shift a little too, because the calibration of each spot's error is in nm. A wrong frame interval moves only the D values, with any setting.

#### The frame interval comes from the camera

The frame interval is the time from one frame to the next, which is not the exposure time. A camera exposing for 20 ms may take a frame every 20.47 or 25 ms, because reading out the chip takes time too. u-track is often given the exposure (20 ms), and that would move every D.

*A diagram here is on the website: https://longestgiraffe.github.io/smtracker-site/docs/timing-and-pixel-size*

So for u-track movies, SMTracker looks for the movie's *camera file*, the `.nd2` or `.stk` it was recorded as. When the camera file sits in the condition folder, in `raw/` or loose, and records its timing, SMTracker takes the frame interval from it and ignores the number u-track was given. The camera file must have the same name as the movie's folder in `tif/`, such as `raw/series002.nd2` for `tif/series002/`.

This applies to u-track movies tracked in SMTracker too. TrackMate tracks and track tables take the frame interval from what you saved on Calibration & save, or from `format.json`.

#### A movie with no pixel size or frame interval

A u-track movie tracked without a pixel size, or without any frame interval, can't be analyzed. When you open the project in the analysis, SMTracker lists such movies. Click **Leave out and open the rest** to go on without them.

A TrackMate or track-table condition without a pixel size or frame interval doesn't open at all: save them on Calibration & save, or add them to `format.json` ([Tracks from other tools](#tracks-from-other-tools)).

To fix a u-track movie:

| Missing | Fix |
| --- | --- |
| Pixel size | Set it in u-track and track again, or add an `acquisition.json`, as described below |
| Frame interval | Set it in u-track and track again, or put the movie's camera file in the condition's `raw/` folder |

#### Keeping a movie: `acquisition.json`

If you know a u-track movie's pixel size, you can keep the movie without tracking it again. Put an `acquisition.json` in the folder that holds `tif/`, and list the movies it applies to:

```
PdhA_LB/
  acquisition.json
  tif/
    series002/       u-track's record is empty
```

`acquisition.json`:

```json
{"pixel_size_nm": 106.0, "movies": ["series002"], "stated_by": "your name", "why": "u-track's record is empty"}
```

| Key | What it means |
| --- | --- |
| `pixel_size_nm` | The pixel size in nm. Required |
| `movies` | The movies it applies to, by their folder names in `tif/`. Required |
| `stated_by`, `stated_on`, `why` | Optional: who stated the value, when and why. SMTracker doesn't read them, but they stay in the file, so someone checking the analysis later knows where the number came from |

Two limits keep this file from hiding mistakes:

- It's used only where u-track's record is empty, never in place of a value u-track recorded.
- It can state a pixel size only. Any other key, such as a frame interval, makes SMTracker refuse the whole file. For the frame interval, see [A movie with no pixel size or frame interval](#a-movie-with-no-pixel-size-or-frame-interval).

### What SMTracker writes

*See what SMTracker adds to each condition folder and which step adds it, so you know what you can copy, move or delete.*

SMTracker keeps its results next to your data. That way a project stays complete when you copy or move its folder, and you can always see what was done to it. It only adds new files and folders beside your own. The preparation window never moves, renames or changes your movies.

The one exception is the older classic tracking window, opened with **Classic window** on Detect & link. When you save ranges there, it files loose movies into `raw/` and snapshots into `phase/`, without keeping copies.

```
PdhA_LB/
  series001.nd2      your movie
  series001.tif      your phase snapshot
  bleaching/         crops.json, curves, .png (written by SMTracker)
  sum/               series001.tif (written by SMTracker)
  trackmate/         series001.xml, run.json (written by SMTracker)
  tif/               series001/TrackingPackage/… (u-track) (written by SMTracker)
  cell_meshes/       series001.mat (written by SMTracker)
  day_map.json       (written by SMTracker)
  .smtracker_warm/   cache (written by SMTracker)
```

*Your own files are plain; what SMTracker writes is highlighted.*

#### What each step writes

Each row is one file or folder in the condition folder, the step that writes it, and what's inside.

| Written | By | What it is |
| --- | --- | --- |
| `bleaching/crops.json`, with the curves and a `.png` per movie | **Save all** (or a row's **Save**) on Calibration & save | Each movie's range and the calibration. This is the record of what was tracked |
| `sum/<movie>.tif` | The same save, unless **Save writes** is *No projection* | A projection of the movie, for the cell outlines and for display. One that another program wrote is kept |
| `tif/<movie>.tif` | **Export stacks** on Calibration & save | The movie cut to its range, for running u-track outside SMTracker |
| `trackmate_radius_check/` | **Check a wider radius (about 2 min)** on Detect & link | Test tracks, kept apart from the real ones |
| `trackmate/<movie>.xml`, `run.json` | **Track** with TrackMate | The tracks, and the settings they were tracked with |
| `trackmate/<movie>.spotfit.csv` | The same run, with **Fit each spot's precision** on | Each spot's measured precision |
| `tif/<movie>/`, `tif/utrack_run.json` | **Track** with u-track | The movie's frames and tracks in u-track's own layout (`TrackingPackage/`), and the settings |
| `cell_meshes/<movie>.mat` | **Cell outlines** | One outline per cell, in Oufti's format, so Oufti's own files work too |
| `cell_masks/<movie>.tif`, `cell_meshes_auto/` | Saving in **Edit outlines** | The outlines as a label image, and the automatic outlines kept before your corrections |
| `day_map.json` | Opening the workspace in the analysis | Which movie belongs to which [replicate](#how-replicates-are-counted), and any held-out movies or paired replicates |
| `kinetics_additions.json` | A **Kinetics** run in the analysis | When each addition happened |
| `.smtracker_warm/` | The analysis | A cache, so the project opens much faster the next time |

A [Batch](#batch) run also writes its report, `smtracker_batch_<date>-<time>.csv`, into the first parent folder. In the folder you open in the analysis, SMTracker writes the main `.smtracker_warm/` cache, and `smtracker_design.json` if you save a folder design.

You can delete `.smtracker_warm/` at any time: SMTracker rebuilds it the next time you open the project, which takes longer. Don't delete the other files unless you want to redo that step.

#### The outlines folder

SMTracker also finds outlines in a folder named `cell meshes`, `cellmeshes` or `cell-meshes`, for example from an older project or from Oufti. New outlines are written into whichever of these folders is already there.

#### A movie without outlines

The analysis keeps only tracks that lie inside a cell. When you open the project in the analysis, SMTracker leaves out every tracked movie that has no outline file, names them in a window, and opens the rest. If no movie has outlines, the project doesn't open.

To include such a movie, draw its outlines in [Cell outlines](#3-cell-outlines) and open the folder again.

#### Check that tracks and outlines line up

Trackers and segmentation tools don't always count pixels from the same origin: TrackMate and Python tools count from 0, and MATLAB tools such as Oufti and u-track from 1. Outline files that record their origin are shifted onto the tracker's automatically. If an outline file doesn't record it and the origins differ, every track sits a pixel off its cell, and the cell filter cuts tracks along one side of every cell.

QC checks this for you. Its **Is the comparison fair?** table shows, per condition, how far the tracks sit from their outlines (**Outlines**, x and y, in pixels):

- About 0 is right.
- Half a pixel or more, in both x and y in the same direction, means the tracker and the outlines count pixels from different origins.

See [QC](#qc).

If QC shows half a pixel or more, the outline files were made for the other kind of tracker and don't record their origin: typically outlines from Oufti or another MATLAB tool with TrackMate tracks. Outline the cells again in [Cell outlines](#3-cell-outlines), whose files record their origin, or track the movies with the tracker the outlines were made for. Editing the old files in SMTracker's editor doesn't help, because the editor keeps the origin it read.

## 2. Image preparation and tracking

**Image preparation + Tracking** turns a folder of raw movies (`.nd2`, `.stk` or TIFF stacks) into tracks. A TIFF stack records no timing, so you type its frame interval on Calibration & save. You choose the frames of each movie to keep, enter how it was imaged, set how molecules are detected and linked, then run the tracker. It's the first of SMTracker's three stages. To try it on movies with a known answer, the **Simulated data** card in the first window can build test movies to track here. Skip it if your tracks come from another program, because the analysis reads them directly: see [Tracks from other tools](#tracks-from-other-tools).

#### Open it

In SMTracker's first window, click the **Image preparation + Tracking** card. Click the card itself: pressing Return opens the analysis instead. In the window that opens, click **Open folder…** and choose a condition folder, or a folder that holds several conditions. For the layouts SMTracker recognizes, see [Organizing your folders](#organizing-your-folders).

SMTracker starts reading the movies right away and opens on [Movies & crop](#movies-crop).

#### The window

The sidebar on the left lists the pages under **Preparation**, top to bottom in the order you work through them. Below them, **Conditions** lists your conditions. Each has a progress bar and a line that says how many movies are read, how many have ranges, and whether they're saved. Click a condition to work on it.

| Page | What you do there |
| --- | --- |
| [Movies & crop](#movies-crop) | Check every movie and choose the frames it keeps for tracking |
| [Calibration & save](#calibration-save) | Enter the pixel size, frame interval and optics, then save |
| [Detect & link](#detect-link) | Choose the tracker and set how it detects and links molecules |
| [Track](#track) | Run the tracker on one condition or all of them |
| [Batch](#batch) | Track many folders in one run, without opening them |

The first three pages end with a button naming the next page, such as **Next: Calibration & save**. It lights up when the page's work is done, and the note beside it says what's still missing. Track ends with **Done, back to the menu** instead, and Batch has neither: it works without an open folder. These buttons don't lock anything. Once a folder is open, you can click any page in the sidebar at any time.

On the right of the header are four buttons:

- **Open folder…** opens another folder.
- **Help** opens SMTracker's built-in manual in your browser, at this stage's chapter.
- **Options** opens the values every new condition starts from. See [Preparation & tracking options](#preparation-tracking-options).
- **Back to menu** closes this window and returns to the first window, where you open cell outlines or the analysis.

While a job runs (tracking, a batch, a check of a wider radius, or exporting stacks), you can't open another folder or go back to the menu. Stop the job first, on the page that started it. Closing the window quits SMTracker. A job in progress finishes the movie it's on, then stops.

#### What you need

You don't need anything outside SMTracker until you track: reading movies, choosing ranges, calibrating and measuring thresholds all happen inside it. To track, you need one of:

- **Fiji with TrackMate**, for TrackMate.
- **MATLAB with u-track**, plus MATLAB's Statistics, Optimization and Image Processing toolboxes, for u-track.

SMTracker doesn't include either. It looks for them in the usual install folders on its own, and the Track page says what it found. For where to get them, see [Install and open SMTracker](#install-and-open-smtracker).

#### What it writes

SMTracker never moves, renames or changes your movies. Everything it creates goes into each condition's folder:

| Page | Writes |
| --- | --- |
| Movies & crop | Nothing |
| Calibration & save | `bleaching/crops.json` (the ranges and calibration), the curves in `bleaching/`, and a projection of each movie in `sum/`. **Export stacks** also writes `tif/<movie>.tif`. |
| Detect & link | Nothing, except **Check a wider radius (about 2 min)**, which writes `trackmate_radius_check/` |
| Track | TrackMate: `trackmate/<movie>.xml`, `trackmate/run.json`, and a `.spotfit.csv` beside each track file. u-track: `tif/<movie>/` and `tif/utrack_run.json`. |
| Batch | As Track, plus a report, `smtracker_batch_<date>-<time>.csv`, in the first parent folder you added |

For what each file holds, see [What SMTracker writes](#what-smtracker-writes).

### Movies & crop

*Choose which frames of each movie to track: let SMTracker propose a range, then check and adjust it.*

Choose which frames of each movie go into tracking. A movie usually starts too dense to track, while the field bleaches down to single molecules, so its first frames are left out. SMTracker reads every movie and proposes where its usable part begins, and you check each proposal on a chart. You need this page for every new folder.

A movie's *range* is the frames it keeps for tracking, from a start frame to an end frame. Nothing on this page is written to disk: the ranges are saved on the next page, [Calibration & save](#calibration-save).

To get going quickly, skip to [Propose a range for every movie](#propose-a-range-for-every-movie).

#### Reading the movies

**Movies & crop** is where you check every movie and decide which frames it keeps. It's the first page you see after opening a folder, and SMTracker starts reading the movies in the background. You can look around while it works.

SMTracker measures the mean of every frame and counts the spots in every 25th frame. Each movie takes a few seconds. A card titled **Reading movies** shows the progress, and each movie in the list shows *reading…* until it's done.

A folder you've saved before opens with its saved ranges, and its movies' curves load from the `bleaching/` folder without being read again. In a new folder, no movie has a range until you propose one, cut, or set it by hand.

#### What the page shows

The page has a list of movies on the left and a chart of the selected movie on the right. Together they show where each movie's range sits and whether its kept frames are flat.

The list has one row per movie of the condition on screen. Under each name you see its range, or *no range*, and its number of frames. The dot is green once the movie is read, gray while it's reading, and red if it couldn't be read. *· unsaved* marks a range you haven't saved yet.

Click a movie to show its chart. The chart has three panels:

| Panel | What it shows |
| --- | --- |
| **frame mean** | The mean of every frame, with the kept range drawn over the whole curve in orange. |
| **spots / frame** | The number of spots in every 25th frame, from a quick count made while reading (not from TrackMate). The kept range is in orange here too. |
| **kept range** | The frame mean of the kept range alone, on its own scale, so you can see whether it's flat. A line above it says how much the mean drops across the range. |

With no range set, or a range that covers the whole movie, the bottom panel is labeled **preview** instead. It previews the range from the floor to the last frame.

On the top two panels, the solid orange lines are the range's edges. The dashed red line is the *floor*: the earliest frame the *strictest* rule would accept. It stays where it is when you choose another rule. The dotted green lines show the last proposal, until you edit the movie by hand.

Under the chart, a line lists the floor each rule finds, then the floor SMTracker settles on. It reads `floors: density … · amplitude … · field mean … · flat … → no earlier than …`.

#### Propose a range for every movie

Proposing is the quickest way through the page. Start with the default rule, then check a few movies by eye.

##### Check the start rule

Leave **Start rule** at *strictest* for your first run. The other rules are explained under [Start rule](#start-rule).

##### Propose

Click **Propose for this condition**, or **Propose for all conditions** to do every condition at once. Every movie that's been read gets a range. A note under the chart says how many were proposed, and how many couldn't be, naming the first two with the reason.

Movies still being read are proposed as they arrive, with the rule selected at that moment.

##### Check a few movies

Click through some movies and look at the bottom panel. The kept range should start where the frame mean has flattened out and stay flat to the end. To fix one that doesn't, see [Adjust one movie](#adjust-one-movie).

> **Check this.** Proposing replaces every range of those movies, including the ones you set by hand. Adjust movies after you propose, not before.

Changing the **Start rule** doesn't re-propose anything. Click a Propose button again to apply it.

#### Adjust one movie

When a proposed start lands too early or too late, fix that movie by hand. Drag an edge of the range on the top two panels, or type the frames in **This movie keeps** … **to** … under the chart. Both change only the movie on screen. To propose this movie again with the current rule, click **Propose** beside those fields.

A range must start at frame 1 or later and end after it starts. It can't end past the movie's last frame. SMTracker refuses anything else, keeps the old range and says why in the note. A range under 100 frames is kept, and the note names it, because it gives few tracks.

#### One start or one length for every movie

When every movie was recorded the same way, you can skip proposing and give them all the same start, end or length. Each cut applies to every movie of every open condition, and to movies read later in the session.

| Control | What it does | Default |
| --- | --- | --- |
| **Every movie starts at** + **Cut all starts** | Every movie starts at this frame and keeps its end. A movie that ends at or before it is left alone, and the note names it. | 500 |
| **ends at** + **Cut all tails** | Every movie ends at this frame and keeps its start. A shorter movie ends at its own last frame. A movie that starts after it is left alone. The note names both. | the longest movie of the condition on screen (4000 until one is read) |
| **or every movie tracks** … **frames from its own start** + **Track N frames** | Every movie keeps its start and keeps exactly this many frames: a movie starting at 400 with 1000 frames ends at 1399. A shorter movie ends at its last frame, and the note names it. | 1000 |

If a movie has no range yet, SMTracker first proposes one with the **Start rule**, then applies the cut. If the rule finds nothing in it, the movie is cut from frame 1, and the note says so.

**ends at** follows the longest movie of the condition on screen as movies are read. Showing another condition resets it to that condition's longest movie, even after you've typed a value, so check it before you click **Cut all tails**. **Cut all tails** and **Track N frames** replace each other: the one you click last decides where movies end.

The last thing you do wins. A cut replaces the ranges you adjusted by hand, and a hand edit after a cut replaces the cut for that movie.

Cuts and hand edits don't check for frames where the laser was off. Only proposals leave those out, so look at the chart if your movies have dark stretches.

#### Start rule

**Start rule** decides where a proposed range begins. All four rules work from the frame mean and spot count that SMTracker measured, and skip frames where the laser was off.

| Rule | Where the range starts |
| --- | --- |
| *strictest* (default) | At the start of the longest flat stretch of the frame mean: at least 200 frames over which the mean stays within **Flat within**. If no stretch is flat, at the latest of the other three floors. |
| *density* | Where the spot count, after its peak, falls to 10 spots per frame or fewer. |
| *amplitude* | Where the median spot brightness, after the spot count's peak, comes within 1.3 times its level at the end of the movie. |
| *field mean* | Where the frame mean, after its brightest frame, falls to the level of the movie's last frames. |

Use *strictest* unless the chart tells you otherwise. If the frame mean never quite flattens but the spots clearly settle, try *density* or *amplitude*. If a rule finds no floor in a movie, SMTracker proposes nothing for it and says so in the note.

The range ends at the movie's last frame. If the laser was off at the end, it ends at the last lit frame instead.

One rule applies to new proposals in the whole session. Its default comes from **Options**.

#### Flat within

**Flat within** sets how much the frame mean may vary inside a stretch that still counts as flat, as a percentage of its level. It defaults to 2 %, but SMTracker chooses a value for each condition once its movies are read, so you rarely type one. It only affects the *strictest* rule.

##### How SMTracker chooses it

The *flat band* is the spread of counts that **Flat within** allows around the frame mean. SMTracker tries 2 %, 1 %, 0.75 %, 0.5 % and 0.25 % in turn. It takes the first value whose flat band is at most 15 % of the signal above the camera's dark level, using the median over the condition's movies, and marks it *automatic*. If even 0.25 % gives a wider band, it uses 0.25 % and says so.

To measure the dark level, SMTracker needs a few dark frames before the laser came on. Without them, or if no movie has a flat stretch at any of these values, it keeps the value already in **Flat within**. The note beside the field says which case applies.

##### When to change it

The percentage is of the raw counts, dark level included. With a high dark level and a weak signal, 2 % can be a large part of the signal, and the floor lands too early. The line under the chart warns you when the flat band is more than a quarter of the signal. When you see that warning, lower **Flat within** to 0.25 to 0.5 %, or give every movie the same start.

A value you type applies to the condition on screen and is marked *typed*. SMTracker then proposes again every movie whose range is still its proposal, with the rule that made it, and keeps the ones you've changed.

**Flat within** isn't saved with the ranges. When you open the folder again, SMTracker chooses it again, so type your value again if you had changed it. A new default in **Options** only reaches conditions that don't have their own value yet.

#### When a movie can't be proposed or read

Most problems show up in the note under the chart, with the movie's name and the reason:

- *No floor.* The rule found nowhere the movie settles. Try another rule, or set the range by hand.
- *Laser off inside the range.* The proposed range holds frames where the laser was off. Set the start after them by hand, or give every movie one start.
- *Too short.* A proposed range of fewer than 5 frames can't hold a track, so SMTracker refuses it. Set the range by hand, or try another rule.

A movie that comes in while it's being read and can't be proposed gets no range and no note. Its row shows *no range*.

A movie that couldn't be read turns red, and the chart shows the reason. For example, the file isn't a single-channel time stack, the movie was never lit in the frames SMTracker checked, or it holds no spots at all.

#### Go on to Calibration & save

**Next: Calibration & save** lights up once every movie of every condition is read and has a range. The note beside it says what's missing. A movie that couldn't be read has no range, unless one was saved before, so it keeps the button off. To go on anyway, click **Calibration & save** in the sidebar.

#### Check where to end, from your tracks

Come back to this page once a condition is tracked and its cells are outlined. SMTracker can then measure where background starts to swamp the cells near the end of each movie.

##### Open the guide

Select **Show** on **How these settings work**, then click **Every movie ends at** in its list of settings.

##### Measure

Click **Measure from the tracks here**.

For each block of 200 frames, SMTracker compares the rate of localizations outside the cells with the rate inside, and shows how much data each block holds. It changes no ranges.

This needs TrackMate tracks in `trackmate/`, cell outlines, a brightfield snapshot (in `phase/` or beside the movie) and a saved range for each movie. Movies without them are skipped, and the result counts them by reason.

### Calibration & save

*Enter each condition's pixel size, frame interval and optics, then save the ranges and calibration that tracking will use.*

Enter how each condition was imaged: its *calibration*, the pixel size, frame interval and optics. The pixel size and frame interval turn pixels and frames into µm and seconds, so every D depends on them. Saving writes the ranges from [Movies & crop](#movies-crop) and this calibration into the condition's folder. You need this page for every new folder, and again after any change.

Tracking only ever uses what you saved here. It won't start while a condition has unsaved changes, so save again after any change.

#### Fill in the calibration

The grid has one row per condition. SMTracker fills the frame interval and NA from the movies. You always enter the pixel size yourself.

##### Enter the pixel size

Type the **Pixel size** in nm: the size of one camera pixel in the sample. SMTracker never reads it from the movie files, because the value stored in a `.nd2` can be wrong. If you don't know it, click **Pixel size from the camera…** to work it out from the camera and the magnification.

##### Check the frame interval

Check the **Frame interval**, in ms, against your acquisition settings. SMTracker reads it from the first movie of each condition, where the file records it.

##### Choose the fluorophore

Choose the **Fluorophore** from the list, or type its name. A fluorophore from SMTracker's list fills in its **Emission** peak. Then check the **NA**, which SMTracker reads from `.nd2` files.

##### Save

Click **Save all**, or **Save** on one row. Each row's **Status** then reads *saved*, unless some of its movies still have no range.

If every condition was imaged the same way, fill in the first row and click **Copy first row to all**. It copies the first row's filled fields to every other condition, replacing their values, including the frame interval and NA read from their own movies.

#### The fields

| Field | What it does | Default |
| --- | --- | --- |
| **Pixel size** | The size of one camera pixel in the sample, in nm. Every D scales with its square. | empty |
| **Frame interval** | The time between two frames, in ms. Every D scales inversely with it. | read from the first movie |
| **Fluorophore** | The fluorophore's name. A name from SMTracker's list fills the emission. Any other name is kept as you typed it and fills nothing. | empty |
| **Emission** | The emission peak, in nm. With the NA and pixel size, it sets the expected spot size, which Detect & link uses as its spot radius. | from the fluorophore |
| **NA** | The objective's numerical aperture, also for the spot size. | read from a `.nd2` |

**Save** stays off until a condition has a pixel size and a frame interval. Both must be more than 0. When you type a value outside 30 to 500 nm, or 1 to 500 ms, it's kept but flagged in the note under the grid, so you can check it.

The emission and NA are optional for TrackMate, but without them SMTracker can't work out a spot radius, and you type one on [Detect & link](#detect-link) instead. u-track needs them too, unless you type a **PSF sigma** on its card.

A wrong pixel size or frame interval moves every D, but not the fractions. The one exception is the pixel size with **Localisation error** set to *Per spot*: the fractions then shift a little too, because each spot's error is calibrated in nm.

When you open a folder you've saved before, its rows start from the saved values. A field the save left empty, such as the NA, is then filled from the first movie, so the row may read *unsaved changes* straight away: save again. SMTracker also checks the saved frame interval against the first movie, where the movie records its timing. If they differ by more than 1 %, the note says by how much every D would be off.

##### Fluorophores SMTracker knows

| Fluorophore | Emission |
| --- | --- |
| mNeonGreen | 517 nm |
| GFP / mGFP | 509 nm |
| mCherry | 610 nm |
| HaloTag JF549 | 549 nm |
| HaloTag JF646 | 646 nm |
| SNAP TMR | 580 nm |
| PAmCherry (PALM) | 595 nm |
| mEos3.2 (PALM) | 580 nm |

Capitals don't matter, and a few short names work too: *mng* or *mneon* for mNeonGreen, *gfp* or *mgfp* for GFP / mGFP, *jf549* for HaloTag JF549.

Choosing a fluorophore fills **Emission** only when that field is empty, or was filled by an earlier fluorophore. An emission you typed, or one loaded from an earlier save, stays. If it differs from the fluorophore's peak, the note gives the peak, in case you want to type it.

#### Pixel size from the camera

The pixel size in the sample is the camera's pixel divided by the whole magnification, tube lens included. Click **Pixel size from the camera…** to work it out:

| Field | What to enter |
| --- | --- |
| **Camera pixel** | The camera's pixel size, in µm. Filled from a `.nd2` that records it; an Andor DU-897 has 16 µm. |
| **Objective** | The objective's magnification. Filled from a `.nd2`. |
| **Tube lens** | Any extra magnification between objective and camera: 1.0 if there's none. |

The dialog shows the arithmetic as you type. In **Apply to**, choose the condition picked last in the sidebar, or *every condition*. Then click **Apply**. The pixel size is filled in, rounded to 0.1 nm, but not saved yet.

> **Check this.** A `.nd2` leaves out the tube lens. A 16 µm camera at 100x reads 160 nm, but with a 1.5x tube lens the real pixel is 106.7 nm. Leaving the **Tube lens** at 1.0 when there is one makes the pixel size too large, and every D too large with it. Type your tube lens magnification whenever the microscope has one.

#### What Save writes

Saving a condition writes three things into its folder, in this order:

1. `bleaching/crops.json`: the ranges and the calibration. This is the record tracking reads.
2. The curves read on Movies & crop, for each movie with a range, in `bleaching/`.
3. A projection of each movie's range, in `sum/<movie>.tif`, for the cell outlines later.

**Save writes**, beside **Save all**, chooses the projection. *SD projection*, the default, shows where molecules moved over the range. *Sum projection* adds up the frames of the range. *No projection* writes nothing to `sum/`. The folder is called `sum/` whichever you choose, and **Save writes** starts at *SD projection* each time you open the window.

A projection already in `sum/` that another program wrote, such as a Fiji macro, is kept as it is.

**Save** is available once a condition has a pixel size, a frame interval and at least one range. A condition where some movies have no range can be saved, but it can't be tracked until every movie has one. **Save all** saves every condition that can be saved, including ones already saved, and skips the others.

#### Status

Each row's **Status** says where the condition stands:

| Status | Meaning |
| --- | --- |
| *needs pixel size*, *needs frame interval* | Fill it in before you can save. |
| *no ranges yet* | Set ranges on Movies & crop first. |
| *n movie(s) without a range* | You can save, but tracking needs every movie to have a range. |
| *unsaved changes* | The ranges or the calibration differ from what's saved, or the condition was never saved. |
| *saved* | What's on screen is what tracking will use. |

#### Go on to Detect & link

**Next: Detect & link** lights up once every condition is saved, with nothing changed since and a range on every movie.

#### When the movies can't be read

If a movie format needs a reader that isn't installed (`nd2` for `.nd2`, `tifffile` for `.stk` and TIFF stacks), a card says so and offers **Install**. It installs the reader into SMTracker, which can take a few minutes. Then open the folder again, so the movies that failed are read. In a packaged build nothing can be installed, and the card says so: ask whoever gave you the build for one that includes the reader.

#### Cropped stacks for u-track

u-track can't read a `.nd2` or `.stk`. If you run u-track outside SMTracker, **Export stacks** cuts each movie's saved range into `tif/<movie>.tif`, with the pixel size and frame interval written into the file. It also writes the projection to `sum/` (an SD projection, even with *No projection* chosen) and the curves. SMTracker's own u-track runs and TrackMate don't need this.

Export works only from what's saved, so save first. Each condition has a line saying whether it's ready, and why not. If `tif/` already holds any stacks, the condition isn't ready until you select **Replace stacks already in tif/**. A stack another program wrote is never replaced: move it aside first.

To stop, click **Stop after the movies in hand** while the export runs. The movies being written finish, and the rest aren't written.

### Detect & link

*Set how the tracker finds molecules and joins them into tracks: the detection threshold, spot radius, linking radius and frame gap.*

Set how the tracker finds molecules in each frame and joins them into tracks. With TrackMate, that's four settings per condition, and SMTracker measures or derives most of them for you. With u-track, it's one set of settings for every condition. You need this page before [Track](#track), each time you open a folder.

To get going quickly with TrackMate, skip to [TrackMate](#trackmate).

#### Choose the tracker

Choose *TrackMate (Fiji)* or *u-track (MATLAB)* in **Tracker**, at the top of the page. SMTracker remembers the choice, and the same choice appears on Track and Batch. Changing it changes no settings of either tracker. You can't change it while a run or a radius check is running.

The rest of the page shows the chosen tracker's settings. [Install and open SMTracker](#install-and-open-smtracker) explains the difference between the two.

To use the older tracking window on this folder, click **Classic window**. It opens with the ranges you saved.

#### TrackMate

The grid has one row per condition, with its four settings and a status. Most rows only need their threshold measured.

##### Measure the thresholds

Click **Measure all thresholds**, or **Measure** on one row. SMTracker measures each movie's noise and sets its detection threshold from it. This needs a range on every movie, from [Movies & crop](#movies-crop) (saved or not), and a spot radius.

##### Check the other settings

Check the **Spot radius**, **Linking radius** and **Frame gap**. The spot radius comes from the condition's optics, and the other two start from the standards in [Options](#preparation-tracking-options). Leave them unless you have a reason to change them, explained under [Linking radius and frame gap](#linking-radius-and-frame-gap).

##### Go on

Click **Next: Track**. It lights up once every condition can be tracked: its ranges and calibration saved with nothing changed since, and all four settings set. Usually only the threshold is missing. If it stays off, the note beside it gives the first reason.

| Setting | What it does | Default |
| --- | --- | --- |
| **LoG threshold (median)** | How bright a spot must be, against its movie's own noise, to count as a molecule. Each movie gets its own threshold; the cell shows the median, rounded. | empty until measured, then 7.2 times each movie's noise |
| **Spot radius** | How big a molecule looks to the detector, in px. Worked out from the emission, NA and pixel size on [Calibration & save](#calibration-save). | from the optics (2.80 px at 517 nm, NA 1.49, 106 nm pixels) |
| **Linking radius** | How far a molecule may move between two frames and still be joined into the same track, in px. A longer jump starts a new track. | 5 px |
| **Frame gap** | How many frames a molecule may vanish for and still be joined back into its track. 0 means no gaps. | 3 frames |

A value you type replaces the measured or derived one for that condition. The **Status** column says what each row still needs, for example *needs a range on every movie first* or *set the spot radius*.

An empty cell isn't accepted, so clearing a typed value brings it back. To change a value, type the one you want. A typed value stays typed, and no longer follows its measured, derived or standard value, until you open the folder again.

##### The detection threshold

**Measure** reads up to 8 frames spread over each movie's range, filters them as TrackMate does, and measures the noise. Each movie's threshold is k times its noise, where k is the **Threshold** in Options (7.2). Each movie is then tracked at its own threshold. A movie whose noise couldn't be measured is tracked at the condition's median, and the line under the grid names it.

The line under the grid gives the noise. If the noisiest movie of a condition is more than 1.5 times as noisy as the quietest, SMTracker warns that the condition may hold more than one imaging session, and names any movie far from the median. Nothing is excluded: check those movies yourself.

Keep the measured thresholds for almost every folder. Typing a threshold sets one value for every movie of that condition, instead of one per movie, and measuring again doesn't replace it.

Changing **Threshold** in Options recomputes every measured threshold from the noise already measured, so you don't need to measure again.

> **Check this.** Measure again after you change a movie's range or the spot radius. SMTracker doesn't do it for you and doesn't flag an outdated threshold, so the old one stays in use.

##### Spot radius

SMTracker works out the spot radius as 3 PSF sigmas, from the emission, NA and pixel size on screen, saved or not. It's empty until all three are set, and it updates when they change, unless you typed one.

The spot radius isn't the linking radius. It describes how big a molecule looks, while the linking radius describes how far it moves.

##### Linking radius and frame gap

Under the grid, SMTracker says what the linking radius means for each condition: the fastest D it measures within 10 %, and its *detection ceiling*, the highest apparent D any population can show at that radius. It also shows the frame gap as a time. Both need the frame interval and pixel size.

Leave the linking radius at 5 px. In a reference analysis of six proteins, 5 px reproduced the results, while 12 px raised the slow fraction by 12.8 points. A wider radius adds wrong links, and they look like slow steps. Widen it only when the detection ceiling is below the D you expect, and check first with [Check a wider radius](#check-the-linking-radius-from-your-tracks).

Leave the gap at 3 too. With no gap closing, TrackMate's tracks break up near the threshold, and a gap of 0 shifted the slow fraction by several points. The gap counts frames, so at a slower frame rate the same gap allows a longer disappearance.

A new standard in Options reaches every condition whose value you haven't typed. **Copy linking radius and gap to all** copies the first condition's two values to every other condition, whichever condition is on screen. The copies then count as typed, so they no longer follow Options.

##### Check the linking radius from your tracks

Two buttons help you judge the linking radius. Both are in **How these settings work**, under **Linking radius**. If the guide is folded, select its **Show** checkbox first. They act on the condition whose row you last clicked into, so click a cell in that row first.

- **Measure from tracks here** reads up to 8 movies' tracks you've already made in this condition (u-track's, if it has any). It reports the step lengths and how often a second molecule lies within the radius, and suggests a radius. It needs tracks first, ideally from a pilot run at a generous radius.
- **Check a wider radius (about 2 min)** tracks the condition's first two movies at the linking radius and at 1.6 times it, everything else the same. It reports how many steps the narrower radius missed. It needs everything a real run needs, a saved frame interval and pixel size, and Fiji. The results go to `trackmate_radius_check/` in the condition's folder.

Neither button changes a setting. Treat a wider-radius result as information: the fractions were validated at 5 px, so a result here alone is no reason to widen the radius.

##### What isn't saved

The TrackMate grid lives only in this window. Typed values and measured thresholds aren't saved, so when you open the folder again, measure the thresholds again. The run itself records the values it used, in `trackmate/run.json`.

#### u-track

u-track uses one set of settings for every condition. SMTracker remembers them between sessions, and the standard settings suit most data.

The card shows nine settings in three groups, **Detection**, **Linking** and **Gaps and length**. Each has a link that explains it.

| Setting | What it does | Default |
| --- | --- | --- |
| **Spot significance (alphaLocMax)** | How unlikely a local maximum must be against its background to count as a spot. Smaller is stricter. | 0.025 (u-track's own is 0.05) |
| **PSF sigma** | The spot's width, in px. Empty: worked out from the optics. | empty |
| **Cell-free background test** | An alpha that turns on a test against the background outside cells. Needs cell outlines for every movie. | empty: off |
| **Search radius, largest** | The upper limit of u-track's adaptive search radius, in px: its counterpart of the linking radius. | 5 px |
| **Search radius, smallest** | The lower limit of the adaptive search radius, in px. | 0 px |
| **Radius multiplier** | How many of a track's own step SDs a link may reach. | 3 |
| **Frames bridged** | How many frames a molecule may vanish for and still be joined: u-track's counterpart of the frame gap. | 0 (no gap closing) |
| **Shortest segment kept** | Shorter track segments are dropped before gap closing. | 5 localizations |
| **Merges and splits** | Whether u-track may join or split tracks. *None* keeps every track separate. | *None* |

**Show all settings** adds 16 advanced settings to the same groups. **Standard settings** puts six of the settings back to your standard in Options, if you've set one, and every other setting back to SMTracker's. The line under the card names every setting that differs from SMTracker's standard, even where it matches yours.

The table under the card has one row per condition: its PSF sigma, its cell outlines and whether it's ready. u-track needs saved ranges, a saved pixel size and frame interval, and the NA and emission saved, unless you type a PSF sigma. With the background test on, every movie also needs cell outlines.

A value u-track wouldn't accept is refused, and the note says why. Number fields don't take letters or minus signs at all.

### Track

*Run TrackMate or u-track on one condition or all of them, and see what each run produced.*

Run the tracker on your prepared conditions. Each condition is tracked from the ranges and calibration you saved, and the tracks are written into its folder, where the analysis reads them. You need this page once per folder, after [Detect & link](#detect-link).

To track many folders without opening each one, use [Batch](#batch) instead.

#### Before you track

Track reads the ranges and calibration you saved on [Calibration & save](#calibration-save), never what's only on screen. If you changed a range or the calibration after saving, save again first. The Detect & link settings are used as they stand, and need no saving.

Each row of the conditions table says whether that condition is ready. A condition isn't ready when, for example:

- it was edited since the last save, or has never been saved;
- a movie has no saved range;
- with TrackMate, it has no detection threshold, or another setting on Detect & link is empty;
- with u-track, the pixel size, the frame interval, or the NA and emission (or a typed PSF sigma) weren't saved;
- with u-track and the **Cell-free background test** on, a movie has no cell outlines;
- it was tracked before, and the Replace checkbox under [Settings](#settings) is cleared.

The row gives the exact reason. Fix it on the page it names, then come back.

A ready row also shows an estimate of the time: about 30 seconds a movie with TrackMate, or about 0.13 seconds a frame plus 10 seconds to start with u-track, divided by **Movies at once**.

#### Find Fiji or MATLAB

The card at the top shows the program the chosen tracker runs in: Fiji for TrackMate, MATLAB with u-track for u-track. SMTracker looks for them on its own, in the usual install folders: for Fiji when the window opens, for MATLAB and u-track once u-track is the chosen tracker. When it finds them, the card says where, and you don't need to do anything.

If it finds nothing, or the wrong install, point it to the right one:

- TrackMate: click **Choose Fiji folder…** and choose your Fiji folder (on a Mac, `Fiji.app`).
- u-track: click **Choose matlab.exe…** and choose MATLAB in a release's `bin` folder (`matlab.exe` on Windows, `matlab` on a Mac). Then click **Choose u-track folder…** and choose the folder that holds `software/@UTrackPackage`.

SMTracker remembers a choice only when it finds the program in the folder you chose. If it doesn't, the card says so. [Install and open SMTracker](#install-and-open-smtracker) covers getting Fiji and u-track.

A row can read *ready* even when no Fiji or MATLAB was found. In that case, SMTracker refuses when you click **Track**, and the note under the table says why.

#### Track your conditions

Track one condition first to see that the settings work, or every ready condition at once.

##### Check the tracker

Check that **Tracker**, at the top, shows the tracker you set up on [Detect & link](#detect-link).

##### Start

Click **Track** on a condition's row, or **Track all ready conditions** to track every ready one. One condition runs at a time. Clicking **Track** while another condition runs puts it in the queue, and its row says *queued*.

##### Follow the run

Watch the running row: it shows how many movies are done, with a progress bar. The log at the bottom adds one line per movie as it finishes, with what it found and how long it took. A card in the window shows the progress over all queued conditions and, after the first movie, the time left.

When the last condition is done, a banner says what each one produced. Click **Done** in it to close the window and go back to the menu, where the analysis is the next step. If the run took a minute or more, a window pops up with the same summary. Change that time with **Pop-up when a job ran at least** in [Options](#preparation-tracking-options).

While a condition is tracking, you can't change the tracker or **Movies at once**, open another folder, or go back to the menu.

##### The first u-track run

The first time you click **Track** with u-track in a window, SMTracker starts MATLAB to check u-track and the toolboxes it needs (Statistics, Optimization, Image Processing). This takes about 30 seconds, and the run starts when MATLAB answers. Don't close the window meanwhile. If the check fails, or a toolbox is missing, the queue is dropped and the note says why.

SMTracker then writes every movie's frames to disk before MATLAB starts tracking, so the progress bar stays at zero for a while. The bar and the log move as MATLAB finishes each movie. MATLAB runs minimized.

#### Settings

| Control | What it does | Default |
| --- | --- | --- |
| **Replace an earlier run in trackmate/** (TrackMate) or **Replace an earlier u-track run in tif/** (u-track) | Selected: a condition tracked before is tracked again. Cleared: it isn't ready, and its row says so. | cleared |
| **Movies at once** | How many movies of a condition are tracked side by side, from 1 to 4: one Fiji each with TrackMate, or that many MATLAB sessions with u-track. The tracks don't depend on it, only the time. | half this computer's threads, less one |

The Replace checkbox starts cleared each time you open the window, because two runs in one folder can't be told apart afterward. Select it only when you mean to track a condition again.

- With TrackMate, a condition counts as tracked when `trackmate/` holds any track file. A new run writes each movie's track file over the old one of the same name, and other old files stay.
- With u-track, a condition counts as tracked when even one of its movies already has a u-track run. A new run replaces each movie's frames and tracks.

> **Check this.** Replacing deletes nothing first. If a replacing run stops or fails part way, the folder holds new results for the movies that ran and old ones for the rest. Finish the run before you analyze the condition.

**Movies at once** is the same number on Batch, and SMTracker remembers it. It's never more than the number of movies. Raise it to finish sooner, if the computer has memory to spare. Each u-track session reads its frames from disk, so on a spinning hard disk more sessions don't help. In that case, set a **u-track work folder** on an SSD in Options: u-track then works there and moves each movie's results into its condition when it's done.

Each u-track session holds its own MATLAB, which takes up to about 1.5 GB of memory.

#### Stop a run

Click **Stop after this movie**. The movies being tracked finish, so no track file is left half written, and nothing new starts. Stopping also drops every condition still in the queue.

The condition then reads *done, incomplete*, and the movies that didn't run are counted in its row. Track it again later with the Replace checkbox selected.

Closing the window during a run stops it the same way.

#### What a run writes

Each condition's tracks go into its own folder:

| Tracker | Where |
| --- | --- |
| TrackMate | `trackmate/<movie>.xml`, one per movie, and `trackmate/run.json` with the settings and outcome. With **Fit each spot's precision** on in Options (the default) and the optics saved, a `.spotfit.csv` beside each track file. |
| u-track | For each movie, `tif/<movie>/` with its frames (in `tif/<movie>/<movie>/`), `movieData.mat` and the tracks in `TrackingPackage/`. Plus `tif/utrack_run.json` with the settings and outcome, and `tif/utrack_job/` with MATLAB's working files and log. |

After each condition, its row says how many *usable* tracks it produced: tracks of at least 5 localizations, the ones the analysis keeps. The log adds a summary, and flags TrackMate movies that look unlike their neighbors (they're flagged, never dropped).

A TrackMate movie that takes more than an hour fails. The other movies go on, and the failure is named in the log.

[What SMTracker writes](#what-smtracker-writes) lists every file the app creates.

### Batch

*Track every prepared condition under several folders in one unattended run, overnight if you like, and get a report of what ran.*

Track many folders in one go, for example overnight. You add the folders, SMTracker lists every condition inside them, and it tracks each one that's ready, one after another. Use it once your conditions are prepared and saved, when tracking them one at a time on [Track](#track) would keep you at the computer.

**Batch** needs no open folder. It works from what's saved on disk, so it can track folders you haven't opened in this session.

#### Before you start

A batch never proposes ranges or guesses a calibration. A condition runs only if what it needs was saved before, as tracking always does ([Calibration & save](#calibration-save) explains why). Otherwise it's skipped, and the first thing missing is named in its row.

For every condition, you need:

- **A saved range for every movie.** Set the ranges on [Movies & crop](#movies-crop), then click **Save all** on Calibration & save.
- **The saved calibration the tracker needs:**
  - TrackMate: the emission, NA and pixel size, so SMTracker can work out the spot radius.
  - u-track: the pixel size and frame interval, plus the NA and emission (or a PSF sigma typed on the u-track card on [Detect & link](#detect-link)). With the **Cell-free background test** on, every movie also needs cell outlines.
- **No unsaved edits.** If the condition is open in the window with edits you haven't saved, the batch skips it and names it, even though an earlier version is saved on disk.

A condition that's open in the window runs with its Detect & link values as they are in the window, typed or measured, saved or not.

TrackMate also needs a detection threshold. A folder you haven't measured in this window doesn't have one, because thresholds aren't saved. The batch can measure it for you: see [Measure a missing threshold](#measure-a-missing-threshold).

#### Run a batch

##### Add the folders

Click **Add parent folder…** and choose a folder that holds condition folders, or one condition folder. Repeat for each folder.

The table lists every condition it finds, one level down: the parent folder, the condition, its *layout* (how its movies are arranged in the folder), its number of movies and whether it's ready. A folder that holds whole projects one level down isn't searched deeper. SMTracker says so: add each project as its own parent folder.

##### Check the table

Read the **Status** column. *ready* rows run, and each shows a time estimate. *skipped* rows say what's missing: fix those conditions on the earlier pages and save, or let the batch skip them. The line under the table sums up how many conditions run and roughly how long they take.

##### Run

Choose the **Tracker** and the other [settings](#settings), then click **Run batch**. SMTracker lists the folders again as it starts, so it runs what's on disk at that moment.

**Run batch** is available once at least one row is ready. It's refused while a condition is tracking on Track or a radius check is running, and when Fiji or MATLAB isn't set up. The note under the table says why.

With u-track, if MATLAB hasn't been checked yet in this window, SMTracker first starts it, which takes about 30 seconds (see [The first u-track run](#track)). The batch starts when MATLAB answers.

#### Settings

The settings apply to every condition in the batch.

| Control | What it does | Default |
| --- | --- | --- |
| **Tracker** | The same choice as on [Detect & link](#detect-link): changing it here changes it there too. u-track uses the u-track card's settings for every condition. | the tracker you used last |
| **Replace earlier runs** | Selected: conditions tracked before are tracked again. Cleared: they're skipped. | cleared |
| **Measure a missing threshold inside the saved crops (k = 7.2)** | TrackMate only. Selected: a condition with no detection threshold gets one measured just before its run. Cleared: it's skipped. The k comes from **Threshold** in [Options](#preparation-tracking-options). | selected |
| **Movies at once** | How many movies of a condition are tracked side by side, from 1 to 4. Conditions always run one after another. It's the same number as on [Track](#track). | the number you used last |

The two checkboxes aren't remembered: each time you open the window, **Replace earlier runs** starts cleared and the threshold checkbox starts selected. The list of parent folders starts empty too.

##### Replace earlier runs

With TrackMate, a condition counts as tracked before when its `trackmate/` folder holds any track file. With u-track, it counts when even one of its movies already has a u-track run in `tif/`. With the checkbox cleared, such a condition is skipped. [Track](#track) explains what a replacing run overwrites, and what it leaves.

##### Measure a missing threshold

For each movie, SMTracker takes 8 frames spread over its saved range, measures the noise there, and sets the threshold to k times that noise. Each movie is then tracked at its own threshold, as when you click **Measure** on [Detect & link](#detect-link). A movie whose noise couldn't be measured is tracked at the condition's median threshold.

The measured threshold isn't saved. Running the same folder again measures it again. A condition still needs its spot radius: without saved optics (or a radius typed on Detect & link, for an open condition), it's skipped even with this checkbox selected.

#### While it runs

Each row's **Status** shows how far that condition has come, and the log at the bottom lists every movie as it finishes. A card in the window shows the progress over all conditions.

While a batch runs, you can't open another folder, track on Track, change the tracker, or go back to the menu. On Windows, SMTracker keeps the computer from sleeping until the batch ends (the display may still turn off). On a Mac or Linux, nothing keeps the computer awake, so turn off sleep for a long batch.

To stop, click **Stop after the movie in hand**. The movies being tracked finish, so no track file is left half written, and nothing new starts. Closing the window stops the batch the same way.

If one condition fails, the batch names it in the log and the report, and goes on with the next.

#### The report

The report is a CSV file named `smtracker_batch_<date>-<time>.csv`, written into the first parent folder. It has one row per condition, skipped ones included. SMTracker writes it as the batch starts and again after every condition, so it's up to date even if the batch is stopped.

| Column | What it holds |
| --- | --- |
| `parent`, `condition`, `folder`, `layout` | Where the condition is, and how its folder is arranged |
| `tracker` | `trackmate` or `utrack` |
| `status` | What happened (below) |
| `movies`, `movies_tracked`, `movies_failed`, `movies_stopped` | Movie counts |
| `tracks`, `usable_tracks` | All tracks, and those with at least 5 localizations |
| `error` | Why a condition was skipped or failed, if it was |

The status of each condition is one of:

| Status | Meaning |
| --- | --- |
| *tracked* | Every movie was tracked. |
| *tracked, with failures* | Some movies failed. |
| *stopped part way* | You stopped the batch during this condition. |
| *failed* | No movie was tracked, or the run hit an error, such as a threshold that couldn't be measured. |
| *not run: stopped* | You stopped the batch before this condition started. |
| *skipped* | The condition wasn't ready. |

If the report can't be written, for example because it's open in a spreadsheet, the batch goes on. The log says so straight away, and the summary says so at the end. When the batch has run for at least a minute, a window pops up with the summary. You can change that time with **Pop-up when a job ran at least** in [Options](#preparation-tracking-options).

### Preparation & tracking options

*Set the standards new conditions start from: the crop rule, the TrackMate and u-track settings, and when to be told a job has finished.*

Choose the values each page starts from in a new condition, so you don't type the same settings into every folder. Each of these values is a *standard*: a value you set on a page always wins over it. Most people never change them. Change one when your microscope or your proteins consistently need a different value.

#### Open and save

Click **Options** in the window's header, with or without a folder open. The **Preparation & tracking options** screen groups the settings under **When a job finishes**, **Crop proposals**, **TrackMate standard** and **u-track standard**, each with a line explaining it.

Click **Save** to keep your changes, or **Cancel** to discard them. **Reset to defaults** sets every field back to SMTracker's defaults, but writes nothing until you save. If one value isn't accepted, nothing is saved, and a line in red says which. In practice, that's a **u-track work folder** that doesn't exist, or u-track values that don't fit together, such as a smallest search radius above the largest.

Saving applies at once, including in the window that's open. A value still at its old standard moves to the new one, and a value you changed on a page since you opened the window stays. A note in the header names the crop and u-track settings and the thresholds that moved. Untyped linking radii and frame gaps move too, without a mention.

SMTracker stores the options for you on this computer, not in your data folders.

#### When a job finishes

Choose when SMTracker tells you a job has finished, so you can work in another program during long runs.

| Option | What it does | Default |
| --- | --- | --- |
| **Pop-up when a job ran at least** | A job that runs at least this long, such as reading movies, saving or tracking, ends with a pop-up saying what it did, and the taskbar flashes. A shorter job ends quietly. Set it to 0 for a pop-up after every job. | 60 s |

This is the same setting as on the **Cell outline options** screen: changing it on either screen changes both.

#### Crop proposals

These set how [Movies & crop](#movies-crop) proposes ranges.

| Option | What it does | Default |
| --- | --- | --- |
| **Start rule** | The rule Movies & crop uses each time you open the window: *strictest*, *density*, *amplitude* or *field mean*. A rule you choose on Movies & crop lasts only until you close the window. | *strictest* |
| **Flat within** | How far the frame mean may vary inside a stretch that still counts as flat. Lower it for a camera with a high dark level and a weak signal. | 2 % |

For what each rule does, see [Start rule](#movies-crop) on Movies & crop.

**Flat within** matters less than it seems. Once a condition's movies are read, SMTracker chooses its own value for that condition, and the standard no longer applies to it. The standard is used only where SMTracker can't choose, for example when the movies have no dark frames before the laser.

#### TrackMate standard

These set what [Detect & link](#detect-link) starts from with TrackMate. The defaults suit most data.

| Option | What it does | Default |
| --- | --- | --- |
| **Linking radius** | The linking radius each new condition starts with, in px. It caps the fastest D that can be linked: at 50 ms, 5 px covers D up to about 0.34 µm²/s. | 5 px |
| **Frame gap** | How many frames a molecule may vanish for and still be linked, for each new condition. 0 means no gap closing. | 3 frames |
| **Fit each spot's precision** | After TrackMate, refits every tracked spot to measure its precision, and saves it beside the track file. The tracks don't change. The analysis needs it for **Localisation error** set to *Per spot*. | on |
| **Threshold** | The multiplier k for measured thresholds. The **Measure** buttons on Detect & link set each movie's threshold to k times its noise. It isn't a threshold itself. | 7.2 times the noise |

The linking radius and frame gap reach every condition whose value you haven't typed, including the folders a [Batch](#batch) runs. Values copied with **Copy linking radius and gap to all** count as typed, so they don't follow.

Saving a new **Threshold** updates every threshold you've measured in the open window, using the noise SMTracker already has, so nothing is measured again. Thresholds you typed stay as they are.

The default k was calibrated on one microscope. On another, compare TrackMate's spot count with a u-track run on the same movie, and adjust k until they match.

**Fit each spot's precision** is set only here: no page has a switch for it. It needs the condition's emission, NA and pixel size from Calibration & save. If one is missing, the run's log on Track says the precision wasn't fitted.

#### u-track standard

These set what the u-track card on [Detect & link](#detect-link) starts from. The defaults suit most data. Change them to match an earlier u-track analysis you want to compare against.

| Option | What it does | Default |
| --- | --- | --- |
| **u-track work folder** | Where u-track works on its frames before moving each movie's results into its condition. Empty means beside the data. Choose a folder on an SSD if your data is on a spinning hard disk: 40 movies of 2500 frames need about 3 GB. | empty |
| **Spot significance (alphaLocMax)** | How unlikely a spot must be against its local background. Smaller is stricter. | 0.025 |
| **Search radius, largest** | The upper limit of u-track's search radius, in px. u-track sets each track's radius from its own steps, within these two limits. | 5 px |
| **Search radius, smallest** | The lower limit of that search radius, in px. | 0 px |
| **Radius multiplier** | How many of a track's own step SDs a link may reach. | 3 |
| **Frames bridged** | How many frames a molecule may vanish for and still be joined. 0 means no gap closing. | 0 |
| **Shortest segment kept** | A *segment* is a piece of track before gaps are closed. Shorter segments are dropped. 5 is the analysis's own minimum. | 5 localizations |

The work folder applies from the next u-track run. Choose a folder that exists: SMTracker refuses one that doesn't, and the red line says so.

> **In the app.** Once you've edited the u-track card, it remembers its settings between sessions, so a new u-track standard doesn't always reach it. It does when the card still holds the old standard as you save, or when you click **Standard settings** on the card.

## 3. Cell outlines

**Cell outlines** is where you outline every cell, automatically, and correct the outlines by hand. The analysis keeps only tracks that lie inside a cell outline, so every tracked movie needs its outlines. Outlines drawn too tight drop the free molecules near the membrane and raise the slow fraction, so check them in [Edit outlines](#edit-outlines) before you analyze.

It's the second of SMTracker's three stages, but you don't have to track first. The window works *field* by field: a field is the image of one imaged position, either its phase snapshot or the movie's projection in `sum/`.

#### Open it

In SMTracker's first window, click the **Cell outlines: Segmentation + Mesh editor** card. In the window that opens, click **Open folder…** and choose a condition, or a folder of conditions.

In Cell outlines, a *condition* is still one folder, and it needs images to outline:

- phase snapshots, in a folder named `phase` (or `brightfield`, `bf`, `trans`, `dic`), or lying beside the movies; or
- a `sum/` folder, written when you save on [Calibration & save](#calibration-save). Despite its name, it holds an SD projection of each movie by default.

Every condition starts on the phase model. A condition with only `sum/` needs a model that reads it: see [Segment](#segment).

You can also open a folder one level higher, whose subfolders each hold conditions, for example one folder per protein. Each subfolder opens as a group. If two conditions in different groups share a name, the window can't tell them apart and opens neither group, without a message. Open each group on its own instead ([Organizing your folders](#organizing-your-folders)).

#### The window

The sidebar lists the pages under **Cell outlines**, and below them, your conditions. Each condition's line says how many outline files it has, out of its fields, and how many fields you corrected by hand.

| Page | What you do there |
| --- | --- |
| [Segment](#segment) | Outline every field automatically, condition by condition |
| [Classical settings](#classical-settings) | Tune the classical segmenter on one field, with a live preview |
| [Edit outlines](#edit-outlines) | Correct the outlines by hand, field by field |
| [Train a model](#train-a-model) | Fine-tune a model on your corrected outlines |
| [Batch and Options](#batch-and-options) | Segment many folders in one run, and set the standards a new session starts from |

Segment ends with **Next: Edit outlines**, and Edit outlines with **Done, back to the menu**, once a folder is open. The other pages have neither. These buttons don't lock anything: you can click any page in the sidebar.

On the right of the header are four buttons:

- **Open folder…** opens another folder.
- **Help** opens SMTracker's built-in manual in your browser, at this chapter.
- **Options** opens the standards a new session starts from. See [Batch and Options](#batch-and-options).
- **Back to menu** returns to the first window.

#### Phase or fluorescence

Use phase contrast unless you have a reason not to: it works for any protein, and the default model, *bact_phase_omni*, reads it. Fluorescence works only for a cytosolic label: a membrane- or DNA-bound label gives outlines about half the true cell area. For every model, see [Segment](#segment).

#### What you need

SMTracker segments with either the *Omnipose models* (*bact_phase_omni*, *bact_fluor_omni* and the pipelines built on them) or the *classical segmenter*, a fast threshold-and-shape method. The classical segmenter and Edit outlines work everywhere. The Omnipose models need libraries that the packaged app leaves out; to use them, run SMTracker from source ([Install and open SMTracker](#install-and-open-smtracker)).

> **Check this.** The classical segmenter reads only the `sum/` projection. So in the packaged app, outlines can only come automatically from fluorescence: a membrane- or DNA-bound label, or a condition without `sum/`, has to be outlined by hand in Edit outlines, or segmented from source.

A graphics card speeds the Omnipose models up. SMTracker checks for an NVIDIA GPU each time a folder opens with at least one condition.

#### What it writes

Everything goes into each condition's folder, beside your data:

| Written | By | What it is |
| --- | --- | --- |
| `cell_meshes/<field>.mat` | Segment, Batch, and saving in Edit outlines | One outline per cell, in Oufti's format. An existing folder spelled differently, such as `cell meshes` or `Cell Meshes`, is used instead. |
| `cell_masks/<field>.tif` | Saving in Edit outlines | The outlines as a label image |
| `cell_meshes_auto/<field>.mat` | The first save in Edit outlines | The automatic outlines, kept once before your corrections |
| `cell_meshes/<field>.mat.curated.json` | Saving in Edit outlines | A marker that you corrected this field by hand |

Trained models go to SMTracker's model store, not into your data: `%APPDATA%\SMTracker\models` on Windows, or `~/.config/smtracker/models` on macOS and Linux.

#### Next

To start, outline your first condition on [Segment](#segment).

### Segment

*Outline the cells of every field automatically, with an Omnipose model or the classical segmenter.*

Draw the cell outlines of every field automatically, condition by condition. You'll need it for every new folder. Then check and correct what the model got wrong in [Edit outlines](#edit-outlines).

#### Open the folder

Open the folder in the **Cell outlines** window, as described in the [Overview](#3-cell-outlines). The Segment page lists one row per condition.

#### Segment the conditions

##### Check each row's model

Each row has a **Model** and the folder it **Reads**. The default, *bact_phase_omni (reads phase)*, suits most data. Change it on a row if you need to, or set the first row and click **Copy the first row to all**.

##### Segment

Click **Segment all ready** to segment every ready condition, one after another, or **Segment** on one row. The **Status** column says why a row isn't ready, for example *set the folder it reads (phase)*, or *all 12 field(s) outlined; tick Replace to segment again*.

##### Follow the run

The **Run** card shows the progress, the time left and a log line per field, with its number of cells. The first field also loads the model, so it's the slowest. When the run ends, click the button that names the condition, such as **Open ProteinA in the editor**, to check its outlines.

To stop, click **Stop after this field**. The field in hand finishes, so no outline file is left half written, and the conditions still waiting don't run.

#### Which model

| Model | What it reads, and when to use it |
| --- | --- |
| *bact_phase_omni (reads phase)* | The phase-contrast snapshots. Works for any protein, and the right choice for most data. |
| *bact_fluor_omni (reads sum)* | The projection in `sum/`, written by Calibration & save. Only for a cytosolic label. |
| *Phase pipeline: bact_phase_omni, chains split (reads phase)* | The phase model with extra steps that split constrictions and chains of cells. |
| *Sum-image pipeline: bact_fluor_omni, chains split (reads sum)* | The same for the projection. |
| *Cellpose-SAM (generalist; downloads 1.2 GB at first use)* | The phase-contrast snapshots. A general model, not trained on bacteria in particular. Needs the [segmentation pack](#install-and-open-smtracker). |
| *Classical, sum image (no GPU, no training)* | A fast threshold-and-shape segmenter on the projection, weaker on some data. Tune it on [Classical settings](#classical-settings). |
| *your model (trained, reads …)* | A model you made on [Train a model](#train-a-model). It reads the images it was trained on. |

Segmenting the fluorescence of a membrane- or DNA-bound label gives outlines about half the true cell area, so use a phase model for those. After a run, the log warns you if the signal sits at the edge or in the center of the outlines, the sign of such a label.

**Model for a new condition** in [Options](#batch-and-options) sets the model each new condition starts with.

> **Refused.** Cellpose-SAM needs the segmentation pack, PyTorch and Cellpose, which SMTracker downloads on request ([Install and open SMTracker](#install-and-open-smtracker)). Until then the page offers **Get segmentation models…**. The Omnipose models and the two pipelines run only from source, with Omnipose installed. The classical segmenter and Edit outlines work either way.

#### The row's folder

**Reads** shows the folder the model reads, and it follows the model. Phase models read a folder named `phase` (or `brightfield`, `bf`, `trans`, `dic`), and otherwise the snapshots beside the movies. Sum models read a folder named `sum` (or `fluorescence`, `fluo`, `epi`, `signal`, `proj`).

If two folders match and SMTracker can't tell which holds this condition's images, or none matches, the row reads *(set the folder)*. Choose one from the list, or choose *choose another folder…* to pick any folder.

A folder you choose by hand stays until you change the row's model, click **Copy the first row to all**, or save a new default model in Options while the row is on the old one. Each of these sets the folder back to follow the model.

**Copy the first row to all** gives every condition the first row's model, ignored ones included. If you chose the first row's folder by hand, conditions with a folder of that name get it too. Otherwise every other row's hand-chosen folder is cleared.

#### Settings

These apply to the runs you start on this page.

| Control | What it does | Default |
| --- | --- | --- |
| **Replace existing outlines** | Selected: every field is segmented again, even fields you corrected by hand. Cleared: only fields without an outline file are segmented. | cleared |
| **Use GPU** | Runs the Omnipose models on an NVIDIA graphics card. Available only when SMTracker finds one; otherwise it reads *Use GPU: unavailable* with the reason, which in the packaged app is *torch is not installed in this application*. The classical segmenter always runs on the CPU. | from **Use the GPU when one is found** in Options (on) |
| **Ignore** / **Include** | Leaves a condition out of **Segment all ready** and the Batch, and turns its **Segment** button off. SMTracker remembers it. | included |

> **Check this.** **Replace existing outlines** overwrites the outlines you corrected in Edit outlines, with no copy kept. To redo only some fields, delete their files in `cell_meshes/` and run with the box cleared.

With the box cleared, a stopped run continues where it left off, and the row's button reads **Continue**. **Replace existing outlines** starts cleared each time you open the window. A GPU speeds the Omnipose models up a lot.

#### What a run writes

Each field gets one outline file, `cell_meshes/<field>.mat`, in Oufti's format, with one outline per cell. If the condition already has an outlines folder spelled differently, such as `cell meshes`, SMTracker writes there.

A field where no cells are found, or that fails, gets no file and stays on the list to segment. With **Replace existing outlines**, its old outline file stays.

#### Go on to Edit outlines

**Next: Edit outlines** lights up once nothing is running and every condition has as many outline files as fields. Ignored conditions need their outlines too. Fields left without cells keep it off until you draw their outlines by hand.

### Classical settings

*Tune the classical segmenter on one field, with a live preview, before you segment with it.*

Tune the classical segmenter on one field at a time, and see the result as you move the sliders. The classical segmenter finds glowing cells in the `sum/` projection by threshold and shape. It needs no GPU and no training, and it's the one to use when SMTracker can't run the Omnipose models ([Segment](#segment)). It's fast, but weaker than Omnipose on some data.

The settings only matter for conditions whose **Model** on [Segment](#segment) is *Classical, sum image (no GPU, no training)*. Skip this page if you segment with an Omnipose model.

#### Preview a field

The page shows one field's projection with what the classical segmenter finds in it.

##### Choose a field

Choose the **Condition** and the **Field**. The arrows beside **Field** step to the previous or next field, and wrap around at the ends. The list holds every field with a projection in `sum/`, written by [Calibration & save](#calibration-save).

A condition without projections has no fields, and the preview reads "Open a folder and pick a condition." Save it on Calibration & save first.

##### Move the sliders

Move a slider, and the preview redraws when you pause. Cyan lines are outlines of kept cells, and a red cross marks each candidate SMTracker dropped. The lines under the preview count the cells and list the most common reasons for dropping, for example *Dropped: 15.2 px wide*.

##### Use the settings

Click **Use these settings**. Segment and Batch then use them. The note under the button says whether what you see is saved.

The preview writes nothing, and moving the sliders changes no run until you click **Use these settings**.

#### The settings

Each slider changes the preview as you move it. Start with **Cell width**; most people leave the rest near their defaults.

| Setting | What it does | Default |
| --- | --- | --- |
| **Cell width** | The width of a cell in this image, in px. Candidates much thinner or wider are dropped (outside 0.55 to 1.45 times this width), and touching cells are split relative to it. *B. subtilis* is about 9 px wide at 106 nm pixels. | 9 px |
| **Sensitivity** | The threshold, as a multiple of Otsu's automatic level. Lower finds dimmer cells and more background; higher finds only the brightest cells. | 1 x |
| **Split touching cells** | How deep the waist between two touching cells must be before they're split, as a fraction of half the **Cell width**. Lower splits more readily; higher keeps chains whole. | 0.3 |
| **Minimum length** | Candidates shorter than this are dropped, in px. Small specks are removed before that, without a cross. | 12 px |
| **Maximum length** | Candidates longer than this, such as a chain of cells left whole, are dropped, in px. All the way left means no maximum. *B. subtilis* is about 20 to 50 px long at 106 nm pixels. | none |
| **Grow to the edge** | Grows each kept cell outward by this fraction of the cell width, so the outline reaches the cell's edge. The threshold alone finds only the cell body. | 0.2 x width |
| **Smoothing** | A blur before the threshold, in px. More joins patchy fluorescence into one body; too much joins neighboring cells. | 1 px |

Set **Cell width** to your cells' width in pixels first, and most of the rest follows. Then adjust **Sensitivity** until dim cells are found but the background isn't. If chains of cells come out whole, lower **Split touching cells** rather than setting a **Maximum length**, because a maximum drops the whole chain instead of splitting it.

Raise **Minimum length** if debris or small fragments are kept as cells. Lower **Grow to the edge** if outlines spill past the cell's edge, and raise it if they sit inside the cell. It grows by whole pixels, so a small value can round to no growth at all: at a 9 px width, 0.05 grows by 0 px. Raise **Smoothing** only for patchy fluorescence that splits one cell in two.

Some rules aren't on the page. Before thresholding, SMTracker flattens the background. Candidates must be solid enough to be one cell, and a candidate that overlaps a bright band along the image's edge is dropped.

#### Which runs use them

**Use these settings** saves one set for every classical run, on Segment and in Batch, in every folder and later session. The same set applies to every condition.

A Batch takes the saved set when it starts. On Segment, each condition takes it when its own run starts, so conditions still waiting in the queue pick up a set you save meanwhile.

**Reset to defaults** moves the sliders back to the defaults but doesn't save them: the note reads *Not saved yet*, and Segment and Batch keep using your saved settings until you click **Use these settings**.

#### The preview and a run

The preview is close to what a run writes, but not identical. A run also drops very small outlines and small outlines at the image's border, and fits each cell's outline at a fixed width rather than at **Cell width**. So a run can write a few cells fewer than the preview counts, and their outlines can differ slightly.

### Edit outlines

*Check the automatic outlines field by field, and correct them by hand: split, merge, draw, delete, reshape.*

**Edit outlines** is where you check the outlines field by field and fix what the model got wrong. An outline that's too tight, or two cells drawn as one, changes which tracks the analysis keeps. Plan to look at every field before you analyze.

> **Check this.** Outlines drawn too tight drop the free molecules near the membrane and raise the slow fraction. Correct them here, then check the **Near edge** column in [QC](#qc).

#### Correct a condition

Work through one condition at a time: open it, step through its fields, and save.

##### Choose a condition

Click **Edit outlines** in the sidebar, then a condition in the sidebar's list. Its fields appear in the **Fields** list, and its first field opens. A field without an outline file reads *(no outlines yet)*.

##### Step through the fields

Press `←` or `→`, or turn the mouse wheel over the image, to go to the previous or next field. Look at each one, and fix what's wrong with the [tools below](#fix-the-outlines).

##### Save

Press `Ctrl+S`, or click **Save field**, to save the field on screen. Click **Save all** to save every field you've edited, in every condition. On a Mac, `Ctrl` is probably `Cmd`.

Edits stay in memory until you save, even when you move to another field or condition. You can't open another folder or go back to the menu while edits are unsaved, and closing the window asks whether to leave them.

#### Fix the outlines

Each tool has a key and a button in the tool bar above the image. The tool you choose stays on until you choose another, and the **Keys** card under the image lists the shortcuts.

| Key / button | What it does |
| --- | --- |
| `S` **Select** | Click a cell to select it, then drag its body to move it, a round pole handle to lengthen it, a square bend handle to reshape it, or an edge diamond to move that edge alone. |
| `D` **Draw** | Draw a cell: click one pole, then the other. The new cell takes the **Width**. `Esc` cancels after the first click. |
| `X` **Split** | Click a cell where it should be cut. Each half must be at least as long as the cell is wide. |
| `M` **Merge** | Merge the selected cell with the cell whose nearest pole is closest. The two must touch: poles at most 12 px apart. |
| `Del` **Delete** | Delete the selected cell. `Backspace` works too, and right-clicking a cell deletes it in any tool. |
| `B` / `N` **Add bend** / **Straighten** | Add a bend to the selected cell, or remove one. |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo. Each field keeps its own history of up to 50 steps; a whole drag is one step. |
| `Esc` | Cancel a draw, or deselect and go back to Select. |

When SMTracker refuses an edit, the status line under the image says why, for example that two cells don't touch. A cell you've drawn, split, merged or reshaped is redrawn as a smooth capsule and marked with `*` in the **Cells** list. Moving a whole cell doesn't.

**Width** sets the width of the next cell you draw, from 1 to 60 px (9 px by default). Typing in it while a cell is selected resizes that cell with every keystroke, so typing 12 first makes the cell 1 px wide. Selecting a cell doesn't show its width in the box.

Shortcuts don't work while the cursor is in the **Width** box. Press `Enter`, or click the image, to use them again.

#### See the image better

These change only what you see; nothing is saved.

| Key | What it does |
| --- | --- |
| `1` `2` `3` | Switch views: phase, the projection, and an overlay of the two (more with more channels). |
| `C` | Cycle the **Stretch**: *imagej* (what ImageJ shows), *septa* (septa stand out), *outline* (outlines stand out). |
| `R` | Reset brightness and contrast to 100 %. |
| `[` / `]` | Contrast down or up by 15 %. |
| `+` (or `=`) / `-`, `F` | Zoom in or out; fit the field to the window. |
| `Ctrl` + wheel | Zoom in or out around the cursor. The wheel alone steps through the fields. |
| `F11` | Focus mode: the image fills the screen, with a slim bar of tools. |

Drag with the middle mouse button to pan.

**Bright** and **Contrast** change the image on screen, and in the overlay only the phase image underneath (both 100 % by default; `R` or **Reset** sets them back). **Tint** and **Alpha** set the color and strength of the fluorescence in the overlay (**Alpha** 50 % by default).

Click **Channels…** to open **Image channels**, where you choose the folder the outlines are drawn on (*phase* by default) and up to three fluorescence folders shown beside it, such as *sum* or *gfp*. SMTracker remembers them. Choosing another folder to draw on reads every field again, so save your edits first: with unsaved edits, SMTracker refuses the change and closes the dialog.

#### Check the alignment

An outline that sits beside its cell's fluorescence makes the analysis keep the wrong localizations, because the analysis keeps only those inside an outline. So when a field opens, SMTracker checks whether the outlines sit where the fluorescence of the first fluorescence folder is, by sliding them up to 8 px each way.

The note under **Shift to fluorescence** gives the result: *aligned*, *cannot tell* (too little signal), *shifted by* a number of pixels, or *alignment not checked*. It stays empty when the field has no fluorescence image or no cells.

If the offset is more than 3 px, the button reads **Shift all by (dx, dy) px**. Click it to move every outline by that offset at once, in one undo step. Otherwise it stays off.

#### What saving writes

Saving a field writes:

- `cell_meshes/<field>.mat`, the outlines in Oufti's format, with the cells numbered again in list order, and any outlines without ribs after them, unchanged;
- `cell_masks/<field>.tif`, the same outlines as a label image, when SMTracker can read the image's size;
- `cell_meshes/<field>.mat.curated.json`, a marker that you corrected the field by hand, which the sidebar counts as *corrected*;
- the first time, if the field had an outline file, a copy of the automatic outlines in `cell_meshes_auto/`.

> **Check this.** `Ctrl+S` saves the field on screen even if you haven't changed it, and marks it as corrected, so save only fields you've checked. On a field with no outlines where you've drawn nothing, it saves nothing, and the status line says so. If you delete every cell of a field, saving writes an empty outline file, and the analysis then drops that movie's tracks.

Outlines from other tools that store only a contour, without the cross-lines (ribs) of Oufti's format, can't be edited. The status line names them when the field opens. Saving keeps them in the file as they are, so the analysis still uses them.

#### When you're done

When every field is saved, click **Done, back to the menu**, and open the folder in the analysis. To train a model on your corrections first, see [Train a model](#train-a-model).

### Train a model

*Fine-tune a segmentation model on your outlines, on this computer or on a GPU machine elsewhere, and add it to Segment's model list.*

Train a segmentation model on outlines you've corrected, so the next folders need fewer corrections. You fine-tune Cellpose-SAM (or, run from source, an Omnipose model) on your own data, here if this computer can train, or on another machine such as Google Colab. Skip this page while the pretrained models or the classical segmenter outline your cells well enough.

Training happens in a separate window, **SMTracker · model trainer**. Click **Train a model** in the outlines window's sidebar, then **Open the trainer**.

#### What training needs

Training needs two things: outlines worth learning from, and a computer that can train.

- **Outlines to learn from.** The trainer uses every outline file in a condition's `cell_meshes/` folder that has a matching image, whether you corrected it or not, and its list calls them all *corrected*. So correct the fields in [Edit outlines](#edit-outlines) first, and remove the outline files of fields you haven't checked. Fields whose files hold no outlines are skipped.
- **To train here:** the [segmentation pack](#install-and-open-smtracker) with GPU support and an NVIDIA graphics card with about 6 GB of memory or more. Cellpose-SAM is a large network: a 4 GB card runs out of memory, and SMTracker says so before it starts. Otherwise, build a training set and export it to a machine that can train, such as Google Colab. Fine-tuning an Omnipose model needs SMTracker run from source with Omnipose.

A note under the trainer's buttons says whether this computer can train, and if not, why. No minimum number of fields is established for this data; the build's summary says how many cells you're training on.

#### Build and train a model

Train in six steps, from your conditions to a model in Segment's list.

##### Open the folder

The trainer opens on the outlines window's folder, if one is open. Otherwise click **Open folder…** and choose the folder that holds your conditions. The trainer lists each condition with its number of fields, all selected. Clear the ones you don't want to train on.

##### Choose what to train on

Leave **labels from** at *cell_meshes (what you corrected)*, and **images in** at *phase* for phase-contrast snapshots, or choose *sum* for the projections. The model then reads the same kind of image on Segment.

##### Choose how to train

Leave **start from** at *bact_phase_omni* (or choose *bact_fluor_omni* for projections). Leave **hold out** at 0.2 and **epochs** at 100 for a first model.

##### Build the training set

Click **Build training set**. Its label also counts the fields of the selected conditions. SMTracker writes the images, labels and held-out fields into `training_set/` in the folder you opened, and the trainer's log summarizes what's in it.

The settings are fixed when you build. If you change one afterward, build again.

##### Train

Click **Train here** if this computer can train. The log says when it has started and when it's done, with the path of the new model file, under `training_set/models/`. There's no progress bar and no per-epoch output, and a run can't be stopped.

Otherwise, click **Export to…** and choose a folder, for example on Google Drive. The exported `training_set/` holds its own `train.py`, a README and a Colab notebook, and doesn't need SMTracker. Train there, then download the model file.

##### Add the model to SMTracker

Click **Save a model file…** and choose the trained model file. Type a name, and a short description of what it's for, when SMTracker asks. SMTracker copies the model into its model store: `%APPDATA%\SMTracker\models` on Windows, or `~/.config/smtracker/models` on macOS and Linux.

Then open the folder again in the outlines window. The model appears in each condition's **Model** list on Segment, as *(trained, reads phase)* or *(trained, reads sum)*.

To remove a saved model, select it in the trainer's list of saved models and click **Delete**. SMTracker asks first, then deletes it from the store.

Training doesn't add a model anywhere by itself: until you save it, it only exists as a file in `training_set/models/`.

> **Check this.** **Save a model file…** records what the model reads from the training set built in this trainer session. If you trained elsewhere and haven't built the same set again in this session, the model is listed as *(trained, reads phase)*, even if it was trained on projections. Build the same training set again before you save it.

#### The settings

| Setting | What it does | Default |
| --- | --- | --- |
| **labels from** | Where the labels come from: *cell_meshes (what you corrected)*, the outline files, or *masks/ (label images)*, images in a `masks/` folder where each cell's pixels hold its own number. | *cell_meshes (what you corrected)* |
| **images in** | The folder of images the model learns from, and later reads on Segment: *phase*, *sum*, or any folder name you type. With *phase* and no `phase/` folder, the snapshots beside the movies are used. | *phase* |
| **start from** | The model to fine-tune: *bact_phase_omni*, *bact_fluor_omni*, or *: train a NEW model from scratch:*. Fine-tuning never changes the starting model; it writes a new file. | *bact_phase_omni* |
| **hold out** | The share of fields kept out of training, from 0 to 0.5, so the new model can be compared with its starting model on fields neither has seen. With two or more fields, at least one is always held out; with one field, none is. | 0.2 |
| **epochs** | How long to train, from 1 to 2000. | 100 |

Fine-tune rather than train from scratch. Cellpose suggests a new model needs far more labeled images than a fine-tune, hundreds rather than tens.

#### Check the new model

The trainer doesn't score a model itself, so every model saved from it is listed as *not scored against its base model*. To compare the new model with its starting model, segment a few fields you didn't train on with both models, and compare them in [Edit outlines](#edit-outlines). The exported Colab notebook also has a scoring cell, for fine-tuned models.

#### Limits of the trainer

The trainer has a few rough edges. Work around them like this:

- It finds conditions one level below the folder you open. Open the folder directly above your conditions, not a condition itself or a folder two levels up.
- Building again writes into the same `training_set/` without clearing it, so images, held-out fields and models from an earlier build stay, and are trained on. Delete `training_set/` before you build a smaller set. **Export to…** also merges into an earlier export.
- Closing the trainer while it trains freezes the window until the training has finished.

### Batch and Options

*Segment the conditions of many folders in one run, and set the model and GPU that every session starts with.*

Segment many folders in one go, without opening each one, and set the standards the Cell outlines window starts from. Use Batch once your conditions have their phase snapshots or projections, when segmenting them one at a time on [Segment](#segment) would keep you at the computer.

#### Batch

**Batch** needs no open folder. It segments every condition it finds under the folders you add, one after another. Unless you select **Replace existing outlines**, it leaves the fields that already have outlines alone.

##### Add the folders

Click **Add parent folder…** and choose a folder that holds condition folders, or one condition folder. Repeat for each folder. The table lists every condition it finds, with its fields, the fields without an outline, its model and whether it's ready. A folder with nothing to segment is one row, saying why; if it holds whole projects one level down, add each project as its own folder.

##### Check the model and the GPU

To segment with another model, change **Model for a new condition** in [Options](#options) first. The batch uses that model for every condition, and the folder that model reads, not the model or folder set on a Segment row. The line under **Replace existing outlines** names it.

The batch uses a GPU only if **Use GPU** is selected on Segment. SMTracker looks for a GPU when a folder opens with at least one condition, so in a fresh window the batch runs on the CPU. To use the GPU, open a folder first.

##### Run

Click **Run batch**. It's available once there's a folder, a model this copy of SMTracker can run, and at least one ready row. SMTracker lists the folders again as it starts, so it runs what's on disk at that moment. Each row's **Status** shows how far that condition has come, and the log lists every field as it's done.

To stop, click **Stop after the field in hand**. The field being segmented finishes, and nothing new starts. Conditions that hadn't started read *not run: stopped*. Closing the window stops the batch the same way.

##### Settings

These apply to every condition in the batch.

| Control | What it does | Default |
| --- | --- | --- |
| **Replace existing outlines** | Selected: every field is segmented again, even fields you corrected by hand. Cleared: only fields without an outline are segmented. It's separate from the checkbox of the same name on Segment. | cleared |
| **Ignore** / **Include** | Leaves a condition out of the batch. SMTracker remembers it by the condition's folder, and the same list applies on Segment. | included |

> **Check this.** With **Replace existing outlines** selected, the batch overwrites every hand-corrected field in every folder you added. Leave it cleared unless you mean to start over.

A condition's *layout* is how its movies are filed; a line under the table names the conditions not filed in `phase/` and `raw/`. If the model is the classical segmenter, the batch uses the settings saved on [Classical settings](#classical-settings). The list of parent folders isn't remembered: a new window starts with none.

While a batch runs, you can't segment on the Segment page, open another folder or go back to the menu; and a batch can't start while Segment is running. On Windows, SMTracker keeps the computer from sleeping until the batch ends; on a Mac or Linux, nothing does, so turn off sleep for a long batch. Unlike the [tracking Batch](#batch), this batch writes no report file. The log and the note under the table say what ran, and a pop-up too when the batch ran longer than **Pop-up when a job ran at least**.

#### Options

Click **Options** in the header to open **Cell outline options**, the *standards* a new session starts from. Saving new standards changes only the values still at the old standard; a value you set on a page stays.

| Option | What it does | Default |
| --- | --- | --- |
| **Pop-up when a job ran at least** | A job that runs at least this long, in seconds, ends with a pop-up and a taskbar flash. 0 shows a pop-up after every job. It's the same setting as in the [preparation options](#preparation-tracking-options). | 60 s |
| **Model for a new condition** | The model each condition starts with when you open a folder, and the model every condition uses in Batch. You can change it on each condition's row on Segment. | *bact_phase_omni (reads phase)* |
| **Use the GPU when one is found** | Selected: **Use GPU** on Segment starts selected when a GPU is found. Clear it to run on the CPU while the GPU is busy with something else; that's much slower. | selected |

Saving a new model moves every open condition still on the old standard model to the new one, except the one being segmented. A condition whose model you chose on its row keeps it. A condition that moves loses a folder you chose for it by hand, and the Segment log says so.

Click **Save** to keep your changes, or **Cancel** to discard them. **Reset to defaults** sets every field back, but writes nothing until you save; saving then also resets the pop-up time for the preparation window.

## 4. Analysis

**Analysis** is where you compare your conditions: it loads the tracks and cell outlines, checks them, fits the populations, and makes the figures. It's the third of SMTracker's three stages (the first window also offers simulated data). You need it for every project, once the movies are tracked and the cells are outlined.

#### Open it

In SMTracker's first window, click the **Analysis** card, or press `Return`: it's the default. For choosing the folder and checking what SMTracker read, see [Opening a project](#opening-a-project).

The analysis needs, for every condition:

- **tracks**, from u-track (`TrackingPackage/tracks/`), TrackMate (`trackmate/*.xml`) or another tool's track tables ([Tracks from other tools](#tracks-from-other-tools));
- **cell outlines** (`cell_meshes/`), because the analysis keeps only tracks inside a cell. A movie without them is left out ([When something is missing](#opening-a-project)).

It doesn't need the raw movies, but it reads them where they are: for the camera's frame interval, for the dates that propose the replicates, and for the *Movie* background in [Images & tracks](#images-tracks).

#### The order of work

Work through a project in this order. Each step reads what the one before it produced.

1. **Open the project**, check its replicates, and open the workspace.
2. **Run QC** before you compare anything.
3. **Run the analyses** you need, SQD first.
4. **Export the figures**, one chart at a time or together in the Composer.

#### The pages

The workspace's sidebar groups its sections into **Data**, **Analyses** and **Figures**. For how the window is laid out, see [The workspace](#the-workspace).

| Page | What you do there |
| --- | --- |
| [Opening a project](#opening-a-project) | Choose the folder, say what its folders mean, pick the tracker and confirm the replicates |
| [The workspace](#the-workspace) | Find your way around the window, run a section and choose its uncertainty |
| [QC](#qc) | Check every condition before you compare them |
| [Images & tracks](#images-tracks) | See the sample size, and each movie with its outlines and tracks |
| [SQD: Populations](#sqd-populations) | Fractions and apparent D from the squared displacements, the main analysis |
| [SQD: Jump distances](#sqd-jump-distances) | The jump-distance distribution |
| [MSD](#msd) | Plot mean squared displacement against lag, as a cross-check |
| [Apparent Diffusion](#apparent-diffusion) | One apparent D per track, and the populations they form |
| [HMM](#hmm) | Fit states with memory: switching rates, the slow state, jump angles |
| [Spatial Map](#spatial-map) | Where in the cell each kind of motion happens |
| [Kinetics](#kinetics) | Populations over time after a trigger |
| [Clustering](#clustering) | Tracks grouped by the shape of their MSD |
| [SMTracker 2.0 view](#smtracker-2-0-view) | See what SMTracker 2.0 would have said about the same data |
| [Figures](#figures) | Exporting charts, significance, and the Composer |
| [Analysis options](#analysis-options) | The standards a new analysis starts from |

#### What it writes

The analysis never changes your movies, tracks or outlines. It adds:

| Written | Where | When |
| --- | --- | --- |
| `day_map.json` | Each condition's folder | **Open workspace**, where the grouping is new or changed |
| `smtracker_design.json` | The folder you opened | **Use this design** on **What do the folders mean?** |
| `.smtracker_warm/` | The folder you opened | When the project is read, so it opens faster next time |
| `kinetics_additions.json` | Each condition's folder | A **Kinetics** run with addition times per sample or per culture |

Section results are kept in SMTracker's own cache on your computer, not beside your data. Figures, tables and exported tracks go where you choose.

### Opening a project

*Open a folder of tracked conditions in the analysis, say what its folders mean, and check its replicates before the workspace opens.*

Open a folder of tracked conditions and check what SMTracker read from it, before anything is fitted. You do this each time you start the analysis. A project opened before is read back from its cache, so it opens much faster the second time.

Nothing is fitted until the workspace opens, and your movies and tracks are never changed. For the files SMTracker adds, see [What it writes](#4-analysis).

#### Choose the folder

In SMTracker's first window, click the **Analysis** card (or press `Return`). On **Open a project folder**, click **Choose folder…** and choose the folder that holds your condition folders, or one condition folder. You can also drop one folder onto the window.

**Recent projects** lists the last 10 folders you opened, with their conditions and series as counted last time. Click one to open it again. The × beside it removes it from the list and leaves the folder untouched. A folder that can't be found, for example on an unplugged drive, is grayed out.

SMTracker then looks through the folder. A page lists each condition as it's found, with a progress bar once it reads the tracks. Click **Cancel** to go back.

Up to three pages can follow, depending on your data: **What do the folders mean?**, **Choose the tracker**, and **Confirm project**.

#### What do the folders mean?

This page appears when your movies sit two or more folders below the one you opened, for example `data/ProteinA/glucose/…`. It also appears when a saved design no longer fits the folders, or when the movies sit at different depths. It shows one row per folder level, from the top down, with example folder names. Choose a role for each level:

| Role | What it means |
| --- | --- |
| *Factor* | Something you compare, such as a protein, a medium, a strain or a treatment. Give it a name. A condition is one combination of factor values, such as "ProteinA glucose". |
| *Replicate* | Each folder is one replicate (an independent culture) of the condition above it, whatever its name, a date included. You can mark only the level directly below the last factor as *Replicate*. |
| *Skip* | A folder that only wraps the next level, such as `results`. Its name isn't used. |

SMTracker proposes the roles, or starts from one of your saved designs if it fits. Factor names must differ, and a few names the analysis uses itself, such as *day* or *cell*, are refused. Under the rows, it shows the conditions the design produces, or why it can't load the folders. The factors you name here are the ones you can group charts by in the workspace.

- Click **Use this design** to save it with the project, as `smtracker_design.json` in the folder you opened. To reuse it in later projects, type a name in the **Save to my designs as** field first; it then appears under **Saved designs:**.
- Click **Load without a design** to load the folder with SMTracker's automatic rules ([Organizing your folders](#organizing-your-folders)). SMTracker doesn't ask again for that folder on this computer.

A saved design always wins over the automatic rules. If it no longer fits, SMTracker asks again. Click **Load without a design** then, and SMTracker renames the file to `smtracker_design (not used).json`.

#### Choose the tracker

This page appears when a condition holds tracks from more than one source, for example u-track and TrackMate, or when the conditions come from different sources. For each condition with a choice, choose which tracks to read, then click **Open**. SMTracker never picks one for you.

On the same movies, u-track and TrackMate differ by about 1 percentage point in the slow fraction and 10 to 18 % in D. Compare conditions tracked the same way. When the conditions end up read from different trackers, the page says so.

#### When something is missing

- **Movies tracked without a pixel size or frame interval** can't be analyzed: every D depends on both. For u-track movies, SMTracker names them and asks; click **Leave out and open the rest** to go on without them. Other track tables without them are refused. See [Frame interval and pixel size](#frame-interval-and-pixel-size) to keep them.
- **Movies without cell outlines** are left out, and a notice names them. If no movie has outlines, the folder is refused. To include them, outline them in [Cell outlines](#3-cell-outlines) and open the folder again.
- **When nothing can be opened**, the page says why and what to do. For example, the movies aren't tracked yet, or the track tables don't state their units. See [When something is refused](#when-something-is-refused).

#### Confirm project

**Confirm project** shows what SMTracker read, condition by condition, before anything is fitted. Click a condition's tab to see it. Two buttons at the right of the tabs switch every condition between two views: *Record*, what the files recorded, and *Groups*, how the movies are grouped into replicates.

##### Record

The top card gives the condition's **Pixel size**, **Frame interval**, **PSF width**, **Search radius** and **Localization precision**, as recorded. A pixel size, frame interval, PSF width or search radius the files didn't record reads "–", never a default. A localization precision that wasn't recorded shows the theoretical value, and its caption says so. When conditions differ, a note under the card says how SMTracker reads them:

- pixel sizes within 1 % load at the first condition's size, and the card gives the shift in D; further apart, the folder is refused;
- frame intervals up to 10 % apart, within a condition or between conditions, are read at one interval, with the steps rescaled so D stays comparable. Change the limit with **One frame interval for movies within** in [Analysis options](#analysis-options);
- search radii stay each condition's own, because the radius caps the fastest D a track can show.

The **Replicates** table lists, per replicate, its series, imaging date, frames, cells, and tracks kept out of those found. A **Note** flags a replicate whose movies have an unusual number of frames. An imaging date read from when u-track imported a movie is grayed out, because it isn't the date the movie was taken.

##### Groups

The **Analysis groups** board shows how the condition's movies (*series*) are grouped into replicates, as cards in one lane per group. The counts above it give the series, cells and tracks that will go into the fit.

> **Check this.** On a first open, the board proposes one group per imaging day ([How replicates are counted](#how-replicates-are-counted)). **Open workspace** saves these groups as the replicates, so check them on the board first.

To change the groups:

- Drag a series card to another lane, or select several and click **Move to…** in the bar that appears.
- Click **Regroup…** to cut the series into a number of groups at once, by series number or in groups of a fixed size, with a preview.
- Clear a series' checkbox to hold it out of the fit. It keeps its group, so selecting it again puts it back.
- Click **Reset to replicates** to go back to one group per imaging day.

**Filter** shows only some series, for example those held out, and doesn't change the fit. The **Cards** / **Table** switch changes the view. **Hide** folds the board away, and **Show** brings it back.

##### Pair replicates

Select **Pair replicates** when replicates of different conditions belong together, for example one culture split into two media, or one imaging session. Each replicate then gets a **Pair** tag, and replicates of other conditions with the same tag are its partners. Only **Paired t (declared pairs)** reads the tags. Without pairs, every replicate stands alone, and the paired test isn't offered.

**Propose from imaging days** gives replicates imaged on the same date the same tag. It's a proposal, and it replaces every tag: check each one.

##### Colors and order

Each condition keeps one color in every chart. Click a tab's color dot to choose another, or **Custom colour…** for any color. The **Palette** field above the tabs changes the whole set (Okabe-Ito by default), and **Make a palette…** builds your own. Drag a tab to change the conditions' order, which the workspace keeps ([The workspace](#the-workspace)).

#### Open the workspace

The line at the bottom says whether anything needs your attention in the condition on show, such as series not in a group. Click **Open workspace**.

SMTracker applies the board and saves each condition's grouping, held-out series and pair tags as `day_map.json` in its folder, where they're new or changed, so the next open starts from them. Click **Cancel** instead to go back without saving the board; colors you chose are kept.

The [workspace](#the-workspace) opens on [QC](#qc).

### The workspace

*Find your way around the analysis window: run a section, choose how its uncertainty is measured, add pages, order the conditions, and use the project menu.*

The workspace is the analysis window: every section, its Run, and its figures live here. It opens on [QC](#qc) when you click **Open workspace** on **Confirm project**. Most of what this page describes is shared by every section, so the section pages don't repeat it.

#### The layout

The window has three columns: the sidebar, the page, and the right rail.

- **The sidebar**, on the left. Its header gives the project's name and its number of tracks, with the track cap when one is set. Click the name to open the [project menu](#the-project-menu). Below it come three groups:
  - **Data**: **QC** and **Images & tracks**, for looking at the data before any fit.
  - **Analyses**: **SQD**, **MSD**, **Apparent Diffusion**, **HMM**, **Spatial Map**, **Kinetics** and **Clustering**, then **SMTracker 2.0 view**, which opens [SMTracker 2.0's tabs](#smtracker-2-0-view) in a window of their own.
  - **Figures**: the **Composer** button ([Figures](#figures)), at the foot, under **Running**.
- **The page**, in the middle, under a breadcrumb such as "SQD > Jump distances". Click an earlier crumb to go back to it. On a condition's own page, the last crumb opens a list of the other conditions to switch to.
- **The right rail**, with the tools for the page on screen ([The tools on the right](#the-tools-on-the-right)).

The button left of the breadcrumb folds the sidebar to icons, and opens it again. Click a section to open it, on the page you last used there.

##### A section's pages

A section with several pages shows them as a strip at its top:

| Section | Pages |
| --- | --- |
| SQD | *Populations*, *Jump distances* |
| MSD | *Curves*, *Checks* |
| Apparent Diffusion | *Populations*, *Distributions*, *Per cell*, *Checks* |
| HMM | *State diagrams*, *Slow state*, *Jump angles*, *Table* |

QC, Images & tracks, Spatial Map, Kinetics, Clustering and the Composer have one page each.

#### Run a section

Nothing is computed until you ask. Before a section's first result, its page shows what the section does and a **Run analysis** button. The **Compute** mark at the top of the right rail does the same from any page.

##### Open the settings

Click **Run analysis** or the **Compute** mark. Every section except QC opens its settings first, in a dialog such as **Configure SQD**. QC runs at once, because it has no settings.

##### Choose and run

Choose the settings, then click **Run**. Click **Cancel** to close the dialog without running.

Each setting has a line of help under it. The dialog starts from the last Run's settings, or the first time from the standards in [Analysis options](#analysis-options), where the section has any. SQD, Apparent Diffusion and HMM have two tabs: **General**, and **Fits**, with one row per condition. A Run computes every loaded condition, except on SQD and HMM, where the **Fits** tab chooses which conditions are fitted.

While a first Run computes, the page says **Computing…** with its progress, and the **Compute** mark spins. A later Run dims the chart in place until the new result is ready. To stop a Run, point at the mark and click its stop square.

##### Running

Each section runs on its own, with its own results and messages, so you can start SQD and go on to MSD while it fits. **Running**, at the foot of the sidebar, lists every job with its progress and the conditions done. Click a job to go to its page.

When a job ends, a note says how long it took, or why it couldn't be computed. If you're on another page, **Show** takes you to the job's page. If you run a page again while it's still computing, SMTracker asks first: **Proceed** stops the running job and starts the new one.

The work runs on every core but one, below normal priority, so the computer stays usable. On a computer with four or more cores, a quick job such as a preview isn't held up behind a long fit.

##### Uncertainty

SQD, Apparent Diffusion, HMM and MSD ask how each condition's spread is measured, under **Uncertainty**. On SQD, Apparent Diffusion and HMM, the choice changes the point itself as well as its bar. [MSD](#msd) always shows the pooled D, with the spread beside it.

| Choice | What each point and bar shows | Use it for |
| --- | --- | --- |
| *Replicates* (the default) | Each replicate fitted on its own; the mean ± SD of the replicates. The SD is a spread between replicates, not a confidence interval. | Data with replicates. |
| *Movies* | The pooled fit, with a 95 % interval from resampling movies. Slow. | One imaging session only. |
| *Cells* | The pooled fit, with a 95 % interval from resampling cells. Slow. | One imaging session only. |
| *Chunks* | The movies dealt at random into chunks, each fitted as a replicate; their mean ± SD. | Data without replicates. A condition with replicates is drawn as its pooled fit. |
| *None* | All replicates pooled into one fit, with no interval. | A quick look. |

With two or more replicates, the *Movies* and *Cells* intervals are withheld, because movies and cells of one replicate aren't independent. MSD offers no *Chunks*.

The number of chunks is at most 5 by default, from 2 to 20: **Chunks per condition** on SQD, the *chunks* field beside **Uncertainty** on Apparent Diffusion and HMM. Each chunk holds at least 3 movies. A condition whose movies can't fill two chunks is dealt by cells instead. Chunks show how precisely these cells were measured, and say nothing about another culture.

#### When results are kept

Every result is kept in SMTracker's cache on your computer, so a Run you've done before comes back at once, after a restart too. A result is kept per condition: changing an exclusion, an outline or a replicate of one condition recomputes only that condition. A setting shared by every condition recomputes them all. Shared clusters on Clustering is one result over all its conditions.

A result stays on screen until you run that section again, or until the window is rebuilt: switching the theme, changing QC's track cap, or applying exclusions. After a theme switch, SQD's figure comes back by itself, and running another section again returns its result from the cache. A new track cap or new exclusions change the tracks, so the affected conditions are computed again.

#### The tools on the right

The right rail holds the tools that act on the page on screen. From the top:

| Tool | What it does |
| --- | --- |
| **Compute** mark | Runs the page on screen, opening its settings first, or stops its Run. Its hover says which, for example *Run SQD…*. Dimmed on a page with nothing to compute. |
| **Visibility** | Chooses which of the page's figures are drawn. Shown only where there's a choice; a dot marks it while a figure is hidden. |
| **Significance** | Chooses which conditions are compared, the test, and the brackets on the chart ([Figures](#figures)). Shown only on SQD's and Apparent Diffusion's *Populations* and on HMM's *Slow state*. |
| **Conditions** | Their colors and order, in every section ([below](#the-order-and-colors-of-the-conditions)). |
| **Metadata** | The settings and replicates of the result on screen, with **Copy**. SQD and the Spatial Map print it; the other sections don't yet. |
| **Help** | Offers the manual on the website or the copy that came with SMTracker (no network needed), both in your browser at the Analysis chapter. The two hold the same text. Also **Check for updates**. |
| **SMTracker 2.0 view** | The clock icon. Opens SMTracker 2.0's tabs on these conditions, in a window of their own ([SMTracker 2.0 view](#smtracker-2-0-view)). |
| **Options** | Opens [Analysis options](#analysis-options). |
| **Theme** | Switches between light and dark. |

Each chart has its own **Axes** and export buttons, described in [Figures](#figures).

#### Added pages

An added page is a second copy of a section with its own settings and results, for example to fit the same conditions with three populations and with four.

Point at an analysis row and click its **+**, or right-click the row and choose **New page**. The new page (for example "SQD 2") starts with the settings of the page you're on and none of its results, and opens at once. Click **Run analysis** to compute it. QC, Images & tracks and the Composer have no added pages.

The added pages are listed under their section. One page is on screen at a time: click a row to switch to it, and click the section's own row to go back.

- **Rename**: double-click the row and type. `Return` keeps the new name; `Escape`, or clicking away, keeps the old one.
- **Reorder**: drag the row up or down, or right-click it and choose **Move up** or **Move down**. The section's own page stays first.
- **Delete**: the **×** on the row, or right-click. The page's running job stops, and **Undo** brings it back with its settings and results for 8 seconds. Only the last deleted page can be brought back.

Exported figures from an added page carry its name in their file name, and sessions keep every page. On Spatial Map, **Use the same D as the … fit** has a list of which page supplies the D. **Map this fit** on a fit's page makes a new Spatial Map page from it, and opens its settings.

#### The order and colors of the conditions

Set the order and colors that every chart, table and exported figure uses. Click **Conditions** on the right rail. The first time, the order is the one you set on **Confirm project**; once you apply an order, it's remembered for the project.

- **Order**: drag a condition by its grip, or use its arrows. A bar then reads "New order, not applied yet": click **Apply order** to apply it, or **Cancel**. Nothing is fitted again. A panel already in the Composer keeps its old order until you add it again.
- **Colors**: click a condition to choose one of the palette's colors (taking another condition's color swaps the two), or **Custom colour…**. The palette above the list changes the whole set. Okabe-Ito is the default where it has a color for every condition; a palette with too few colors can't be chosen. With more conditions than any palette holds, SMTracker uses the largest, makes the extra colors as distinct as it can, and names any two that are hard to tell apart.

##### Grouped by your factors

With a [folder design](#opening-a-project) of two or more factors, two more fields appear:

- **Group by** chooses the nesting, outer factor first. "protein → medium" puts each protein's media side by side under the protein's name; "medium → protein" does the reverse. The comparison charts, legends and exports then draw the conditions in groups. The arrows move a condition within its group, and at the group's edge they move the whole group.
- **Colour by** gives every value of one factor its own color, and each condition a shade of it. *Condition* gives every condition its own color.

Neither changes a number.

#### The project menu

The project menu holds what acts on the whole project: exports, sessions, and leaving the workspace. Click the project's name at the top of the sidebar to open it.

| Item | What it does |
| --- | --- |
| **Export tracks for vbSPT…** | Writes `vbSPT<condition>.mat` for each condition, with one track per entry, x and y in nm, and each track's replicate, movie and cell. A track is split wherever it skips a frame. A note gives the time step to enter in vbSPT. |
| **Export tracks for SMMTrack…** | Writes `SMMTrack_<project>_<condition>.csv` for each condition (x and y in µm, t in s), and a `..._tracks.csv` with each track's replicate, movie and cell. |
| **Import SMMTrack dwell times…** | Reads one `<condition>.txt` per condition from a folder and draws the trapping-time histogram and its cumulative curve, with each condition's events, mean, median and SD. No test is offered, because the events carry no cell or replicate. |
| **Per-spot calibration…** | Chooses the calibration that Per spot fits use ([SQD: Populations](#sqd-populations)). |
| **Compare with the simulation's truth…** | Only for a simulated set ([below](#a-simulated-set)). |
| **Save session…**, **Open session…** | Keep the results and settings in one file ([Sessions](#sessions)). |
| **Report a problem…** | Saves one zip of the error log, the app's version and its settings, to send to the developers. It holds no tracks, images or results. |
| **Open another folder…** | Closes the workspace and goes back to choosing a folder. |
| **Back to menu** | Closes the workspace and goes back to SMTracker's first window. |

Both exports write the tracks the analysis uses: filtered, capped, and without the excluded movies, cells and tracks.

**Open another folder…** and **Back to menu** close the workspace at once, without asking, and stop any running job. Every result stays in the cache, so it returns at once when you open the folder and run the section again.

##### Sessions

A session keeps what the workspace shows, so you can draw it again later without fitting anything. **Save session…** writes one `.smtsession` file. It holds every section's results and settings, the added pages, the significance choices, the marked foci, typed axis ranges and the page on show. The Composer's panels aren't in it: save a composition from the Composer.

**Open session…** checks the session before drawing it:

- A session of another project closes this one and opens the session's folders, if they're still there.
- Its track cap and exclusions are applied first.
- If any condition's tracks differ from the saved ones, it refuses to open, and names the condition.

Then it draws every section it can, and names any it couldn't restore. A session holds only plain data, never code, and SMTracker refuses a file that holds anything else.

##### A simulated set

**Compare with the simulation's truth…** works on a data set made by SMTracker's generator: the folder above its condition folders holds `truth/truth.json`. It lists each population as the simulation declared it, beside the fit of the last SQD Run on the SQD page on show, matched slowest to slowest. Nothing is refitted. Where the Run fitted a different number of populations, a note asks you to compare them by hand.

Compare a fitted share with the *counted* share, what the simulated molecules did, rather than the declared one. A fitted D is apparent: a bound population reads the localization precision, and a fast one is shortened by the cell wall and the linking radius.

### QC

*Check every condition before you compare them: track counts, yield, whether the comparison is fair, whether the replicates agree, and what was left out.*

Check your conditions before you compare their fractions. QC lays the conditions side by side on everything that can bias a fraction, and shows each condition's tracks, cells and outliers. Run it first in every project. It runs again on its own when you apply exclusions or a track cap.

A mark in QC means "look at this", not "this is wrong". Hover over a mark for its reason, and decide whether it matters for your comparison.

#### Run QC

The workspace opens on QC. Click **Run analysis**, or the Compute mark on the right rail; after the first result, run it again with the Compute mark. QC has no settings, so it runs at once on every loaded condition. To stop it, click the Compute mark again ([Run a section](#the-workspace)).

QC reads the tracks the analyses use: after the length filter (at least 5 localizations), the *cell filter* (only tracks inside a cell outline), any track cap and any exclusions. It writes nothing to disk.

The page shows five cards, described below. Click a condition's row in **Conditions** for its [detail page](#a-conditions-detail-page). Three of the tables have a copy button, to paste them into a spreadsheet.

#### Conditions

**Conditions** gives one row per condition, so you can see their sizes side by side. Each row shows its tracks, localizations and cells with tracks (cells that hold at least one kept track). It also shows a sparkline of track lengths, the share of tracks inside a cell, the number of flagged cells, and a status.

| Status | Meaning |
| --- | --- |
| *clean* | No cell is an outlier, and no replicate has a low yield. |
| *flagged* | At least one cell is an outlier on its median apparent D (see [Outliers](#outliers)). |
| *low yield* | No cell is flagged, but a replicate's tracks per cell per 1000 frames lie more than 3.5 median absolute deviations (scaled to match an SD) below the condition's median replicate. Hover over the status to see which. This needs at least 3 replicates. |

The status reflects only outlier cells and low yield. The marks in the other cards don't change it, so read them too.

#### Is the comparison fair?

This table puts the conditions side by side on what can bias a fraction. Hover over a heading for its rule. A marked value is orange.

| Column | What it shows | Marked when |
| --- | --- | --- |
| **Tracker** | The tracker each condition was read from | The trackers differ; the conditions on the less common tracker are marked. u-track and TrackMate give slightly different fractions and D on the same movies. |
| **Replicates** | Replicates / movies | Fewer than 3 replicates, where a per-replicate interval or test is very wide. |
| **Interval** | The frame interval every D uses | Some of the condition's movies were rescaled to it from a slightly different interval (up to 10 % by default, in [Analysis options](#analysis-options)); the number in parentheses is how many. |
| **Steps** | Median steps per track | The longest median is 1.25 times the shortest or more. Long tracks are slower on average, so step-weighted fractions lean toward the slow state; the difference can also be real. |
| **Yield** | Tracks per outlined cell per 1000 frames | The highest is 1.5 times the lowest or more. Labeling density changes crowding and linking, and few tracks per cell make per-cell numbers noisy. |
| **In cell** | The share of found tracks inside a cell outline | The conditions differ by 15 points or more, because the cell filter then removes different molecules. |
| **Near edge** | The share of found tracks dropped although every point lies within 1 px of an outline | 10 % or more: tight outlines drop free molecules near the membrane and raise the slow fraction. |
| **Outlines** | The offset of the tracks from their cells' outlines, in px, in x and y | Half a pixel or more in both, in the same direction, and clearly more than chance: the tracker and the outlines count pixels from different origins, so the cell filter cuts one side of every cell. |
| **Crowded** | The median share of localizations with another inside the search radius in the same frame | 10 % or more, where links can join two molecules. |

For **Steps**, **Yield** and **In cell**, only the highest and the lowest conditions are marked. The other columns are judged per condition. The line under the title gives the pixel size and localization precision every condition uses, and says so when the pixel sizes or search radii differ.

When **Near edge** is marked, try the **Outline tolerance** option in [Analysis options](#analysis-options), which keeps tracks just outside an outline, and compare the fractions with and without it. When **Outlines** is marked, see [Check that tracks and outlines line up](#what-smtracker-writes).

#### Do the replicates differ?

This table asks, per condition, whether one replicate came out unlike the others by more than its movies differ among themselves.

QC fits each condition once, with two populations whose Ds are shared by all its movies, and gives each replicate its slow fraction: the mean over its movies. It then deals the movies out at random, 4,999 times, into groups of the replicates' sizes.

| Column | What it shows |
| --- | --- |
| **Replicates** | The replicates and the movies each holds. A movie with fewer than 50 steps is left out. |
| **Slow share** | Each replicate's slow fraction |
| **Spread** | The SD of the replicate values, in percentage points |
| **By chance** | The same SD for the random deals: median, and the 2.5 to 97.5 % range |
| **p** | The share of random deals at least as spread as the real replicates |

A **p** under 0.05, and its **Spread**, are orange: look at that condition's replicates before averaging them. With twenty conditions, about one is marked by chance. A condition with one replicate, or a replicate with one movie, can't be tested, and its row says why.

#### Cell length

This card shows every condition's outlined cells on one length axis, in µm along the centerline. A bar spans the middle half, a tick marks the median, and the whiskers run from 5 to 95 %. Hover over a row for its numbers.

A condition is marked when the longest median is 1.25 times the shortest or more. Cell length changes how often a free molecule meets the wall and how many molecules a cell holds, though the difference can be real. A merged chain of cells reads as one long cell, so check very long cells in [Edit outlines](#edit-outlines).

#### Contributes nothing

This card lists, per condition, what's in the folders but not in the numbers, so you know how much smaller the sample is than the folder suggests:

- movies with no tracks from the tracker, or without outlines;
- movies whose outlines hold no cell, or whose tracks all lie outside the outlines;
- replicates with a low track yield;
- tracks left out by the track cap, or by exclusions you made.

##### Exclusions

Click **Exclusions…** on this card to leave a movie, a cell or a track out of every analysis, each with a written reason. Nothing is deleted: remove an exclusion and click **Apply** again to put it back.

##### Add an exclusion

Choose *Movie*, *Cell* or *Track*, type its key (for example `glucose_series004:3`), and type why. Then click **Add**.

To start from QC's flags, click **Use** beside one of the up to 8 outlier movies and cells under **Suggested by QC**, from QC's last run: it fills in the kind and key. QC never excludes anything by itself. An exclusion without a reason is refused, because it couldn't be reproduced later.

##### Preview

Click **Preview with and without** to see, for each affected condition, how many tracks it keeps and how its slow fraction moves. This is a quick two-population fit; a Run measures it again with its own settings.

##### Apply

Click **Apply**. The workspace is rebuilt without the excluded data, every section's results are cleared, and QC runs again.

SMTracker remembers the last set you applied to a project. When nothing is excluded, **Restore** offers it again.

#### A condition's detail page

Click a condition's row in **Conditions** to open its detail page. The breadcrumb at the top lets you switch to another condition. Each card has an export button.

| Card | What it shows |
| --- | --- |
| **Track-length distribution** | One bar per track length, on a log scale. Tracks removed by the length filter or the track cap are gray, with a dashed line at each limit. |
| **Localizations per frame** | One bar per movie: the share of localizations with another inside the search radius in the same frame. The band is the usual range; movies outside it are orange. It's a lower bound, because only localizations that passed tracking and the cell filter are counted. |
| **N per group** | Tracks, localizations and cells per replicate and movie |
| **Per-cell outliers** | Each cell's median apparent D, sorted; flagged cells are orange. **Inspect** opens the [Outliers](#outliers) dialog. |
| **Tracks inside a cell outline, per movie** | Tracks inside and outside the outlines, per movie, with the share inside. A movie unlike its condition has an orange edge. |

##### Track cap

Click **Max length** on **Track-length distribution** to leave very long tracks out of every analysis, in every condition. Long tracks are often stuck objects, autofluorescent spots or slow molecules, so a cap lowers slow fractions a little.

Type the **Longest track kept**, in steps; there's no cap until you set one. The dialog says how many tracks and localizations it would leave out in all, and how many tracks per condition. Click **Apply**. Once a cap is set, **No cap** puts every track back. The cap lasts until you close the project (a saved session keeps it), and setting it rebuilds the workspace, like an exclusion.

##### Outliers

A cell, movie or replicate is flagged when its median apparent D lies beyond the median ± 3.5 median absolute deviations of its condition, scaled to match an SD. Outliers are flagged, never dropped.

Click **Inspect** on **Per-cell outliers** to see them three ways: *In order* (as recorded), *Sorted*, and *By movie*. Choose the **Outlier unit**: *Cell*, *Movie* or *Day* (a replicate). Nothing in this dialog changes the analysis. To leave an outlier out, use [Exclusions](#exclusions).

### Images & tracks

*See how many cells and tracks each condition holds, and look at each movie with its outlines and tracks before you fit anything.*

Look at your data before you fit it: how many cells and tracks each condition holds, and each movie with its outlines and tracks, played back over the recorded frames. Use it to spot a movie that tracked badly, a cell outlined wrongly, or a replicate with few tracks. Nothing on this page changes the analysis.

Click **Images & tracks** in the sidebar's **Data** group, beside QC. It needs no Run: it counts the cells as soon as you open it.

#### Sample size

The **Sample size** table gives, per condition, its replicates, movies, frames, cells, cells without tracks, tracks, and tracks per cell: in the median cell, then over all cells per 1000 frames and per minute. These are densities of detected tracks, not concentrations.

Choose *By replicate* or *By movie* to list the condition's replicates or movies under it. A replicate marked *low yield* has unusually few tracks per cell, by the rule in [QC](#qc). Click a row to open its first movie in the [movie view](#the-movie-view).

- Click the save button to save the table as a CSV. The file always holds every condition, replicate and movie, whatever the table shows, with a few more columns.
- Click the export button to export it as a figure, or add it to the Composer.

Cells are all outlined cells, including those with no track, and each track counts once, in its cell. The line under the table names any movies that weren't counted, and why.

#### The movie view

The view shows one movie with its cell outlines (yellow) and tracks. Click the movie button to choose another from a gallery of every movie in the condition, or step with the arrows beside it. The breadcrumb at the top switches to another condition.

| Control | What it does | Default |
| --- | --- | --- |
| **Background** | The image under the tracks: *Phase*, *Sum*, *Movie*, or one of your own channels. | *Phase* |
| **Colour by** | *Apparent D*: each track colored by its apparent D, slow blue to fast red, on one scale for the whole condition. *Per track*: every track its own color. *Single*: every track in one color. | *Apparent D* |
| **Show** | Draws the cell **Outlines** and the **Tracks** over the image. Clear either to hide it. | both selected |

Every background is offered, even where its image is missing; the image then says why.

##### Cells and tracks

Click a cell in the image, or in the list beside it, to zoom to it. The arrows in the side card step from cell to cell, and **Whole field** goes back. In a zoomed cell, click a track to highlight it: it's drawn wider, and the list shows its frames, localizations and D. Click its row in the list again, or click away from any track, to clear the highlight. While **Mark foci** is selected, a click places a focus instead.

With a track highlighted, **Play this track** plays the movie from just before it starts, and **Open in napari** opens the movie in napari at that track. Other pages open a track here the same way, for example **Show** on Apparent Diffusion.

##### Play the movie

Click play, or press `Space`, to play the tracks in order of time, on any background. Each molecule leaves a trail of its last 20 frames. Choose ¼×, 1× or 4× the acquisition's frame rate, drag the time slider, or click stop to see every track at once again.

#### The recorded movie

The *Movie* background plays the recorded frames under the tracks, each molecule's dot on its own frame. SMTracker takes the frames from the stack TrackMate was given, the frames u-track read, or the raw `.nd2` or `.stk` file over the range that was tracked, in that order.

Before it uses a source, SMTracker checks that it has the frames tracked and that the spots are brightest on their own frame. A movie off by a frame is refused, and the image gives the reason for each source it tried. With fewer than 12 spots to check (away from the movie's first and last frames and the image's edge), the movie isn't checked, and the caption says so.

On this background, **Black** and **White** replace brightness and contrast, as in Fiji:

| Control | What it does |
| --- | --- |
| **Black**, **White** | The camera counts shown black and white. |
| **Auto** | Leaves 0.35 % of the counts saturated, as Fiji's Auto does. |
| **Reset** | Spans the lowest to the highest count. |
| **Filters** › **Auto per frame (a bleaching movie stays visible)** | Gives each frame its own range, while Black and White are on Auto. |
| **Filters** › **Smooth at the spot's size (… px)** | Smooths each frame over the spot's size, as the detector sees it. |
| **Filters** › **Subtract background** | *Running median* removes each pixel's median over 21 frames: haze goes, moving spots stay. *Rolling* removes the background larger than a spot, frame by frame. *None* by default. |

Changing the smoothing or the background subtraction sets Black and White back to Auto.

#### Display and channels

On the other backgrounds, **Brightness** and **Contrast** adjust the image, and the reset button returns to the automatic stretch. **LUT** colors every background, the movie included: *Grey*, *Green*, *Magenta*, *Cyan*, *Hot* or *Inferno*. Each background keeps its own settings. They change the display only, and the exported figure says so in its legend. SMTracker forgets them when you close the workspace.

##### Your own channels

Click **+** beside **Background** to open **Fluorescence channels**. Give each channel a name, which becomes a pill, and its folder, typed or chosen with **Browse…**. A movie's image is `<movie>.tif` in that folder. A folder inside a condition's folder is looked up in every condition, as `phase/` and `sum/` are; a folder elsewhere serves every condition. You can add up to four channels, each with its own name.

An image of a different size from the movie's phase image (or its sum image, without a phase image) is refused, because the tracks would be drawn in the wrong place. SMTracker remembers the channels for this project on this computer.

#### Export

The export button at the top right exports what's shown as a figure: the zoomed cell, the outlines and tracks you show, and, while paused or playing, that frame.

#### Distances to foci

Measure how close each condition's molecules come to points you mark, such as a focus of another protein or a pole.

##### Mark the foci

Zoom to a cell, then select **Mark foci (click in the zoomed cell)** in the side card. Click each focus in the cell. To remove one, click it again, or click **Clear this cell** to remove them all.

##### Measure

Click **Distances…**. For each condition, SMTracker draws the share of localizations at each distance from their nearest focus (solid), beside chance (dashed): random points spread over the same outlines. Only conditions with marked foci are shown.

SMTracker remembers the foci for this project. In the table, the excess is each cell's median distance minus its chance median, as the mean ± SD over cells. Negative means closer to the foci than chance. With two or more replicates, **Over replicates** gives the mean and SD of the replicates.

### SQD: Populations

*Fit each condition's squared displacements as a mixture of populations, and compare their fractions and apparent D between conditions.*

Compare how much of each condition sits in each diffusing population. SQD fits each condition's squared displacements as a mixture of populations, and draws each one as a bubble: its apparent D up, its fraction as the bubble's size. It's the main analysis, so check your conditions in [QC](#qc) first.

SQD has two pages on its strip: **Populations**, this page, and [Jump distances](#sqd-jump-distances), which shows how well each number of populations fits.

#### Run it

Choose how SQD fits, and which conditions, before each Run. Click **Run analysis** (before the first result) or the **Compute** mark on the right rail to open **Configure SQD** ([Run a section](#the-workspace)). It has two tabs: **General**, for how the fit is made, and **Fits**, for which conditions are fitted with which models. Each setting has an info button (ⓘ) that explains every choice.

The dialog keeps its settings from one Run to the next; closing it with **Cancel** keeps what you changed. The first time, **Weighting**, **Uncertainty**, **ECDF**, **Optimiser** and the four-population box start from [Analysis options](#analysis-options). A condition with fewer than 20 tracks isn't fitted, and its column says so.

##### General

| Setting | What it does | Default |
| --- | --- | --- |
| **Force same D across** | Which data share one D: each condition's replicates, the conditions of a factor, the whole data set, or none ([below](#force-same-d-across)). | *Replicates* |
| **Uncertainty** | What each bubble and its spread show ([below](#uncertainty)). | *Replicates* |
| **Weighting** | *Step*: every displacement counts once, so the fraction is the share of observed time, as in SMTracker 2. *Track*: every track counts once, so it's the share of molecules. It applies to D as well. | *Step* |
| **ECDF** | How many points along the ECDF the fit is evaluated at; every displacement is always used. *Fast*: 2000 points, and the fractions change by under 0.001 percentage points. *Full*: every point. | *Fast* |
| **Overlay** | *Both* draws the fit with **Same D across replicates** beside **Same D across the whole data set**, to show what a shared D costs. The overlay is always a mean over replicates. | *Off* |
| **Time lag** | How many frames apart the displacements are: *1* to *4*. A longer lag tests whether a slow population is a state or the floor. *1–4* fits all four lags at once, as SMTracker 2's "1-4 together". | *1* |
| **Optimiser** | *Fast* uses the model's exact derivatives and is about twofold faster at 3 populations. *Standard* estimates them numerically; it gives the same fits, for checking a result against an older one. | *Fast* |
| **Localisation error** | *Not modelled*: every population has one typical step length. *Per spot*: each step gets its own localization error ([below](#per-spot-localization-error)). | *Not modelled* |

Long tracks are mostly slow, so the slow fraction reads lower with *Track* weighting. Under **ECDF**, *Custom* takes 100 to 200,000 points; below about 1000, it's no faster and less accurate.

At a longer **Time lag**, a diffusing state keeps its D, while a population of localization error alone reads 2 to 4 times lower. At *1–4*, the number of populations is chosen at lag 1, and localization error isn't modelled.

**Overlay** needs every row on one model and one number of populations, without a row's own **Force same D across**, and isn't offered under *Conditions*. **Fraction × D** isn't available while it's on.

##### Fits

**Fits** has one row per condition. Clear a row's checkbox to leave that condition out of the Run.

- **Models**: click a pill to choose which fits to make. **Best (BIC)** picks the number of populations with the lowest BIC, from 1 to 3; **1** to **4** force that number; **saSPT** uses the number of modes saSPT resolves. BIC recommends a number; it doesn't settle the biology.
- **Force same D across**: a row can have its own, *Replicates* or *Don't force anywhere*. Until you pick one, it follows the General tab.
- **Every condition**, at the top, sets every row at once.
- **Best (BIC) also tries 4 populations (about fourfold slower)**, under the rows, lets Best choose 4. Its standard is in [Analysis options](#analysis-options).

Click one model for a single fit. Click two or more, on any row, to compare fits: each condition is drawn as a group of columns, such as "1-pop" and "2-pop", in shades of its color. Where Best chose a number you also clicked, it's drawn once, as Best. A significance bracket compares one model across conditions, never two fits of one condition.

saSPT assumes one D per track, so molecules that switch states blur its spectrum, and it can undercount. Check the saSPT spectrum before trusting its count.

#### Force same D across

SQD fits the D on all the data it's forced across together, then fits each fraction with that D held.

| Choice | What it does |
| --- | --- |
| *Replicates* | One D per condition, fitted on all its replicates together; each replicate's fractions are then fitted at that D. |
| *Conditions* | With a [folder design](#opening-a-project) only; with several factors, one choice per factor. Conditions that share a value of the factor are fitted together, with one D and one number of populations. Use it when the factor is the molecule (one protein in several media), never across different proteins. |
| *Whole data set* | One D for every condition. The fractions then measure the same thing everywhere, but no condition has this D on its own. |
| *Don't force anywhere* | Every replicate fitted on its own, D included; each bubble is the mean over its replicates. A replicate with few tracks can give a noisy D. |

Under *Conditions* and *Whole data set*, the conditions share one fit, so the rows on **Fits** are grayed out: choose the models on the **Every condition** line. Best then chooses one number on the pooled data of the group, or of every condition.

Under *Replicates* or *Chunks* uncertainty, a significance bracket between two conditions fitted differently is refused, because the difference would include the method, and the line under each column says how its D was fitted.

#### Uncertainty

**Uncertainty** sets what each bubble and its spread show; the choices are explained on [The workspace](#the-workspace). On SQD:

- *Movies* and *Cells* draw the pooled fit with a 95 % interval from 200 resamples. They're slow, and aren't computed under *Conditions* or *Whole data set*.
- *Chunks* deals each condition's movies into up to **Chunks per condition** chunks (5 by default), each of at least 3 movies, or 3 cells when there are too few movies. They let Welch or the equivalence test compare conditions imaged without replicates.
- A condition with two or more replicates is drawn as its pooled fit under *Movies*, *Cells* or *Chunks*, with no spread, and the caption says why.
- With one replicate, *Replicates* draws no ±: choose *Movies* or *Cells* for an interval over that session.

#### Per spot localization error

Under *Not modelled*, a bound molecule's jumps are read through one exponential. A molecule placed less precisely then looks faster, and part of a large bound population can read as a slow-moving one. *Per spot* gives each step its own localization error, from the precision the tracker reported for its two spots, and fits by maximum likelihood. D is then corrected for localization error, but not for motion blur, and a bound population's D isn't resolved.

- u-track always reports a precision; TrackMate does when **Fit each spot's precision** was on. A condition without one, such as a track table or a simulation's true positions, is fitted without the error model in the same Run, and the caption says so.
- A Run with *Per spot* and Time lag *1–4* is refused, because each jump would be counted up to four times.
- Forcing one D across conditions with and without per-spot precision is refused.

The reported precision becomes the true error through a calibration measured on immobile spots. Choose it in the project menu under **Per-spot calibration…**:

- **Built-in**: the tracker's own curve, measured on a stationary synthetic set.
- **This project's file**: `precision_calibration.json` in the conditions' folders or the folder above.
- One from your library.

Until you choose, a project uses its own file where there is one, and otherwise the built-in curve. The choice holds for every *Per spot* Run of the project: SQD, HMM, Apparent Diffusion, the maps and Kinetics.

To make a calibration, click **Make one from a stationary measurement…** in **Per-spot calibration…**:

##### Track a stationary measurement

Image fixed cells expressing the fusion, or the fluorescent protein stuck to glass, and track it with u-track at your data's settings. This is a separate measurement, not your data. No outlines are needed.

##### Measure it

Click **Choose folder…**, choose the tracked movies, then click **Measure**. The result shows the curve, the number of steps and tracks, the share of mislinks modelled, and a table by decile. It also checks that the spots don't move: steps two frames apart must be at most 1.10 times the one-frame steps.

##### Keep it

Name it and click **Add to library**. With **Use it for this project** selected, the project uses it at once. **Also save into a project folder…** writes `precision_calibration.json` where another project can find it.

#### Reading the bubbles

Read both ends of the D axis as bounds: the slowest D sits at or below the localization floor, and the fastest near what the tracker can link. Base conclusions on the fractions.

Height is the apparent D, on a log scale; the bubble grows with the fraction, which is written on it, for example "73±3%" or "67% [63–71]". The line under each column gives its sample, such as "3 replicates · 1,271 tracks", or "3 replicates pooled" where no spread is drawn.

- Hover over a bubble for its D and fraction. A D that sits on one of the fit's bounds reads "not resolved": read it as below 0.0001 or above 10 µm²/s, not as a value.
- Click a bubble to open that condition's replicates: one column per replicate, with a table of each replicate's cells, tracks and fractions. Where you [paired replicates](#opening-a-project), a **Pair** column gives the tag the paired test reads. The breadcrumb's last crumb switches to another condition.

**Fraction × D**, the switch beside **Bubbles** on the card's header, draws every population of every condition as one point. Its fraction runs across and its D up, colored by condition, with a marker per population. Bars show only where the Run has a spread; with a D shared across replicates there's no vertical bar. Click the eye beside a condition to hide it, from the export too.

##### Check the fit

A line under the chart, in orange, names what to look at before you trust a population:

- a population under 5 % (its D is poorly determined, and a test on it is too confident), or two populations less than twofold apart in D: fit one population fewer;
- under *Conditions*, a condition whose own D is 1.5-fold or more from its group's: sharing may move a change in D into the fractions;
- a number of populations you set where Best chooses fewer, with the ΔBIC;
- a D on one of the fit's bounds, or one read low because steps past the tracker's linking radius break the track;
- a significance bracket that stands on a flagged population.

Best stops at 3 populations unless you allow 4; when it reaches that limit, the caption says so.

##### Exact-zero jumps

A simulation's true positions put a bound molecule's steps at exactly 0. These steps form their own population, with D 0 and a fraction equal to their share. The other populations are fitted to the moving jumps. (Where the positions sit on whole pixels, or fewer than 20 jumps move, the zeros stay ordinary jumps, and the note says why.) The number of populations you choose is the total, so 3 reads bound, intermediate and free, and 1 is refused. The bubble sits at a tick labeled **0**, and **Check the fit** gives the number of zero jumps. Forcing one D across such a condition and a measured one is flagged, because the measured condition's slowest population is then empty by construction.

#### The card's tools

The header of **SQD populations** has five buttons: **Axes**, export the figure, export the table (or add it to the Composer), copy the table, and show the data table.

- **Axes** sets where the D axis (and on Fraction × D, the fraction axis) starts and ends. An empty field is automatic, and **Auto** clears a typed range. A range that can't be drawn, such as a log axis from 0, is refused with the reason.
- **Copy** puts the table on the clipboard at full precision, to paste into Excel; a spread gets its own column. Under *Replicates* uncertainty, it ends with one column per replicate: the values each bubble is the mean of. A right-click on any table offers **Copy table** too.
- The data table lists every drawn population with its D, fraction and **n**, the independent units the fraction was averaged over. Its filters change the table and what **Copy** takes, not the chart.

**Map this fit**, at the right of the strip, adds a [Spatial Map](#spatial-map) page that maps this fit, and opens its settings. If you close them without a Run, the new page is removed.

After a Run with replicates, **Metadata** on the right rail lists the fit's settings and each replicate's fractions, slowest first. Significance brackets and exports are described in [Figures](#figures); clear **n** in the export dialog to leave out the sample line.

### SQD: Jump distances

*See each condition's jump distances against the 1-, 2- and 3-population fits, and check how many populations the data support.*

See how well 1, 2 or 3 populations describe each condition's jumps, before you trust the fractions on [SQD: Populations](#sqd-populations). Each condition gets its jump-distance histogram with the fits drawn over it, and two residual plots. A table scores every fit on the data, and on replicates or cells held out of the fit.

Open it from the strip at the top of **SQD**. It doesn't need a Populations Run first.

#### Run it

Choose which conditions to fit and how strictly to test them. Click **Run analysis** (before the first result) or the **Compute** mark on the right rail to open **Configure jump distances** ([Run a section](#the-workspace)):

| Setting | What it does | Default |
| --- | --- | --- |
| **Conditions** | The conditions to fit; clear one to leave it out of this Run. | *All* |
| **Fit up to** | *3 populations* fits 1, 2 and 3 populations, as **Best (BIC)** does on Populations by default. *4 populations* adds a fourth, and takes about four times as long. | *3 populations* |
| **Held out** | What the held-out test leaves out. *Replicates*: each replicate in turn, the stricter test. *Cells*: up to five groups of the condition's cells, for single-session data; it's the gentler test, because the cells share their replicates with the cells fitted. | *Replicates* |

Click **Run**.

The fits take three settings from **Configure SQD** on Populations: its **Time lag**, its **Optimiser** and its **Localisation error**. Its **Weighting** doesn't apply: the curves here are always step-weighted. Under *Per spot*, every fit on this page uses each jump's own localization error, except for a condition whose tracks carry no per-spot precision. When the Time lag is *1–4*, the histogram is drawn at lag 1.

A Run comes in two passes. The histograms, curves and residuals appear first; the ΔBIC and held-out columns read "computing…" until the second pass fills them in.

#### A condition's card

Each condition has a card, one per row. Its caption gives the number of jumps and tracks, the time lag τ, and which fit the curves are.

| Control | What it does | Default |
| --- | --- | --- |
| **Show** | One pill per fit. Click a pill to draw or remove that fit; the last one on stays on. | 2 at first, then the number BIC prefers |
| **Components** | Draws each fit's populations (a 1-population fit has none): in shades of the condition's color for one fit, as thin dashed lines in each fit's color for several. | selected |
| **Y axis** | *Probability*: the bars sum to 1, as in SMTracker 2 and Spot-On. *Counts*: jumps per bin. *Density*: per nm, total area 1. The curves follow; only the scale changes. | *Probability* |

One fit is drawn alone in the text color. Several fits are each drawn in their own color and line style: 1 population dotted, 2 solid and 3 dashed, as in SMTracker 2, and 4 dash-dot. A pill set by hand is kept when the second pass arrives.

The bars are 10 nm wide, up to the 99.5th percentile of the jumps. The bins only draw the jumps; every fit uses the jumps themselves.

Beside the histogram, two residual plots show the same fits:

- **Pearson residual**: (observed − expected) / √expected per bin, in counts whatever the y axis. Counting noise alone mostly stays within ±2. With one fit, bars beyond ±2 are orange.
- **ECDF residual** (the **Model − ECDF** plot): the fit's cumulative distribution minus the jumps' own, unbinned.

Drag the histogram's key to move it, and double-click it to put it back; a card's export keeps it where you put it. **Axes** on the card sets the x and y ranges; an empty field is automatic, and a new y unit sets y back to automatic.

**Same y scale**, above the cards, puts every histogram of one unit on the largest range among them. Click **Own y scales** to go back. Use *Probability* or *Density* for this, because counts of different numbers of jumps don't compare.

##### The table

Under each card, a table scores every fit two ways: on all the jumps, and on the jumps it was fitted without.

| Column | What it shows |
| --- | --- |
| **Populations** | The number of populations fitted. |
| **D (µm²/s)** | Each population's D. |
| **Fractions** | Each population's share of the jumps. |
| **ΔBIC** | The fit's BIC minus the lowest, from maximum-likelihood fits. 0 is the fit BIC prefers. |
| **Largest ECDF gap** | The largest distance between the fit and the jumps' distribution, in percentage points. A description, with no p-value. |
| **Held-out ΔlogL (replicates)** or **(cells)** | How well the fits made without each replicate (or group of cells) predict the jumps left out, summed over the folds, minus the best. 0 is the best. |
| **Folds won** | In how many folds that fit predicted best. |

The line under the table sums it up: which number BIC prefers, "inconclusive" when the runner-up is less than 2 behind, and which number the held-out test prefers. BIC and the held-out test can disagree; read them side by side.

The held-out test needs at least 2 replicates, or 4 cells. With one replicate, the line says "held out: 1 replicate": choose *Cells* to test a single session. When some jumps are exactly zero and form an immobile population, the 1-population fit isn't made, and the table starts at 2.

#### All conditions

The **Each condition** / **All conditions** switch at the top right shows every condition's histogram on one screen, on one jump-distance axis.

- **Show for every condition** and its **Components** set every card at once. A card without that fit keeps its own, and no pill is lit while the cards differ.
- **Residuals**: select **Pearson** or **ECDF** (both cleared to start) to draw that plot under every histogram.
- **Plots**: *One plot per condition*, or *One plot per fit*, which gives each fit its own plot, alone with its components. With one plot per fit, *Conditions down* puts each condition in a row, and *Conditions across* swaps rows and columns.
- **Columns**, with one plot per condition, sets how many plots stand across. *Auto* places up to 3.
- **Size** (*S*, *M*, *L*) sets the size of each plot. Double-click a plot to span the page, and again to restore it.
- **Y axis** sets the unit of every plot and every card. **Same y scale** also puts each kind of residual plot on one range. **Axes…** sets the ranges of every plot here; a plot's own **Axes** wins. A card's typed ranges don't carry over.

#### Export

The export button on a card exports its figure; **Export all…** exports every condition as one figure, laid out as shown. Both open the export dialog ([Figures](#figures)) with these rows of their own:

| Row | What it does |
| --- | --- |
| **Fit** | The fits drawn, starting as the card shows them. On Export all, *Cards* draws each condition as its card. |
| **Show** | Adds **Components**, **Pearson residuals** and **ECDF residuals** to the usual boxes. A card's figure starts with both residuals; Export all starts on the page's **Residuals**. A plot left out also leaves the source data. |
| **Y unit** | Starts on the card's unit; Export all on the cards' unit where they agree, otherwise *Probability*. |
| **Grid** | Export all, one plot per condition and two or more conditions only: rows × columns, starting on the page's **Columns**. |

The figure follows the page's **Same y scale**. A card's figure starts on its typed ranges; Export all offers only an x range, starting on **Axes…**. A card's source data holds every bin and a *Fit quality* sheet with every fit made; Export all's holds a *Fits* sheet of each condition's fits. A figure exported before the second pass ends says its BIC and held-out columns are empty. **Add to composer** keeps these choices in the panel.

### MSD

*Plot each condition's mean squared displacement against lag time, fit the early lags, and see why the slope moves with the fit window and track length.*

Use MSD as a cross-check on [SQD](#sqd-populations): it plots each condition's mean squared displacement against lag time and fits a line through the early lags. The slope gives one D through every population at once, dominated by the fastest, so it's never a population's D. The populations and their fractions come from SQD.

MSD has two pages on its strip: **Curves** and **Checks**. A Run fits every loaded condition.

#### Run it

Choose the fit window, the weighting and the direction before each Run. Click **Run analysis** (before the first result) or the **Compute** mark on the right rail to open **Configure MSD**:

| Setting | What it does | Default |
| --- | --- | --- |
| **Lags fitted** | How many early lags the line is fitted to: *3* to *6*. Later lags rest on fewer, slower tracks and the cell wall bends the curve, so a longer window gives a lower D. | *4* |
| **Curve drawn** | How far the curve is drawn, *to lag 10* or *to lag 20*, so you can see it bend at the cell wall. It doesn't change D. | *to lag 10* |
| **Weighting** | *By pairs*: each track counts by its number of pairs at that lag, as in SMTracker 2, so the mix shifts toward long, slow tracks. *Each track once*: every track counts the same. | *By pairs* |
| **Direction** | *Camera plane*: x and y of the image, D = slope / 4. *Along the cell*, *Across the cell*: each step projected onto its cell's centerline, D = slope / 2. Across a rod about a micron wide, the curve flattens within a few lags. | *Camera plane* |
| **Uncertainty** | *Replicates*, *Movies*, *Cells* or *None* ([Uncertainty](#the-workspace)). *Movies* and *Cells* give a 95 % interval from 200 resamples. | *Replicates* |

Click **Run**. MSD's settings don't come from [Analysis options](#analysis-options), though the cell filter and frame interval set there apply here too.

*Along the cell* and *Across the cell* need outlines with a centerline. Tracks in a cell without one are left out of those directions, so **Tracks** counts fewer. (SMTracker 2 divided the one-axis slope by 4, which halved D.)

#### Curves

**Ensemble MSD, every condition** draws every condition's curve on one chart. Filled points were fitted, hollow ones only drawn, and the fit line runs on, dashed, past the fitted lags. Its caption repeats the direction, lags and weighting.

- Click a condition's eye in the key to hide it, from the chart and its export.
- Click a condition's name, or near its curve, to see its replicates: each replicate's curve and fit, beside the pooled one. Click in the chart again to go back.
- Hover a lag to read each condition's MSD there.

The bars show the spread you chose: the SD of the replicates' MSD at each lag, or the 95 % interval of the resampled curves. With *None*, one replicate, or a withheld interval, this chart draws no bars. A later lag that too few replicates or resamples reach has no bar either.

##### The numbers

The table gives each condition's **Tracks**, **D (µm²/s)**, **Spread** and **Intercept (µm²)**. D is always the pooled fit; **Uncertainty** changes only the spread written beside it:

| Spread | Meaning |
| --- | --- |
| ± SD (n replicates) | The SD of the replicates' own D, each fitted alone. |
| [low – high] (n movies or cells) | The 95 % interval from resampling. |
| *withheld* | No movie or cell interval: the condition holds two or more replicates, or too few movies or cells to resample. |
| *one replicate*, *too few replicates* | Not enough replicates to give a spread. |
| – | *None*. |

The intercept isn't σ. For a free molecule imaged continuously, it's 2 · d · (σ² − D · τ / 3), where d is the number of dimensions: 2 in the camera plane, 1 along or across the cell. So motion blur lowers it, and the cell wall raises it.

Click **Save MSD per cell (CSV)…** to save every cell's MSD, one row per cell and lag up to **Curve drawn**, in the Run's weighting and direction. It's SMTracker 2's "Save MSD in csv" for all cells at once. Tracks outside every outline are left out. The columns are `condition`, `replicate`, `movie`, `cell`, `delay_s`, `mean_msd_um2`, `std_msd_um2` (the spread of the cell's tracks), `n_tracks` and `n_pairs`.

##### Single tracks

To see the spread behind each curve, select **Single tracks**: one card per condition, with up to 120 of its tracks (of at least 3 lags) behind the ensemble curve. Where the shared chart has no bars, these cards draw the SEM over tracks instead, which is only indicative, because tracks of one replicate aren't independent. Hover a point for how many tracks reach that lag: later lags rest on fewer, longer tracks. With two or more replicates, click the chart to swap the tracks for the replicates' curves.

#### Checks

**Checks** shows why D moves with the fit window and the track length, one card per condition. It's computed to lag 8, in the Run's direction.

- **Fit window**: D from the first 2, 3, 4, 5, 6 and 8 lags, by pairs (solid) and with each track once (dashed); a gray line marks the Run's **Lags fitted**. A D that falls as the window grows is expected: pair weighting shifts the mix toward long, slow tracks, and the wall bends every curve.
- **Track length**: D from each track-length class on its own (4–5, 6–9, 10–19, 20–49 and 50+ steps), in the Run's weighting. A class with fewer than 10 tracks is drawn faded and not quoted.

The sentence under each chart sums it up: the change in D across the windows, and the D of the shortest and longest tracks.

##### Beside SMTracker 2

**What v2 printed, and why it differs** compares this page's D with what SMTracker 2's MSD tab gave for the same data. SMTracker 2 fitted only tracks of at least 9 localizations, each cut to lags 1 to 8, all 8 lags fitted unweighted. Its localization-error panel fitted a line per track over lags 1 to 7 and averaged √|b/4|, where b is the line's intercept. That folds negative intercepts onto positive values.

Each condition's card has a table with this page's D, SMTracker 2's ensemble fit, its per-track σ, the share of negative intercepts and, where it has one, the project's σ. A histogram shows the per-track σ with the sign of the intercept kept, so you can see what the fold hid.

#### Export

Every card has an export button, and the shared chart and the Single tracks cards have **Axes**: lag time (x, s) and MSD (y, µm²). A typed range stays through a new Run until you click **Auto**. See [Figures](#figures).

### Apparent Diffusion

*Give each track one apparent D from all its steps, fit the populations they form, and see how they spread over cells, switching and track length.*

Fit the populations from whole tracks, as a second view beside the single jumps of [SQD: Populations](#sqd-populations). Each track gives one apparent D: the mean of its single-frame squared jumps divided by 4τ. Steps across a frame gap aren't used. The populations are a gamma mixture fitted with each track's own number of steps, so every step is used, and each molecule counts once, however long its track.

Apparent Diffusion has four pages on its strip: **Populations**, **Distributions**, **Per cell** and **Checks**. A Run fits every loaded condition.

#### Run it

Choose how the populations are fitted, and which models, before each Run. Click **Run analysis** (before the first result) or the **Compute** mark on the right rail to open **Configure Apparent Diffusion** ([Run a section](#the-workspace)). Apparent Diffusion takes nothing from [Analysis options](#analysis-options).

##### General

| Setting | What it does | Default |
| --- | --- | --- |
| **Force same D across** | Which data share one D, as on [SQD: Populations](#sqd-populations). Under *Don't force anywhere*, D has a spread too. | *Replicates* |
| **Uncertainty** | *Replicates*, *Movies*, *Cells*, *Chunks* or *None* ([Uncertainty](#the-workspace)). Under *Chunks*, a field sets how many. | *Replicates* |
| **Shortest track** | *5 points*, *10 points* or *20 points*: 4, 9 or 19 single-frame steps. A longer minimum makes each value more precise, but keeps fewer tracks, and those lean toward slow molecules (**Checks** shows how much). 5 points is u-track's minimum. | *5 points* |
| **Best (BIC) up to** | The most populations Best may choose: *3 populations*, as in SQD's Best, or *4 populations*. A 4 chosen on **Fits** is always fitted. | *3 populations* |
| **Localisation error** | *Not modelled*: a bound molecule placed less precisely reads faster, so part of the bound population can be fitted as a middle one, or as two of similar D. *Per spot*: each track gets its own localization error, and a bound population's D reads near 0. | *Not modelled* |

*Per spot* uses the tracker's per-spot precision and the [calibration](#sqd-populations) SQD uses. Forcing one D across conditions is the stronger assumption: it can enlarge the difference between conditions, even though the fit gets worse. Under *Conditions*, a condition alone in its group keeps its own D.

Under *Per spot*, a condition whose tracks carry no per-spot precision, such as a track table, is fitted without it, and the caption says so. If no condition carries it, the Run is refused, and the status line says why. Forcing one D across conditions with and without it is refused too.

Tracks with an apparent D of exactly 0, as in a simulation's true positions, form their own immobile population at D 0, so one population is refused.

##### Fits

**Fits** has one row per condition, with the models to fit: **Best (BIC)**, **1**, **2**, **3** and **4**, several at once. With two or more, each condition is drawn as a group of columns, one per fit. **Every condition** sets every row. A row can also set its own **Force same D across**, *Replicates* or *Don't force anywhere*. Under *Conditions* or *Whole data set*, the rows are grayed out, because the conditions share one fit and its models.

The other pages read one fit per condition: Best's when **Best (BIC)** is chosen, otherwise the smallest number of populations of 2 or more that's chosen (1 only when it's the only one).

Each condition's **Distributions** and **Checks** cards fill in as it's fitted. The **Populations** bubbles and **Per cell** appear once every condition is done.

#### Populations

The **Apparent-D populations** chart works as on [SQD: Populations](#sqd-populations), with its table, copy and export buttons, significance brackets and **Axes**. A warning line above the caption appears when a bracket stands on a population the fit checks flag.

Under *Not modelled*, when a condition's two slowest populations have similar D (under twofold apart), the caption says they may be one bound population read as two. Run with *Per spot*, or fit one population fewer. When you choose more populations than Best would, the caption says by how much.

Click a bubble to see that condition's replicates (or chunks): a bubble per replicate, each with its own fractions, and a table ending in their mean ± SD. Click **← All conditions** to go back.

**Map this fit**, at the right of the strip, adds a [Spatial Map](#spatial-map) page that maps this fit, and opens its settings. If you close them without a Run, the new page is removed.

##### Cross-checks

On a condition's replicate page, **Cross-checks (SQD, HMM, saSPT)…** opens two tabs for that condition. Each computes only when you click its **Run analysis**, and takes a while:

- **SuperPlots**: an SQD fit with its cell bootstrap, and an HMM fit.
- **Dynamics**: an HMM fit with bootstrap, a saSPT spectrum and a clustering null.

Use the breadcrumb to go back.

#### Distributions

Each condition's card shows the histogram of its tracks' apparent D on a log axis, with the fits drawn over it. Its caption gives the tracks and the number of populations, and a line above the chart gives the fit's D and fractions.

| Control | What it does | Default |
| --- | --- | --- |
| **Show** | One pill per fit. Click a pill to draw or remove that fit; the last one on stays on. | The fit the other pages read |
| **Components** | Draws each fit's populations: in shades of the condition's color for one fit, as thin dashed lines in each fit's color for several. | selected |
| **Residuals** | Adds the Pearson residual per bin and the ECDF residual (the fit's cumulative distribution minus the tracks', in percentage points) under the histogram. | selected |

Three dashed lines mark the axis:

- **cut**: where the populations cross, for a track of median length. Drawn only while **Show** holds just the fit the other pages read.
- **immobile**: what localization error alone adds to an apparent D.
- **ceiling**: the fastest apparent D the tracker's search radius lets through, where the condition recorded one.

The curves are the pooled fit of every track. Where the bubbles on **Populations** come from replicates or chunks, their D differs slightly, and the card names both.

The fit-quality table reads as on [SQD: Jump distances](#sqd-jump-distances): each fit's D and fractions, **ΔBIC**, **ECDF gap**, **Held-out ΔlogL** and **Folds won**. The held-out test leaves out each replicate in turn. With one replicate, it splits the cells into up to five folds instead (at least 4 cells are needed), which is the weaker test. Hover over the table for its notes. A line under it counts the tracks too short for **Shortest track**.

**Pick tracks**: drag across the histogram, or click a bar, to list up to 12 tracks in that range, longest first, with their D, steps, movie, cell, replicate and slow share. Click **Show** on one to open it in [Images & tracks](#images-tracks), playing from just before it starts.

**All conditions**, at the top, puts every condition's histogram on one page, drawn as its card shows it. **Export all conditions…** exports them as one figure.

#### Per cell

**Per cell · each cell's slow share** shows whether the slow population is present in every cell. Each dot is one cell's slow share: the mean chance, over its tracks, that a track is in the slowest population. Dots are sized by their tracks and shaded by their replicate. An open circle is each replicate's mean over its cells, and the line is the mean ± SD of the replicates. Cells with fewer than 3 tracks are left out, and a condition fitted with one population has no slow share. Hover a dot to name its cell.

**Slow share against cell length** plots each cell's slow share against its length along the centerline, with each condition's mean per length class. It needs outlines with a centerline.

- **Slow share from**: *Apparent D*, or *SQD*: each jump's chance of being slow under the condition's pooled, step-weighted SQD fit with the same number of populations, so the share of time spent slow.
- The eye beside a condition hides it from the chart, the table and the export.

The table gives each condition's cells, median length, slow share, length-matched slow share and slope in points per µm. Length-matched weights each condition's class means by every condition's cells together, as if all had the same lengths. The classes are the quartiles of all cells, and only classes where every condition has at least 2 cells count. The line under it compares the spread between conditions as measured and as matched. It's descriptive, with no test, because cells of one replicate aren't independent.

#### Checks

Two checks show whether the fractions can be read at face value. Each condition gets one card with both:

- **Switching**: the observed apparent D beside what SQD's populations predict if no molecule switched state within its track. Switching puts tracks in the shaded band between the states; the line under it gives the observed share there and the predicted one.
- **Track length**: the slow share by how many steps a track has (4–5, 6–9, 10–19, 20–49 and 50+). A class with fewer than 10 tracks is drawn faded and not quoted. When long tracks are clearly slower, the line says so: a track-length cut then moves the fractions.

#### Export

Every card has an export button, and the source data hold every bin and fit. The Populations card and each Distributions card have **Axes**; a typed range stays through a new Run until **Auto**, and the export draws it. See [Figures](#figures).

### HMM

*Fit a hidden Markov model of diffusive states: how often molecules switch between slow and fast, how much time they spend in each, and how they turn.*

Use the HMM when the fractions alone can't tell two conditions apart. The same bound fraction can come from short, frequent binding or from long, rare binding; the switching rates show which. Unlike SQD, the model has memory: a molecule's state at one step depends on its state at the step before.

The HMM fits a number of diffusive states, each with its apparent D, and the rates at which molecules switch between them: per second, measured from step to step. Each track is split wherever it skips a frame, and pieces of fewer than 2 steps are left out.

HMM has four pages on its strip: **State diagrams**, **Slow state**, **Jump angles** and **Table**. **Map this fit**, at the right of the strip, adds a [Spatial Map](#spatial-map) page that maps this fit, and opens its settings; if you close them without a Run, the new page is removed.

#### Run it

Choose the number of states and how D is shared before each Run. Click **Run analysis** (before the first result) or the **Compute** mark on the right rail to open **Configure HMM** ([Run a section](#the-workspace)). The HMM takes none of the SQD fit's [Analysis options](#analysis-options).

##### General

| Setting | What it does | Default |
| --- | --- | --- |
| **Force same D across** | Which data share one set of D, as on [SQD: Populations](#sqd-populations). Under *Replicates*, each replicate's rates and occupancy are fitted at the condition's pooled D. Shared across conditions, the rates compare the same states. | *Replicates* |
| **Uncertainty** | *Replicates*, *Movies*, *Cells*, *Chunks* or *None* ([Uncertainty](#the-workspace)). *Movies* and *Cells* give percentile intervals from 100 refitted resamples, for a single replicate only. Under *Chunks*, a field sets how many (2 to 20, 5 by default). | *Replicates* |
| **σ correction** | *Off (apparent D)*: D compares with SQD. *Tracker's sigma*: subtracts the tracker's localization σ, as a check; if the slow state disappears with it, it was localization noise. *Per spot*: each step gets its own localization error, so a bound molecule's faint, blurred spots no longer read as short visits to a faster state. | *Off (apparent D)* |
| **Best (BIC) up to** | The most states Best may choose: *3 states* or *4 states*. | *3 states* |

Forcing one set of D across conditions is the stronger assumption: if every D of one condition is lower, the shared D moves that shift into the occupancy and rates.

*Per spot* uses the tracker's per-spot precision and the calibration of [SQD: Populations](#sqd-populations). A condition whose tracks carry none is fitted without it, and its D stays apparent. If no condition carries it, the Run is refused, and the status line says why. A corrected D is labeled σ-corrected or per-spot, never apparent.

Under *Tracker's sigma*, when the project records no σ, a note under the setting names the theoretical value it would use, or says there's none.

##### Fits

**Fits** has one row per condition, with its number of states: **2 states** (the default), **3 states**, **4 states** or **Best (BIC)**. Two states is the standard model. Three lets BIC test whether a third state is supported; its slowest D often sits at the localization floor. Best fits from one state up to its limit and picks by BIC; a difference under 2 is inconclusive.

Clear a row's checkbox to leave that condition out. A row can set its own **Force same D across**, *Replicates* or *Don't force anywhere*. **Every condition** sets every row. Under *Conditions* or *Whole data set*, the conditions share one fit and its number of states, so each row's number of states and Force same D across are grayed out; set the number on **Every condition**. The shared states are fitted first, and the status line counts the fits.

Each condition's card fills in as it's fitted.

#### State diagrams

Each condition's card shows its states and how molecules move between them. Its caption gives the number of states, tracks, steps, the frame interval τ, the σ correction, and how D was forced.

- **The diagram**: one circle per state, its area the occupancy, labeled with its name, D and occupancy. Arrows run both ways between every pair of states. With two states, each arrow carries its rate. With three, the arrows are wider for a faster rate, and the rates table carries the numbers. Dashed arrows mean the rates aren't identifiable: a state was never left, or some resamples held no switch.
- **Switching rates (per second)**: each switch, such as "slow → fast", with its rate. Under *Movies* or *Cells*, the 95 % interval follows; under *Replicates*, the replicates' mean ± SD.
- **The states table**: each state's D, its **Occupancy**, its share at a track's start (**Track start π**), its long-run share the rates imply (**Stationary**), and its **Dwell (model)**.

Occupancy is the time average over the observed steps. Short tracks don't mix, so it lies between the share at the start of a track and the long-run share. The dwell is the inverse of the rate out. It's usually longer than the tracks themselves, so treat it as a model figure, not a measured time. A state that's never left has no dwell.

Under the tables, lines flag what to check:

- a state holding under 5 % of the steps, or two states less than twofold apart in D: fit one state fewer;
- a number of states you set where Best chooses fewer, with the ΔBIC;
- no transition out of a state, so its rates and dwell aren't measured: quote the occupancy instead;
- under *Movies* or *Cells*, rates not identifiable, when resamples held no switch;
- a replicate that couldn't be fitted, and, under Best, a number of states that's inconclusive;
- states left within two frames: molecules caught mid-switch, which may not be a population.

Each card has an export button. **Export all diagrams…** exports several conditions in one figure, with one key: select the conditions beside their names, and set **Grid** to rows × columns. In the export dialog, **± on rates** and **± on occupancy** add the spread where one was computed; the intervals themselves are always in the source data.

#### Slow state

**Across conditions · the slow state** compares every condition's slow state in panels: **Slow-state occupancy**, **Rate into the slow state**, **Rate out of the slow state** and **Slow-state D (apparent)**, which says σ-corrected or per-spot when every condition was corrected. Hover over **How to read these numbers**, at the top, for how to read each panel.

Each replicate is an open circle, their mean is filled, and the bar is ±1 SD between replicates: a spread, not a confidence interval. Under *Movies* or *Cells*, the bar is the 95 % interval instead. Unless **Force same D across** is *Don't force anywhere*, every replicate has the same D, so the D panel shows no spread.

With three states, the rate out is the sum of the slow state's exits, and the rate in is the rate at which a molecule outside it enters it.

- **Dwell time** adds a panel with the slow state's model dwell, in ms.
- Click a condition's eye in the key to hide it from the panels and the export.
- **Join paired replicates** draws a thin line through each pair's circles, where you [paired replicates](#opening-a-project) and two drawn conditions share a tag. A line is for reading, not testing; the export dialog's **Pair lines** box prints them.
- **Axes** has a row per panel, since each has its own unit. A typed range stays through a new Run until **Auto**.

**Significance**, on the right rail, puts brackets on any panel, or on all of them, from the replicates' own fits ([Figures](#figures)). Run with *Replicates* uncertainty for that. The test is **Welch on replicates**, or **Paired t (declared pairs)**. A bracket is refused, with the reason, for a pooled fit, for random chunks, between conditions with different numbers of states, and on the D panel where D was held the same across replicates.

#### Jump angles

**Jump angles** asks whether a molecule in each state turns back more often than it goes forward. It reads the HMM's fits, so run the HMM first. The angles are computed when you open the page.

Each pair of consecutive steps gives an angle, 0° straight on and 180° turned back, and counts for a state by the chance that both steps were in it. Pairs across a switch are left out, and the caption gives their share. Only tracks in a cell with a centerline are used.

**Turning back** is the weight within 30° of 180° divided by the weight within 30° of 0°: 1 means no preference.

- Each condition's card draws each state's angles as a half circle, forward on the left and back on the right, with a line giving its turning back over the replicates.
- **Across conditions · turning back** draws each state's value for every condition on one strip, with a table per state comparing the conditions: Welch on replicates, Holm-corrected. A replicate needs about 20 pairs' worth of weight for its own value.

A free molecule also turns back near the wall, and where its steps are as small as the localization error. Select **Against chance** to compare with free molecules simulated at the same places: the same cells, the state's step size, each spot's localization error and the camera's blur. The strip then shows turning back beyond chance; an excess under ±0.05 isn't read as a difference. It takes longer to compute.

#### Table

**Table** compares one state of every condition side by side. Choose *Slow*, *Middle* or *Fast* under **State**. The table gives its D, its dwell, its share at the start, on average and in the long run, and its rates in and out. A condition fitted without that state shows "–".

### Spatial Map

*Where in the cell each kind of motion happens.*

The Spatial Map shows where in the cell each kind of motion happens. **Compute** opens the settings, one page per setting. Choose the method first.

| Method | What it draws |
|---|---|
| **SQD** (default) | The share of each SQD population at each position along and across an averaged cell. |
| **Apparent Diffusion** | The same for apparent-D populations. |
| **HMM (Kinetics)** | The same for HMM states. |
| **Confinement (SLA)** | Free, mixed and dwelling tracks, with a chance rate for each from matched free diffusion (in the **Metadata** receipt and the exported legend). Set the confinement radius and the **Minimum dwell** (default 5 steps) on their own pages. |
| **SDA** | Maps of localizations, speed, apparent D and acceleration. **Adaptive smoothing** (on) widens the smoothing only in bins with too few steps. |

#### Populations page (SQD, Apparent Diffusion, HMM)

| Setting | What it does | Default |
|---|---|---|
| **Populations** | Number of maps: SQD 1, 2 or 3; Apparent Diffusion 2 or 3; HMM 1, 2 or 3 states. One SQD population draws a single map, **All jumps**. | 2 |
| **Use the same D as the SQD fit** (or the apparent-D or HMM fit) | Each condition's map takes that section's last Run: its D values, number of populations and fit settings. The map fits only the shares at those D values. The counts below and the SQD **Time lag** are grayed while it is selected. Clear it to fit each condition on its own. | Selected |
| **Localisation error** | Shown while the box is cleared and the split is Fitted. **Not modelled**, or **Per spot**, which gives each jump (or four-step piece) its own error through the calibration SQD's Per spot uses. Not available with SQD lags 1-4 together. | Not modelled |
| **Every condition** and one row per condition | Each condition's count: 1, 2, 3 or **Best (BIC)**. The Every condition row sets them all. | |
| **Time lag** (SQD) | Jumps 1 to 4 frames apart; **1-4** fits all four together. | 1 |

If the section is still running, the map waits for it; if it has never run, **Run** starts it. If you run the section again with other settings, the map says "Made with an earlier SQD fit" with a **Run** beside it. The caption and **Metadata** receipt say which D was used.

The grid has a column for every map some condition draws. Every SQD row also draws **All jumps** (every jump once), so that column compares conditions on one footing; Apparent Diffusion draws **All tracks**. A condition at one population shows "One population: no split". A condition at **Best (BIC)** shows "waiting…" until its count is chosen. If your count is higher than Best (BIC) would choose, your count is kept and a warning under the grid says so. Any column can be hidden in **Visibility**.

At **Hard** assignment, an HMM state that no step was most likely in gives an empty map; use Soft. A condition whose tracks never switch state is still mapped, with each track in one state, and the map says so.

#### Split page

| Setting | What it does | Default |
|---|---|---|
| **Fitted** | Cuts each condition where its fitted populations cross. | Default |
| **Manual** | Up to four ranges that you set (jump length in nm for SQD, mean apparent D in µm²/s for Apparent Diffusion), one map per range, the same for every condition. The page shows the chosen condition's histogram and the share in each range. **From the fit** starts the ranges where the fitted maps would cut. Ranges may leave gaps but not overlap; data in no range is not drawn. Needs one time lag. | |

#### Other settings

| Setting | What it does | Default |
|---|---|---|
| **Jump drawn at** | Where a jump is placed: where it began, halfway, or where it landed (SQD, HMM, SDA). Counted at the start, long jumps read as peripheral; halfway pulls them toward the axis. | Start |
| **Assignment** | **Soft** shares each jump among populations; **Hard** gives it to the most likely one. | Soft |
| **Fold** | **On** sums the cell's four quarters, so the map is symmetric. **Off** draws the data as it fell. | Off |
| **Cells** | All, Short, Mid or Long (thirds by cell length). | All |
| **Minimum jumps per bin** | Bins with fewer are left blank; 0 turns it off. | 0 |
| **Bin size** | **1 px** is one camera pixel, read from your data's pixel size and shown on the pill. **In nm** (or the slider, 30 to 400 nm) sets a bin in nm. If no pixel size is recorded, Run asks for nm. | 1 px |
| **Cell frame** | The averaged cell the maps are drawn in. **Measured** is the median outline of this project's own cells; **Deposit** is a fixed frame from an earlier data set. | Measured |
| **Long axis** | **Normalised** stretches every cell to the drawn length, as SMTracker 2 did, so a pole bin is a fraction of each cell. **Pole-anchored** keeps each pole's own size and stretches only the middle, so a pole bin is the same distance from the tip in every cell. | Normalized |
| **Weighting** | **By steps** counts every jump; **By track** counts every track once. Under "Use the same D as the SQD fit" the shares follow that fit's weighting. | By steps |
| **Maximum localisations per track** | **Tracks** leaves out tracks longer than the limit; **Cells + Tracks** leaves out the whole cells that hold one. The line under the setting says how many tracks or cells and what share of the jumps that removes. | Off |
| **Effective sample size** | Leaves blank (gray) the bins worth fewer than the set number of independent tracks or cells. The cutoff is provisional: it has not been checked against a simulation. | Off |

#### Reading the result

- Click a map to open that condition's replicates; click a spot on a replicate's map to list the tracks and cells behind it. The **eye** beside a listed track hides it from the map only, for this session.
- **Where the jumps sit** (under the maps) gives, for each population, every condition's profile along the cell (distance from the nearer pole, nm) and across it (-1 to 1), against an evenly spread population (dashed), with a table of mean |d_norm|, mean distance from the pole and share in the caps. The **eye** beside a condition in the key hides it everywhere.
- Both cards have **Export**, with every bin in the source data.

**Colour scale** (in **Visibility**, on the right rail):

| Choice | Bars |
|---|---|
| **Population** (default) | One bar under each column. |
| **Condition** | One bar to the right of each condition's maps. |
| **Map** | One bar under every map, each on its own scale. |
| **All** | One scale for every map. |

### Kinetics

*Populations over time after a trigger.*

Kinetics is for an experiment with a trigger at a known moment, such as a medium switch or a drug addition. For **Populations over time**, the Run dialog first asks how the compound was added.

| Design | Meaning | Independent unit |
|---|---|---|
| **Added under the microscope** | The compound reaches the cells under the objective; the same cells are imaged before and after. | The replicate (chamber); its movies are time points of one response. |
| **Added to the culture** (default) | You add it to the culture, incubate, mount, then film several movies of that culture. | The culture (replicate); its movies are time points, never replicates. |
| **Added to each sample** | Each sample (slide, tube or movie) is induced at its own time. | The culture the samples came from. |

Under **Added to each sample**, every movie is its own sample unless you give movies of one sample the same name. Type each sample's addition time or press **Paste from Excel...** (lines of `movie<TAB>time`, or a column of times in the order shown). Every movie needs a time; the Run names any that lack one.

#### Clock

| Setting | What it does |
|---|---|
| **Trigger frame in each movie** | The addition happened during the recording. Type the frame per condition (frame 1 is the first tracked frame). |
| **Time after the addition** (Populations over time) | Type the clock time of the addition once per replicate (`10:30`, or `2026-10-23 10:30` if on another date), or once for all in **Same time for every replicate**. Each movie's time is read from its camera file; a movie before the addition is the baseline. A movie without a camera file is named and the condition not run. |

The dialog plots one dot per movie for each selected condition (a model-free apparent D against time since the addition, colored by replicate), so you can see whether anything moves before choosing a readout.

#### Other settings

| Setting | What it does | Default |
|---|---|---|
| **Time resolution** | **Auto** picks a round bin width from your movies and says which; **Width in seconds** lets you type one; **Unbinned** draws no bins. | Auto |
| **Independent unit** | Movie, cell, or **Replicate** (Populations over time). With three replicates the intervals are coarse, and the card says so. | Movie |
| **Readout** | See below. | Mean mobility over time |

| Readout | What it shows |
|---|---|
| **Mean mobility over time** | One D at each time (the mean of every jump); the choice is only "no change" or "the D changed". Quickest and least noisy. A change in D or in the bound share both move it. |
| **Populations over time** | Each population's D and share against time since the addition, from one model fitted to every step at its own time. |
| **Slow fraction (HMM)** | The slow state's share against time, from a two-state HMM, with a linear and an exponential model. |

#### Populations over time

Held-out movies decide what changed (nothing, the shares, the D values or both) and how fast (linear, exponential or sigmoid). Where the shapes cannot be told apart, the first of exponential, sigmoid and linear is drawn bold and the others faintly.

| Setting | What it does |
|---|---|
| **Populations** | 2 or 3. |
| **Localisation error per spot** | Each step gets its own error through the Per spot calibration; a condition without per-spot precision is fitted without it. |
| **Also test whether movies started responding at different times** | Compares with shuffled movies. Slow (about 20 extra fits per condition); not offered for cells. |
| **Also run the permutation test** | Asks whether the order of the clock matters at all (199 shuffles within each replicate). Its p is shown beside the held-out verdict, not instead of it; the smallest p is 0.005. |
| **Compare with a known curve (optional)** | **Browse** for a CSV with columns `t_s, population, D_um2_s, share`; it is drawn dashed as "Reference". A `reference*.csv` in the project folder is picked up automatically; **Clear** drops it. |

The card shows a D panel per population over the share panel: the model as a line with its 95 % band, hollow dots for free fits per time bin and open circles for each replicate's own fit (these show the data, not the estimate). A table gives each population's values before and where the data end, and the half-time. The card says what the data cannot decide (indistinguishable shapes, an unfinished response, too few movies) and lists the window observed, the baseline, the acquisition per time window and a within-track statement.

**Slow fraction (HMM)** prints how far an untriggered slow fraction drifts. That band comes from another protein, not your project; a response has to beat it. A condition whose tracks never switch state is refused.

#### Axes, export and canceling

- **Axes** sets the time (x), the D panels' y and the share y (%). A typed range stays through a new Run until **Auto**. In the Axes popover, **time axis in** switches Seconds (default) and Minutes; it changes the display only.
- **Export** writes the figure, source data and legend. **Export all conditions…** draws every condition on one time axis. The Export dialog's **Slowest D** and **Other shares** boxes (both on) leave out the slowest D and draw only the slowest share. **Export against the mock…** draws each treated condition beside its mock.
- **Cancel** stops a run. A repeated Run on the same data is read from the kept results. A failed condition shows its message on the card.

### Clustering

*Groups tracks by the shape of their MSD.*

Clustering groups each track's MSD curve with k-means. Because k-means always finds clusters, each grouping is also made on tracks that contain none, and the card says whether the data beat that chance. A cluster is a curve shape, not a diffusive state; read it beside the SQD fractions.

**Run** opens the settings.

| Setting | What it does | Default |
|---|---|---|
| **Lags** | Lags per curve (3, 4, 6 or 8); a track needs all of them. | 4 |
| **Feature scale** | **MSD**, or **log MSD**, which damps the noisy tail. | MSD |
| **Clusters** | **Each condition alone**, or **Shared clusters**. | Each condition alone |
| **Number of clusters (k)** | **Data chooses**, or a k you set (shared clusters only). | Data chooses |
| **Largest k the data may choose** | 3, 4 or 5. | 4 |
| **Chance** | **The SQD mixture** draws chance tracks from the condition's fitted SQD populations; **One free population** draws tracks of a single D and only asks whether the curves hold more than one D. | The SQD mixture |
| **Localisation error** | For the SQD mixture: **Per spot**, **The acquisition's**, or **Set** in nm. | Per spot |
| **SQD populations** | **Best (BIC)**, 1, 2 or 3. | Best (BIC) |
| **Chance draws** | 20, 50 or 100. | 20 |

**Cancel** ends the Run's workers.

Each card opens with **Cluster structure beyond the SQD mixture at k = 3**, or **No cluster structure beyond the SQD mixture**, meaning the grouping beats chance by more than two standard deviations of the chance draws. When it does not, the clusters wait behind **Show the clusters anyway**; they are then the best partition of noise. Each cluster, slowest first, shows its share of tracks, MSD curve, D, alpha and share per replicate. Real long tracks are slower on average, so a length effect alone can read as structure beyond the mixture.

**Shared clusters** fits the clusters once on all conditions' pooled tracks, so cluster 1 is the slowest curve shape everywhere. One card shows each condition's share per cluster, a dot per replicate with mean and SD, and a table. **Test conditions** (off) adds Welch's test on the replicates' shares for each pair and cluster, Holm-adjusted; replicates are independent samples, never matched by number. A share is a fraction of tracks, not of steps, and is not an SQD fraction.

### SMTracker 2.0 view

*SMTracker 2.0's tabs and methods on the conditions you have loaded.*

**SMTracker 2.0 view**, the clock icon at the bottom of the right-hand rail, between **Help** and **Options**, opens a separate window that looks and computes like SMTracker 2.0: its tabs **Import**, **SLA**, **MSD**, **Clustering**, **GMM**, **SQD**, **APPD** and **SDA**, with SMTracker 2.0's settings, plots, tables and words, on the conditions this window has loaded. Use it to see what SMTracker 2.0 would have said about the same data.

Each tab works as it did in SMTracker 2.0: pick the settings on the left and click **Run & Plot**. Click a track in the **Import** tab's image to open the **Track Explorer**. **File > Print figures** saves the current tab's plots as 300 dpi PNGs, each name starting with `legacy_`. **Back to SMTracker 3**, at the right end of the menu bar and first in the **File** menu, closes this window and brings the analysis window back.

> **Note.** The numbers are SMTracker 2.0's, including its pooled statistics and its error bars, and they are never shown beside SMTracker 3's. For results, use the analyses in this window.

### Figures

*Export any result as a figure with its source data, significance brackets and the Composer.*

Every chart has an **Export** button. Its dialog has these rows (only those a chart can use appear):

- **For**: Nature (default), Cell, eLife, Slides, Dark slides or Poster. Colors follow the figure's page, not the screen.
- **Width**: **1/4**, **1/3**, **1/2**, **2/3**, **3/4**, **Full** (a fraction of the journal's full width: Nature 183 mm, Cell 174, eLife 170), or Custom in mm. The same fraction means the same width in the Composer.
- **Size**, **Font**: height and font; the font size applies to all text.
- **Files**: PDF and PNG at 600 dpi; SVG and TIFF are also offered. Each figure comes with source data (`.xlsx`, `.csv`) and a draft legend.
- **Title**: filled in from the project and editable.
- **Names**: how each condition is named in the figure (empty keeps its own). Grouped bubble charts list their **Group headers** and **Column labels** here.
- **Show**: the parts this chart has. For the Spatial Map: **Numbers**, **Scale bar** (length typed; empty is 1 µm), **Colour bar** and **Labels**.
- **Map scale** (Spatial Map): cell size in mm per µm. Empty is the standard, 6 mm per µm at a journal column, so map figures compare cell for cell. Where the columns do not fit, the cells shrink and the line under the preview gives the scale drawn.

Text that does not fit takes its short form ("1-pop"), then shrinks to the journal's minimum (Nature 5 pt, Cell 6, eLife 7), then wraps. The line under the preview marks anything shortened or too small. Bubble numbers sit under their bubbles; when columns are narrow, the ± goes on a second line.

#### Significance

Significance is set on the chart (SQD and Apparent Diffusion bubbles and the HMM comparison; off by default) and exported as shown. Open **Significance** on the right rail, pick two conditions, press **+** for each pair, then choose whose fraction (one population or **all**) and the test.

| Test | What it asks | Needs |
|---|---|---|
| **Welch on replicates** | The standard: each condition's replicates form one sample. | 2 replicates a side |
| **Paired t (declared pairs)** | Differences within replicate pairs declared on the Groups board. Use it only where the pairing is real. | **Pair replicates** selected, 2 shared pairs |
| **Equivalence (TOST)** | Are the two the same within a margin (5 points by default)? Shows ≈ or "not ≈". | 2 replicates a side |
| **Mixed model** | Each cell's fraction, cells within movies within replicates. | SQD or Apparent Diffusion bubbles, 2 replicates a side |
| **Bayesian model** | Hierarchical model on each cell's fraction; takes one replicate on one side. | SQD or Apparent Diffusion bubbles, 3 replicates in all |
| **Meta-analysis** | Each replicate with its own uncertainty, random effects between replicates. | SQD or Apparent Diffusion bubbles, 3 replicates a side |
| **Hierarchical bootstrap** | Replicates, then movies, then cells resampled. | SQD or Apparent Diffusion bubbles, 3 replicates a side |

The last four tests use each cell's fraction, so they need cell outlines and a D shared by a condition's replicates (**Force same D across** **Replicates** or **Whole data set**). Beside the Bayesian, meta-analysis and bootstrap p, the legend gives the design's floor ("an exact test on 3 v 3 replicates cannot go below 0.1"). Under Uncertainty **Chunks**, only Welch and equivalence are offered.

A bracket shows **stars** by default (\* p &lt; 0.05, \*\* &lt; 0.01, \*\*\* &lt; 0.001; edit under **Stars at p &lt;**), the exact p, or both. The p is Holm-adjusted over the tests shown; **Correction** offers Benjamini-Hochberg instead, and the legend names the one used. Conditions fitted with different numbers of populations cannot compare "population 1"; choose **D &lt; 0.15** (the share below that threshold). A population flagged by **Check the fit** (under 5 %, or two populations within twofold in D) is still bracketed; the flag line names it.

#### Composer

- **Add to composer** (Export dialog) collects panels; the **Composer** section lays them out as one figure. **Export every figure…** writes every computed chart into one folder.
- Each panel has a **width** (1/4 to Full); a row takes panels in order until the next would pass the full width. **Rows** is Auto or fixed.
- The **pencil** on a panel opens that chart's Export dialog with the panel's settings. **Apply to panel** changes that panel only; **Cancel** leaves it. Journal, width, size, font and files are set in the Composer.
- Every Composer export writes a `<name>.composer` file. **Open…** in the Panels card restores the figure with its data, without the project; **Save…** writes one without exporting. It holds the figure's data, so share it like source data and open only files you made.

### Analysis options

*The standards a new analysis session starts from.*

The **Analysis options** screen (the button at the foot of the analysis rail) sets the standards a new session starts from.

| Option | What it does | Default |
|---|---|---|
| **Weighting** | SQD fit: **Step** counts every displacement once; **Track** counts every track once. | Step |
| **Uncertainty** | SQD fit: **Replicates**, **None**, **Movies**, **Cells** (single session, slow) or **Chunks** (random chunks as pseudo-replicates, for data without replicates). | Replicates |
| **Chunks per condition** | Chunks each condition is dealt into (2 to 20). | 5 |
| **ECDF** | **Fast** fits at 2000 points; **Full** uses every point (slower). | Fast |
| **Optimiser** | **Fast** uses exact derivatives; **Standard** estimates them numerically. | Fast |
| **Best (BIC) also tries 4 populations** | Adds four populations to Best (BIC)'s 1 to 3 (about 70 s more per condition). | Off |
| **Outline tolerance** | Keeps a track when every point lies inside its outline widened by this distance (0 to 500 nm). Tight outlines raise the slow fraction. Applies when a project is next opened; QC says how many tracks it added. | 0 nm |
| **One frame interval for movies within** | Movies whose frame intervals differ by less than this are put on one interval; further apart, a pooled fit refuses them. 0 never rescales. Applies when a project is next opened. | 10 % |

## 5. Reference

### How SMTracker works

*How SMTracker computes its numbers, and how they were checked. For readers who want to know what happens behind each button.*

The chapters before this one explain how to use SMTracker. This Reference part explains how it works: the models behind each analysis, the files it reads and writes, and how its results were checked. Read it when you need to describe SMTracker in a methods section, review an analysis, or understand why a number came out the way it did.

> **Being written.** Most of these chapters are still to come. The planned contents are listed below.

#### Principles behind every analysis

These rules run through the whole program, and most design decisions follow from them:

- **The readout is the fraction.** Fitted D values are bounds set by the localization floor and the detection ceiling, not physical coefficients.
- **The replicate is the independent unit.** Error bars and tests are computed over replicates (or cells), never over tracks or jumps.
- **The data chooses the number of states.** BIC recommends a model; you can choose another, and the recommendation stays visible next to your choice.
- **Nothing is guessed.** A missing pixel size, frame interval or unit stops the analysis with a reason, instead of falling back to a default.

#### Validation

SMTracker 3 is a new Python implementation of the earlier MATLAB SMTracker, which is kept as a frozen reference. The SQD fit reproduces the reference to three significant figures (fractions 37.59 / 62.41 % against 37.6 / 62.4 %), and a test checks this on every build. Recovery of known diffusive states is also checked on synthetic data with a known answer.

#### Planned chapters

- **The methods, one per analysis:** SQD, jump distances, apparent diffusion, HMM, spatial maps, kinetics, clustering and MSD, with their models, assumptions and limits.
- **File formats:** every file SMTracker reads and writes, field by field (`crops.json`, `day_map.json`, `format.json`, `acquisition.json` and the session files).
- **Validation in detail:** the reference comparison and the synthetic benchmarks.

#### In Reference now

- [Reading the numbers honestly](#reading-the-numbers): what to report, and why D values are bounds.
- [Docs components](https://longestgiraffe.github.io/smtracker-site/docs/reference/components): the building blocks of these pages, for whoever writes them.

### Reading the numbers

*Report the fractions. Treat every D as a bound.*

- **Fractions are the result; D values are bounds.** The slow D sits at the localization floor, the fast D near the ceiling set by the linking radius and frame interval.
- **The independent unit is the replicate** (or the cell, within one session), never the displacement or track. Condition comparisons use two independent samples with Welch; replicate 1 of one condition shares nothing with replicate 1 of another.
- **± is a spread, not a confidence interval**, wherever it says SD over replicates.
- **Chunks are not replicates.** They show variation within one culture and cannot see a second culture.
- **"Not significant" does not mean "the same".** Use the equivalence test with a margin chosen beforehand.
- **The number of states is a decision.** BIC recommends; whether a third state is biology or localization-error heterogeneity is for you to judge and record.
- **Run QC first.** Differences in tracker, replicates, track length, outlines or crowding can change fractions on their own.

## 6. Help

### When something is refused

*Every refusal, what caused it, and how to fix it.*

Every refusal names what is missing, in the window that refuses. The wording in the window may differ slightly.

| Message or symptom | What to do |
|---|---|
| "no saved crop ranges", "edited since the last save" | Press **Save** on Calibration & save. |
| "no detection threshold" | Press **Measure** on Detect & link, or type one. |
| "nothing to segment in" | Add a `phase/` folder or a phase snapshot beside the movies. |
| "This build cannot run the segmentation models" | The segmentation package is missing. The classical segmenter and Edit outlines still work. |
| "No conditions found here" | Pick the folder that **contains** the condition folders. |
| "These movies are not tracked yet" | Track the raw movies in Preparation & tracking first. |
| "The track tables do not say their units" | State the units in `format.json` beside the tables, for example `{"units": "um"}`. |
| "The conditions were recorded at different pixel sizes" | Open conditions with one pixel size together (up to 1 % apart is accepted with a warning). |
| A movie in **Contributes nothing** | It has no outlines, its outlines hold no cell, or every track lies outside them. |
| A value in orange in **Is the comparison fair?** | Hover for the rule it breaks. |
| The app behaves like an older version | Install the current one. |
| You want to check the installation | Run `smtracker-console.exe --selftest` in a terminal (`./smtracker --selftest` on a Mac); look for "analysis self-test: OK". It needs none of your data. |
| The app closed, or a window shows an unexplained error | Restart and press **Report a problem…** (bottom left of the first window; in Analysis also in the menu under the project's name), then send the zip, with a sentence on what you were doing, to whoever looks after SMTracker for your lab. |

**Report a problem…** saves one zip: the error log, the build, package versions and SMTracker's settings file. Folder and file names appear in it; tracks, images and results do not. If no window opens, run `smtracker-console.exe --report` in a terminal (`./smtracker --report` on a Mac).

### Changelog

*What changed in each version of SMTracker.*

#### SMTracker 3.0.0

*Released 27 November 2026.* [Download](https://github.com/longestgiraffe/smtracker/releases/tag/v3.0.0)

SMTracker 3 is a new program, written in Python. It no longer needs MATLAB, and it comes with an installer for Windows and for Macs with Apple silicon ([Install and open SMTracker](#install-and-open-smtracker)).

- **From raw movies to tracks.** Crops Nikon `.nd2`, MetaMorph `.stk` and TIFF movies to their usable frames, saves the calibration, and runs TrackMate (Fiji) or u-track (MATLAB) for one condition or many folders overnight ([Image preparation + Tracking](#2-image-preparation-and-tracking)).
- **Cell outlines.** Finds cells with a classical segmenter or, after an optional download of PyTorch and Cellpose, with Cellpose-SAM; corrects them in an outline editor; and trains your own model ([Cell outlines](#3-cell-outlines)).
- **Tracks from other tools.** Reads trackpy, MOSAIC ParticleTracker, TrackIt, Spot-On, evalSPT/SLIMfast and TrackMate CSV tables, or any table whose columns and units you describe ([Tracks from other tools](#tracks-from-other-tools)).
- **Error bars over replicates.** Every fraction gets its spread over independent replicates, and conditions are compared with Welch's test, corrected with Holm or Benjamini-Hochberg ([How replicates are counted](#how-replicates-are-counted)).
- **The data choose the number of populations**, by BIC, from many random starting points.
- **Analyses:** jump distances (SQD), apparent diffusion, MSD, a hidden Markov model with switching rates, populations over time after an addition, spatial maps, clustering, and a QC page that checks whether conditions can be compared fairly ([Analysis](#4-analysis)).
- **Localization error per spot**, from a calibration measured on immobile molecules.
- **Simulated data.** Builds data sets with known populations, diffusion and switching, to compare SMTracker's results with the truth.
- **Figures for print**, each with its source data as `.xlsx` and `.csv` ([Figures](#figures)).
- **SMTracker 2.0 view.** SMTracker 2.0's eight tabs, computed its way, on the conditions you have loaded ([SMTracker 2.0 view](#smtracker-2-0-view)).
- **Installer and updates.** Installs without administrator rights, offers the segmentation pack and Fiji, and looks once a day for a newer release, which it installs only when you click ([Update SMTracker](#install-and-open-smtracker)).
- **Report on GitHub.** Report a problem saves the log as a zip and opens a new report on GitHub with the version and the end of the log filled in; nothing is sent until you submit it.
- **Fixes to SMTracker 2.0:** fraction error bars assigned to the wrong condition, a crash with one population, a limit of seven conditions, and fits that failed without saying so.

#### SMTracker 2.0

*2021*

The MATLAB version, described in Oviedo-Bocanegra et al., *Nucleic Acids Research* 49, e112 (2021), [doi:10.1093/nar/gkab696](https://doi.org/10.1093/nar/gkab696). It needs MATLAB R2014b.

#### SMTracker 1.0

*2018*

The first version, described in Rösch et al., *Scientific Reports* (2018).
