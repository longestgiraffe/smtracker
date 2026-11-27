; THE WINDOWS INSTALLER (27 Nov 2026, asked: "do we provide zip or isntaller that guides through installation with
; asking what else to download?", then "prepare the installer as discussed").
;
; Compiled by `packaging/build.py --installer` with Inno Setup 6 (ISCC), which passes every value below with /D, so
; this file holds no path, no version and no address of its own: the bundle folder, the version and commit, the icon,
; the website and the Fiji page all come from the build and the package (`manual.SITE_URL`, `fiji.FIJI_WEBSITE`).
;
; One user, no administrator: SMTracker goes to %LOCALAPPDATA%\Programs\SMTracker, with a Start menu entry and an
; uninstaller in Windows' installed apps. The zip stays beside it for anyone who cannot install.
;
; WHAT ELSE TO GET is asked, every choice OFF by default:
;   - the segmentation pack: runs the program's own `--segmentation-pack install` (the same pinned, checksum-checked
;     download as the outlines window's button, one copy of that logic), in a console that stays open at the end;
;   - Fiji: opens its download page. MATLAB is licensed and can only be named.
; The uninstaller removes the program folder only: the settings, the log and a downloaded pack stay, and the pack is
; removed from the outlines window's Segmentation pack (or `smtracker-console.exe --segmentation-pack remove`).

#ifndef SourceDir
  #error SourceDir is not defined: compile this through packaging/build.py --installer
#endif

[Setup]
AppId={{AFD0981A-7465-4A96-95DC-2723FFC4B19B}
AppName=SMTracker
AppVersion={#AppVersion}
AppVerName=SMTracker {#AppVersion} (build {#Commit})
AppPublisher=SMTracker
AppPublisherURL={#SiteUrl}
AppSupportURL={#SiteUrl}
DefaultDirName={autopf}\SMTracker
DefaultGroupName=SMTracker
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#SourceDir}\LICENSE.txt
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\smtracker.exe
UninstallDisplayName=SMTracker {#AppVersion}
OutputDir={#OutputDir}
OutputBaseFilename=SMTracker-{#AppVersion}-{#Commit}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; THE UPDATER (27 Nov 2026, `smtracker.updates`): the app downloads this setup program, quits and starts it, over the
; install it came from (same AppId). Inno closes a copy still running, and its loaded .pyd files count.
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Put a shortcut on the desktop"; Flags: unchecked
Name: "segpack"; Description: "Segmentation pack for Cellpose-SAM: PyTorch and Cellpose, about 2.5 GB (also later, in the outlines window)"; GroupDescription: "What else to get:"; Flags: unchecked
Name: "fiji"; Description: "Fiji with TrackMate, for tracking: open its download page"; GroupDescription: "What else to get:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\SMTracker"; Filename: "{app}\smtracker.exe"
Name: "{autoprograms}\SMTracker (with console)"; Filename: "{app}\smtracker-console.exe"
Name: "{autodesktop}\SMTracker"; Filename: "{app}\smtracker.exe"; Tasks: desktopicon

[Run]
Filename: "{cmd}"; Parameters: "/c """"{app}\smtracker-console.exe"" --segmentation-pack install & echo. & pause"""; StatusMsg: "Downloading the segmentation pack (a console window shows the progress)..."; Tasks: segpack; Flags: waituntilterminated
Filename: "{#FijiUrl}"; Tasks: fiji; Flags: shellexec nowait
Filename: "{app}\smtracker.exe"; Description: "Start SMTracker"; Flags: postinstall nowait skipifsilent
