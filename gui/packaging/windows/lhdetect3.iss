; Inno Setup 6 script for LHDetect3; called by build_windows.ps1:
;   ISCC /DAppVersion=1.0.0 /DSourceDir=<PyInstaller folder> /DOutputDir=<folder> /DIconFile=<icon.ico> lhdetect3.iss
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6E1C2B9A-3F4D-4E57-9A1B-2C8D7E5F4A31}
AppName=LHDetect3
AppVersion={#AppVersion}
AppVerName=LHDetect3 {#AppVersion}
AppPublisher=LHDetect3 project
AppPublisherURL=https://github.com/genaev/lhdetect3
AppSupportURL=https://github.com/genaev/lhdetect3
DefaultDirName={localappdata}\Programs\LHDetect3
DefaultGroupName=LHDetect3
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=LHDetect3-{#AppVersion}-win64-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\LHDetect3.exe
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
MinVersion=10.0.17763

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\LHDetect3"; Filename: "{app}\LHDetect3.exe"
Name: "{autodesktop}\LHDetect3"; Filename: "{app}\LHDetect3.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\LHDetect3.exe"; Description: "{cm:LaunchProgram,LHDetect3}"; Flags: nowait postinstall skipifsilent
