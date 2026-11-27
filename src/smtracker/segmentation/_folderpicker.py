"""The folder picker Explorer uses, through ctypes.

Ported unchanged except for its fallback. tkinter's ``askdirectory`` shows the old
"Browse For Folder" tree AND tkinter is not bundled into the frozen build, so the
original's tkinter fallback would behave one way in development and not exist at all
in the shipped application. It is REMOVED rather than kept: the editor page already
falls back to its own in-page browser when this returns nothing, and that fallback is
the same in both builds.

`IFileOpenDialog` with `FOS_PICKFOLDERS` needs no dependency and looks identical
frozen and unfrozen. On anything that is not Windows it returns None, and the page
browses in itself.
"""

import ctypes
import ctypes.wintypes as wt
import os
import sys

CLSID_FileOpenDialog = '{DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7}'
IID_IFileOpenDialog = '{d57c7288-d4ad-4768-be02-9d969532d960}'
FOS_PICKFOLDERS = 0x00000020
FOS_FORCEFILESYSTEM = 0x00000040
SIGDN_FILESYSPATH = 0x80058000


class GUID(ctypes.Structure):
    _fields_ = [('Data1', wt.DWORD), ('Data2', wt.WORD),
                ('Data3', wt.WORD), ('Data4', ctypes.c_byte * 8)]

    def __init__(self, name):
        super().__init__()
        ole32 = ctypes.oledll.ole32
        ole32.CLSIDFromString(ctypes.c_wchar_p(name), ctypes.byref(self))


def _call(ptr, index, argtypes, *args):
    """Invoke method `index` on a COM object's vtable.

    Argument types are passed explicitly: deriving them from the values fails
    because ctypes.byref() returns a CArgObject, which has no from_param.
    """
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
    slot = ctypes.cast(vtbl + index * ctypes.sizeof(ctypes.c_void_p),
                       ctypes.POINTER(ctypes.c_void_p)).contents.value
    proto = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
    return proto(slot)(ptr, *args)


# IFileDialog vtable slots
_GET_OPTIONS, _SET_OPTIONS, _SHOW = 10, 9, 3
_SET_FOLDER, _SET_TITLE, _GET_RESULT = 12, 17, 20
_RELEASE = 2
# IShellItem
_GET_DISPLAY_NAME = 5

PDWORD = ctypes.POINTER(wt.DWORD)
PVOIDP = ctypes.POINTER(ctypes.c_void_p)
PWCHARP = ctypes.POINTER(ctypes.c_wchar_p)


def pick_folder(initial=None, title='Select experiment folder'):
    """Return a folder path, or None if cancelled/unavailable."""
    if sys.platform != 'win32':
        return None      # the page browses in itself; see the module docstring
    ole32 = ctypes.oledll.ole32
    try:
        ole32.CoInitialize(None)
    except Exception:
        pass
    ppv = ctypes.c_void_p()
    try:
        ole32.CoCreateInstance(ctypes.byref(GUID(CLSID_FileOpenDialog)), None, 1,
                               ctypes.byref(GUID(IID_IFileOpenDialog)),
                               ctypes.byref(ppv))
    except Exception:
        return None

    try:
        opts = wt.DWORD()
        _call(ppv, _GET_OPTIONS, (PDWORD,), ctypes.byref(opts))
        _call(ppv, _SET_OPTIONS, (wt.DWORD,),
              wt.DWORD(opts.value | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM))
        if title:
            _call(ppv, _SET_TITLE, (ctypes.c_wchar_p,), ctypes.c_wchar_p(title))
        if initial and os.path.isdir(initial):
            item = ctypes.c_void_p()
            try:
                ctypes.oledll.shell32.SHCreateItemFromParsingName(
                    ctypes.c_wchar_p(os.path.abspath(initial)), None,
                    ctypes.byref(GUID('{43826d1e-e718-42ee-bc55-a1e261c37bfe}')),
                    ctypes.byref(item))
                _call(ppv, _SET_FOLDER, (ctypes.c_void_p,), item)
            except Exception:
                pass

        try:
            _call(ppv, _SHOW, (ctypes.c_void_p,), None)
        except OSError:
            return None                       # cancelled: Show returns ERROR_CANCELLED

        item = ctypes.c_void_p()
        _call(ppv, _GET_RESULT, (PVOIDP,), ctypes.byref(item))
        name = ctypes.c_wchar_p()
        _call(item, _GET_DISPLAY_NAME, (wt.DWORD, PWCHARP),
              wt.DWORD(SIGDN_FILESYSPATH), ctypes.byref(name))
        path = name.value
        return path.replace('/', os.sep) if path else None
    except Exception:
        return None
    finally:
        try:
            _call(ppv, _RELEASE, ())
        except Exception:
            pass


if __name__ == '__main__':
    print(pick_folder(sys.argv[1] if len(sys.argv) > 1 else None))
