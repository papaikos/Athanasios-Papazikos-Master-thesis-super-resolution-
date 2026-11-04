"""Judge App - the NPZ Viewer plus one key for judging tiles by eye.

Judge:    W = mark / unmark the tile on screen. A green check shows on marked tiles and
          the tile's full path is written to judged.txt (one per line) next to this
          script. Marks are reloaded on start, so a session can be resumed. A tile
          inside a .tar is written as the archive's path followed by its member name.

Run:      double-click judge_app.pyw   (or: python judge_app.pyw [file-or-folder])
Needs:    Python 3 + numpy  (Tkinter ships with Python)

Opens:    .npz tiles, and .tar archives containing .npz tiles (read in place,
          nothing is extracted). A folder may hold either or both.
Load:     drag & drop a file, several files, or a folder onto the window, type/paste a path in the
          box and press Enter, or use the Open buttons.
Keys:     Left/Right = prev/next tile    PgUp/PgDn = prev/next archive
          Home/End = first/last
          S = toggle contrast stretch    I = toggle provenance info
          +/- = zoom    0 = 100%    F = fit    Space = flip HR/LR (HR or LR layout)
Mouse:    wheel = zoom at pointer    drag = pan    double-click = fit
Settings: layout (side by side / stacked / HR only / LR only) and which bands show as
          R, G, B - remembered in judge_app.json next to this script.
"""
import io
import json
import math
import os
import re
import sys
import tarfile
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

GAP = 12          # px between HR and LR
CAP = 22          # px caption row height
BG = "#1e1e1e"
FG = "#d0d0d0"
GREEN = "#3ccf4e"  # the judge's check mark
# zoom steps, ~1.2x apart (12.5% .. 6400%). Whole numbers and 1/2, 1/4, 1/8 keep
# every pixel the same size on screen and take the fast path (Tk scales in C)
ZOOMS = [m * 2 ** e for e in range(-3, 6) for m in (1, 1.25, 1.5, 1.75)] + [64]
LAYOUTS = (("side", "Side by side"), ("stack", "Stacked"), ("hr", "HR"), ("lr", "LR"))


# ---------------------------------------------------------------- data ----
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


TILE_EXT = (".npz",)
ARCH_EXT = (".tar", ".tar.gz", ".tgz")


def is_source(name):
    return name.lower().endswith(TILE_EXT + ARCH_EXT)


def list_sources(folder):
    """All .npz files and .tar archives in a folder, natural-sorted."""
    try:
        names = [n for n in os.listdir(folder) if is_source(n)]
    except OSError:
        return []
    names.sort(key=natural_key)
    return [os.path.join(folder, n) for n in names]


def find_sources(folder, depth=2):
    """Sources in the folder; if none, look in its subfolders (e.g. D1 ->
    test/, train/, validation/), up to `depth` levels down."""
    srcs = list_sources(folder)
    if srcs or depth == 0:
        return srcs
    try:
        subs = sorted((e.path for e in os.scandir(folder) if e.is_dir()),
                      key=lambda p: natural_key(os.path.basename(p)))
    except OSError:
        return []
    out = []
    for d in subs:
        out += find_sources(d, depth - 1)
    return out


def index_source(path):
    """List the tiles in a source. A tile is (path, member, offset, size).

    Plain .npz -> one tile. Archive -> one tile per .npz inside; for an
    uncompressed .tar we remember each member's byte offset so a tile can be
    read with a single seek+read, without unpacking anything.
    """
    if path.lower().endswith(TILE_EXT):
        return [(path, None, None, None)]
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    with tarfile.open(path, "r:*") as tf:
        mem = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".npz")]
    mem.sort(key=lambda m: natural_key(m.name))
    if gz:
        return [(path, m.name, None, None) for m in mem]
    return [(path, m.name, m.offset_data, m.size) for m in mem]


def read_tile(tile):
    path, member, off, size = tile
    if member is None:
        return load(path)
    if off is not None:
        with open(path, "rb") as f:
            f.seek(off)
            return load(io.BytesIO(f.read(size)))
    with tarfile.open(path, "r:*") as tf:
        return load(io.BytesIO(tf.extractfile(member).read()))


def to_hwc(a):
    """Return an HxWxC view of CHW / HWC / HW input, all bands kept as stored.
    The channel axis is taken to be the short one."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a[:, :, None]
    if a.shape[0] < min(a.shape[1:]):
        return a.transpose(1, 2, 0)
    return a


def load(src):
    """Read only what we need from the npz (lazy per-key decompression)."""
    with np.load(src, allow_pickle=False) as z:
        keys = z.files
        hr_key = "hr" if "hr" in keys else next(k for k in keys if z[k].ndim >= 2)
        hr = to_hwc(z[hr_key])
        lr = to_hwc(z["lr"]) if "lr" in keys else None
        prov = str(z["provenance"]) if "provenance" in keys else ""
    return hr, lr, prov


def colorize(hr, lr, stretch, bands):
    """Pick 3 bands as R, G, B and scale to uint8 (LR uses HR's stretch).
    Only the picked bands are converted, so many-band tiles stay cheap."""
    def pick(a):
        a = a[:, :, [min(b, a.shape[2] - 1) for b in bands]]
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
        return a.astype(np.float32, copy=False)
    hr = pick(hr)
    if stretch:
        lo, hi = np.percentile(hr[::4, ::4], (1, 99))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        f = lambda a: np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255)
    else:
        f = lambda a: np.clip(a * 255.0, 0, 255)
    to8 = lambda a: (f(a) + 0.5).astype(np.uint8)
    hr8 = to8(hr)
    if lr is None:
        return hr8, None
    h, w = hr8.shape[:2]   # LR upscaled to HR size, so both panels zoom alike
    sy, sx = max(1, h // lr.shape[0]), max(1, w // lr.shape[1])
    up = np.repeat(np.repeat(to8(pick(lr)), sy, 0), sx, 1)[:h, :w]
    out = np.full_like(hr8, 0x1e)
    out[: up.shape[0], : up.shape[1]] = up
    return hr8, out


# --------------------------------------------------------- drag & drop ----
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_app.log")


def log_error(where):
    import traceback
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"--- {where}\n{traceback.format_exc()}\n")
    except Exception:
        pass


def enable_tkdnd(root, widgets, callback):
    """Use tkinterdnd2 (OLE drag & drop) when it is installed."""
    from tkinterdnd2 import DND_FILES
    for w in widgets:
        w.drop_target_register(DND_FILES)
        w.dnd_bind("<<Drop>>", lambda e: (callback(list(root.tk.splitlist(e.data))), e.action)[1])


def enable_native_drop(root, widgets, callback):
    """Windows WM_DROPFILES hook via ctypes - no extra packages needed."""
    import ctypes
    from ctypes import wintypes

    user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, ctypes.c_uint,
                                 wintypes.WPARAM, wintypes.LPARAM)
    set_wl = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
    set_wl.restype = ctypes.c_void_p
    set_wl.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    call_wp = user32.CallWindowProcW
    call_wp.restype = LRESULT
    call_wp.argtypes = [ctypes.c_void_p, wintypes.HWND, ctypes.c_uint,
                        wintypes.WPARAM, wintypes.LPARAM]
    shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                       ctypes.c_wchar_p, ctypes.c_uint]
    shell32.DragQueryFileW.restype = ctypes.c_uint
    shell32.DragFinish.argtypes = [ctypes.c_void_p]
    shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
    try:  # let drops through even if Python runs as administrator
        allow = user32.ChangeWindowMessageFilterEx
        allow.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.DWORD, ctypes.c_void_p]
    except AttributeError:
        allow = None

    old = {}
    pending = []

    def poll():
        try:
            while pending:
                callback(pending.pop(0))
        except Exception:
            log_error("drop handler")
        root.after(100, poll)
    poll()

    def proc(h, msg, wp, lp):
        if msg == 0x0233:  # WM_DROPFILES
            try:
                n = shell32.DragQueryFileW(wp, 0xFFFFFFFF, None, 0)
                files = []
                for i in range(n):
                    ln = shell32.DragQueryFileW(wp, i, None, 0) + 1
                    buf = ctypes.create_unicode_buffer(ln)
                    shell32.DragQueryFileW(wp, i, buf, ln)
                    files.append(buf.value)
                shell32.DragFinish(wp)
                # NOTE: never call Tk from inside a window procedure - doing so
                # corrupts tkinter's thread state and Python aborts silently.
                # Just queue the paths; a Tk timer picks them up (see poll).
                pending.append(files)
            except Exception:
                log_error("WM_DROPFILES")
            return 0
        return call_wp(old[h], h, msg, wp, lp)

    root._drop_proc = WNDPROC(proc)  # keep a reference alive
    root.update_idletasks()
    hwnds = [w.winfo_id() for w in widgets]
    try:
        hwnds.append(int(root.wm_frame(), 16))  # outer frame window
    except Exception:
        pass
    for h in dict.fromkeys(hwnds):
        if allow:
            for m in (0x0233, 0x004A, 0x0049):  # DROPFILES, COPYDATA, COPYGLOBALDATA
                allow(h, m, 1, None)
        shell32.DragAcceptFiles(h, True)
        old[h] = set_wl(h, -4, ctypes.cast(root._drop_proc, ctypes.c_void_p))


def enable_drop(root, widgets, callback):
    """Try tkinterdnd2, then the native Windows hook. Returns a status string."""
    if hasattr(root, "drop_target_register"):
        try:
            enable_tkdnd(root, widgets, callback)
            return "drag & drop: on"
        except Exception:
            log_error("tkinterdnd2")
    if sys.platform == "win32":
        try:
            enable_native_drop(root, widgets, callback)
            return "drag & drop: on"
        except Exception as e:
            log_error("native drop")
            return f"drag & drop failed ({e}) - see judge_app.log"
    return "drag & drop: install tkinterdnd2 (pip install tkinterdnd2)"


# ------------------------------------------------------------------ app ----
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_app.json")
MARKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judged.txt")


def tile_name(tile):
    """Full path of a tile; for one inside an archive, the archive path + member."""
    path, member = os.path.abspath(tile[0]), tile[1]
    return path if member is None else os.path.normpath(os.path.join(path, member))


def load_marks():
    """Tiles marked in earlier sessions: {normcase(name): name}, in marking order."""
    try:
        with open(MARKS, encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        names = []
    return {os.path.normcase(n): n for n in names}


def load_settings():
    """Layout and band choice saved by the last session, else the defaults."""
    s = {"layout": "side", "bands": [0, 1, 2]}
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("layout") in dict(LAYOUTS):
            s["layout"] = saved["layout"]
        bands = [int(b) for b in saved["bands"]]
        if len(bands) == 3 and min(bands) >= 0:
            s["bands"] = bands
    except Exception:
        pass
    return s


class Viewer:
    def __init__(self, root):
        self.root = root
        self.sources = []           # .npz / .tar paths in browse order
        self.tiles = {}             # source path -> list of tiles (indexed lazily)
        self.pos = None             # (source index, tile index)
        self.stretch = False
        self.cache = {}             # path -> (hr, lr, prov)
        self.rcache = {}            # (tile, stretch, bands) -> (hr, lr) as uint8 RGB
        self.lock = threading.Lock()
        self.img = None             # (hr, lr) uint8 RGB at HR size, of the tile on screen
        self.caps = []              # panel captions for that tile
        self.photos = []            # one PhotoImage per panel
        self.src, self.tmp = tk.PhotoImage(), tk.PhotoImage()   # scratch for paint()
        self.z, self.cx, self.cy = 1.0, 0.0, 0.0   # zoom, HR px at panel centre
        self.fit = True             # zoom follows the window size
        self._drag = self._redraw = None
        s = load_settings()
        self.layout, self.bands = s["layout"], s["bands"]
        self.marks = load_marks()   # tiles judged with W, as written to judged.txt

        root.title("Judge App")
        root.configure(bg=BG)

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=6, pady=(6, 3))
        tk.Button(top, text="Open file", command=self.open_file).pack(side="left")
        tk.Button(top, text="Open folder", command=self.open_folder).pack(side="left", padx=(4, 6))
        self.path_var = tk.StringVar()
        self.entry = tk.Entry(top, textvariable=self.path_var, bg="#2b2b2b", fg=FG,
                              insertbackground=FG, relief="flat")
        self.entry.pack(side="left", fill="x", expand=True, ipady=3)
        self.entry.bind("<Return>", lambda e: self.open_path(self.path_var.get()))
        self.entry.bind("<Escape>", lambda e: self.canvas.focus_set())

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0,
                                width=2 * 512 + GAP, height=512 + CAP)

        bot = self.bot = tk.Frame(root, bg=BG)
        bot.pack(side="bottom", fill="x", padx=6, pady=(3, 6))
        tk.Button(bot, text="◀ Prev", width=8, command=lambda: self.step(-1)).pack(side="left")
        tk.Button(bot, text="Next ▶", width=8, command=lambda: self.step(1)).pack(side="left", padx=4)
        self.mark_lbl = tk.Label(bot, text=f"W = mark   ({len(self.marks)} in judged.txt)",
                                 bg=BG, fg="#888")
        self.mark_lbl.pack(side="left", padx=(6, 0))
        self.status = tk.Label(bot, text="", bg=BG, fg=FG, anchor="w", width=1)
        self.status.pack(side="left", fill="x", expand=True, padx=6)
        self.info_btn = tk.Button(bot, text="Info (I)", command=self.toggle_info)
        self.info_btn.pack(side="right")
        self.str_btn = tk.Button(bot, text="Stretch: off (S)", command=self.toggle_stretch)
        self.str_btn.pack(side="right", padx=4)
        tk.Button(bot, text="Settings", command=self.toggle_settings).pack(side="right")
        tk.Button(bot, text="Fit (F)", command=self.zoom_fit).pack(side="right", padx=(4, 16))
        tk.Button(bot, text="+", width=2, command=lambda: self.zoom(1)).pack(side="right")
        self.zoom_lbl = tk.Label(bot, text="", bg=BG, fg=FG, width=5)
        self.zoom_lbl.pack(side="right")
        tk.Button(bot, text="−", width=2, command=lambda: self.zoom(-1)).pack(side="right")

        self.settings = tk.Frame(root, bg=BG)    # shown by the Settings button
        self.settings_on = False
        tk.Label(self.settings, text="Layout", bg=BG, fg=FG).pack(side="left")
        self.layout_var = tk.StringVar(value=self.layout)
        for val, txt in LAYOUTS:
            tk.Radiobutton(self.settings, text=txt, value=val, variable=self.layout_var,
                           indicatoron=False, padx=6, command=self.set_layout
                           ).pack(side="left", padx=(4, 0))
        tk.Label(self.settings, text="Bands →", bg=BG, fg=FG).pack(side="left", padx=(24, 0))
        self.band_boxes = []
        for i, ch in enumerate("RGB"):
            tk.Label(self.settings, text=ch, bg=BG, fg=FG).pack(side="left", padx=(8, 2))
            cb = ttk.Combobox(self.settings, width=3, state="readonly")
            cb.set(self.bands[i])
            cb.bind("<<ComboboxSelected>>", self.set_bands)
            cb.pack(side="left")
            self.band_boxes.append(cb)
        self.nbands = tk.Label(self.settings, text="", bg=BG, fg="#777")
        self.nbands.pack(side="left", padx=6)

        self.info = tk.Text(root, height=14, bg="#151515", fg=FG, relief="flat",
                            font=("Consolas", 9), wrap="none")
        self.info_on = False

        zoom_in, zoom_out = lambda: self.zoom(1), lambda: self.zoom(-1)
        for k, fn in (("<Left>", lambda: self.step(-1)), ("<Right>", lambda: self.step(1)),
                      ("<Prior>", lambda: self.jump(-1)), ("<Next>", lambda: self.jump(1)),
                      ("<Home>", self.first), ("<End>", self.last),
                      ("<s>", self.toggle_stretch), ("<S>", self.toggle_stretch),
                      ("<i>", self.toggle_info), ("<I>", self.toggle_info),
                      ("<plus>", zoom_in), ("<equal>", zoom_in), ("<KP_Add>", zoom_in),
                      ("<minus>", zoom_out), ("<KP_Subtract>", zoom_out),
                      ("<Key-0>", lambda: self.zoom_to(1)),
                      ("<f>", self.zoom_fit), ("<F>", self.zoom_fit),
                      ("<space>", self.toggle_flip),
                      ("<w>", self.toggle_mark), ("<W>", self.toggle_mark)):
            root.bind(k, lambda e, fn=fn: None if e.widget is self.entry else fn())
        c = self.canvas
        c.bind("<Button-1>", self.drag_start)
        c.bind("<B1-Motion>", self.drag)
        c.bind("<Double-Button-1>", lambda e: self.zoom_fit())
        c.bind("<MouseWheel>", lambda e: self.zoom(1 if e.delta > 0 else -1, (e.x, e.y)))
        c.bind("<Configure>", lambda e: self.redraw())
        c.pack(fill="both", expand=True, padx=6)  # packed last: it shrinks, not the bars

        root.after(50, self.setup_drop)

    def setup_drop(self):
        msg = enable_drop(self.root, [self.root, self.canvas], self.on_drop)
        if not self.sources:
            self.status.config(text=msg)

    def on_drop(self, paths):
        paths = [os.path.normpath(p) for p in paths if p]
        files = [p for p in paths if os.path.isfile(p) and is_source(os.path.basename(p))]
        if len(files) > 1:  # several files dropped: browse exactly those
            files.sort(key=lambda p: natural_key(os.path.basename(p)))
            self.set_sources(files, 0)
        elif paths:
            self.open_path(files[0] if files else paths[0])

    # ---- loading
    def open_file(self):
        p = filedialog.askopenfilename(filetypes=[
            ("Tiles / archives", "*.npz *.tar *.tgz *.tar.gz"), ("All files", "*.*")])
        if p:
            self.open_path(p)

    def open_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.open_path(p)

    def open_path(self, p):
        p = os.path.normpath(os.path.expanduser(p.strip().strip('"').strip("'")))
        self.canvas.focus_set()
        if os.path.isdir(p):
            srcs, start = find_sources(p), 0
            if not srcs:
                self.status.config(text=f"No .npz or .tar files in {p} (or its subfolders)")
                return
        elif os.path.isfile(p):
            srcs = list_sources(os.path.dirname(p) or ".")
            norm = [os.path.normcase(f) for f in srcs]
            if os.path.normcase(p) in norm:
                start = norm.index(os.path.normcase(p))
            else:
                srcs, start = [p], 0
        else:
            self.status.config(text=f"Not found: {p}")
            return
        self.set_sources(srcs, start)

    def set_sources(self, srcs, start):
        with self.lock:
            self.sources, self.tiles, self.cache, self.rcache = srcs, {}, {}, {}
        self.pos = None
        # start at the chosen source, or the next one that has tiles
        for si in list(range(start, len(srcs))) + list(range(start)):
            if self.tiles_of(si):
                self.go((si, 0))
                return
        self.status.config(text="No .npz tiles found")

    def tiles_of(self, si):
        path = self.sources[si]
        t = self.tiles.get(path)
        if t is None:
            try:
                t = index_source(path)
            except Exception:
                log_error(f"index {path}")
                t = []
            self.tiles[path] = t
        return t

    def get(self, tile):
        with self.lock:
            hit = self.cache.get(tile)
        if hit is None:
            hit = read_tile(tile)
            with self.lock:
                self.cache[tile] = hit
        return hit

    def get_img(self, tile, stretch, bands):
        key = (tile, stretch, bands)
        with self.lock:
            hit = self.rcache.get(key)
        if hit is None:
            hr, lr, _ = self.get(tile)
            hit = colorize(hr, lr, stretch, bands)
            with self.lock:
                self.rcache[key] = hit
        return hit

    def tile_at(self, pos):
        return self.tiles_of(pos[0])[pos[1]]

    def prefetch(self):
        near = [self.pos] + [q for q in (self.neighbor(self.pos, -1), self.neighbor(self.pos, 1)) if q]
        keep = {self.tile_at(q) for q in near}
        with self.lock:
            for k in list(self.cache):
                if k not in keep:
                    del self.cache[k]
            for k in list(self.rcache):
                if k[0] not in keep:
                    del self.rcache[k]
            st, bands = self.stretch, tuple(self.bands)
            todo = [t for t in keep if (t, st, bands) not in self.rcache]

        def work():
            for t in todo:
                try:
                    self.get_img(t, st, bands)
                except Exception:
                    pass
        if todo:
            threading.Thread(target=work, daemon=True).start()

    # ---- navigation / display
    def neighbor(self, pos, d):
        """Position of the next (d=1) / previous (d=-1) tile, crossing archives."""
        si, ti = pos
        if 0 <= ti + d < len(self.tiles_of(si)):
            return si, ti + d
        s = si + d
        while 0 <= s < len(self.sources):
            n = len(self.tiles_of(s))
            if n:
                return (s, 0) if d > 0 else (s, n - 1)
            s += d
        return None

    def step(self, d):
        if self.pos:
            q = self.neighbor(self.pos, d)
            if q:
                self.go(q)

    def jump(self, d):
        """Previous / next archive (first tile of it)."""
        if not self.pos:
            return
        s = self.pos[0] + d
        while 0 <= s < len(self.sources):
            if self.tiles_of(s):
                self.go((s, 0))
                return
            s += d

    def first(self):
        if self.sources:
            self.pos = (0, 0)
            q = (0, 0) if self.tiles_of(0) else self.neighbor((0, 0), 1)
            if q:
                self.go(q)

    def last(self):
        if self.sources:
            s = len(self.sources) - 1
            n = len(self.tiles_of(s))
            q = (s, n - 1) if n else self.neighbor((s, 0), -1)
            if q:
                self.go(q)

    def go(self, pos):
        self.pos = pos
        self.show()
        self.prefetch()

    def label(self):
        si, ti = self.pos
        tile = self.tile_at(self.pos)
        n, m = len(self.sources), len(self.tiles_of(si))
        short = lambda name: name if len(name) <= 24 else name[:12] + "…" + name[-8:]
        parent = os.path.basename(os.path.dirname(self.sources[si]))
        src = f"{parent} / {short(os.path.basename(self.sources[si]))}"
        if tile[1] is None:
            return f"{si + 1} / {n}    {src}"
        return (f"archive {si + 1} / {n}  ·  tile {ti + 1} / {m}    "
                f"{src}  ›  {short(os.path.basename(tile[1]))}")

    def show(self):
        tile = self.tile_at(self.pos)
        self.path_var.set(tile[0])
        try:
            img = self.get_img(tile, self.stretch, tuple(self.bands))
        except Exception as e:
            log_error(f"read {tile[0]} {tile[1]}")
            self.status.config(text=f"{self.label()}   - cannot read: {e}")
            self.show_mark()
            return
        if self.img is None or img[0].shape != self.img[0].shape:
            self.fit = True             # new tile size: start from fit
        self.img = img
        hr, lr, _ = self.get(tile)
        h, w = hr.shape[:2]
        self.caps = [f"HR  {w}×{h}"]
        if lr is not None:
            self.caps.append(f"LR  {lr.shape[1]}×{lr.shape[0]}  (shown ×{h // lr.shape[0]})")
        self.redraw()
        self.status.config(text=self.label())
        n = hr.shape[2]
        for cb, b in zip(self.band_boxes, self.bands):
            cb.config(values=list(range(n)))
            cb.set(min(b, n - 1))
        self.nbands.config(text=f"this tile has {n} band{'s' * (n != 1)}, numbered 0–{n - 1}")
        if self.info_on:
            self.fill_info(self.get(tile)[2])

    def panels(self):
        """Panels for the current layout: [(img, caption, x, y)] and the
        image size pw, ph they share."""
        p = [(img, cap) for img, cap in zip(self.img, self.caps) if img is not None]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if len(p) == 1 or self.layout in ("hr", "lr"):
            img, cap = p[1 if self.layout == "lr" and len(p) > 1 else 0]
            if len(p) > 1:
                cap += "    Space: HR ↔ LR"
            return [(img, cap, 0, 0)], cw, ch - CAP
        if self.layout == "stack":
            ph = (ch - GAP) // 2 - CAP
            return [q + (0, i * (CAP + ph + GAP)) for i, q in enumerate(p)], cw, ph
        pw = (cw - GAP) // 2
        return [q + (i * (pw + GAP), 0) for i, q in enumerate(p)], pw, ch - CAP

    def redraw(self):
        """Draw once the event queue is empty (merges bursts of wheel/drag events)."""
        if self._redraw is None:
            self._redraw = self.root.after_idle(self.draw)

    def draw(self):
        self._redraw = None
        c = self.canvas
        c.delete("all")
        if self.img is None:
            c.create_text(
                c.winfo_width() // 2, c.winfo_height() // 2, fill="#777", justify="center",
                text="Drop .npz / .tar files or a folder here\n"
                     "or type a path above and press Enter\n\n"
                     "←/→ prev/next tile    PgUp/PgDn prev/next archive\n"
                     "wheel or +/− zoom    drag pan    F fit    0 100%\n"
                     "S stretch    I info    W mark / unmark")
            return
        panels, pw, ph = self.panels()
        if pw < 1 or ph < 1:
            return
        h, w = self.img[0].shape[:2]
        if self.fit:  # largest zoom step that shows the whole tile
            self.z = max((s for s in ZOOMS if s <= min(pw / w, ph / h)), default=ZOOMS[0])
            self.cx, self.cy = w / 2, h / 2
        while len(self.photos) < len(panels):
            self.photos.append(tk.PhotoImage())
        for photo, (img, cap, x, y) in zip(self.photos, panels):
            self.paint(photo, img, pw, ph)
            c.create_text(x + 4, y + CAP // 2, anchor="w", fill=FG, text=cap)
            c.create_image(x, y + CAP, anchor="nw", image=photo)
        self.zoom_lbl.config(text=f"{self.z:.0%}")
        self.show_mark()

    # ---- judging
    def toggle_mark(self):
        """W: mark / unmark the tile on screen and update judged.txt right away
        (a mark is appended; an unmark rewrites the file without that line)."""
        if not self.pos:
            return
        name = tile_name(self.tile_at(self.pos))
        key = os.path.normcase(name)
        marks = dict(self.marks)
        try:
            if marks.pop(key, None) is None:
                marks[key] = name
                with open(MARKS, "a", encoding="utf-8") as f:
                    f.write(name + "\n")
            else:
                with open(MARKS, "w", encoding="utf-8") as f:
                    f.writelines(n + "\n" for n in marks.values())
        except OSError as e:
            log_error("save marks")
            self.status.config(text=f"cannot write {MARKS}: {e}")
            return
        self.marks = marks
        self.show_mark()

    def show_mark(self):
        """Green check on the canvas (top right) if the tile on screen is marked."""
        on = bool(self.pos) and os.path.normcase(tile_name(self.tile_at(self.pos))) in self.marks
        c = self.canvas
        c.delete("mark")
        if on:
            x, y = c.winfo_width() - 12, CAP + 10
            c.create_oval(x - 64, y, x, y + 64, fill=BG, outline=GREEN, width=4, tags="mark")
            c.create_text(x - 32, y + 33, text="✓", fill=GREEN,
                          font=("Segoe UI Symbol", -40, "bold"), tags="mark")
        self.mark_lbl.config(text=f"{'✓ marked' if on else 'W = mark'}   "
                                  f"({len(self.marks)} in judged.txt)",
                             fg=GREEN if on else "#888")

    def paint(self, dst, img, pw, ph):
        """Put the part of img that is in view on dst, a pw x ph PhotoImage.
        Whole-number zoom (or 1/n): numpy only crops and Tk's C code zooms - the
        fast path. In-between zoom: numpy picks a tile pixel for each visible
        screen pixel. Either way the cost follows the window, not the tile."""
        z = self.z
        h, w = img.shape[:2]
        ox, oy = round(pw / 2 - self.cx * z), round(ph / 2 - self.cy * z)  # tile corner
        dst.configure(width=pw, height=ph)    # fixed size: copies are clipped to it
        dst.blank()                           # transparent: canvas background shows
        k, n = (round(z), 1) if z >= 1 else (1, round(1 / z))
        if z == k / n:
            x0, y0 = max(0, math.floor(-ox / z)), max(0, math.floor(-oy / z))
            x0, y0 = x0 - x0 % n, y0 - y0 % n
            x1, y1 = min(w, math.ceil((pw - ox) / z)), min(h, math.ceil((ph - oy) / z))
            if x1 <= x0 or y1 <= y0:
                return
            crop = img[y0:y1:n, x0:x1:n]
            tx, ty = ox + round(x0 * z), oy + round(y0 * z)    # where the crop lands
        else:
            k = 1
            xs = np.floor((np.arange(pw) + 0.5 - ox) / z).astype(np.intp)
            ys = np.floor((np.arange(ph) + 0.5 - oy) / z).astype(np.intp)
            jx = np.flatnonzero((xs >= 0) & (xs < w))
            iy = np.flatnonzero((ys >= 0) & (ys < h))
            if not jx.size or not iy.size:
                return
            crop = img.take(ys[iy], 0).take(xs[jx], 1)
            tx, ty = int(jx[0]), int(iy[0])
        ch, cw = crop.shape[:2]
        src = self.src
        src.configure(data=b"P6 %d %d 255\n" % (cw, ch) + crop.tobytes(), format="PPM")
        if k > 1:
            self.root.tk.call(self.tmp, "copy", src, "-from", 0, 0, cw, ch, "-zoom", k, k)
            src = self.tmp
        self.root.tk.call(dst, "copy", src, "-from", max(0, -tx), max(0, -ty), cw * k, ch * k,
                          "-to", max(0, tx), max(0, ty))

    # ---- zoom / pan
    def zoom(self, d, at=None):
        """Step through ZOOMS (d=1 in, -1 out). The HR point under `at` (canvas
        x, y - the mouse) stays put; without it, the view centre does."""
        if self.img is None:
            return
        if d > 0:
            z = next((s for s in ZOOMS if s > self.z * 1.001), self.z)
        else:
            z = next((s for s in reversed(ZOOMS) if s < self.z / 1.001), self.z)
        dx = dy = 0
        if at:
            panels, pw, ph = self.panels()
            for *_, x, y in panels:
                if x <= at[0] < x + pw and y + CAP <= at[1] < y + CAP + ph:
                    dx, dy = at[0] - x - pw / 2, at[1] - y - CAP - ph / 2
        self.set_view(z, self.cx + dx / self.z - dx / z, self.cy + dy / self.z - dy / z)

    def zoom_to(self, z):
        if self.img is not None:
            self.set_view(z, self.cx, self.cy)

    def zoom_fit(self):
        self.fit = True
        self.redraw()

    def set_view(self, z, cx, cy):
        h, w = self.img[0].shape[:2]
        self.z, self.fit = z, False
        self.cx, self.cy = min(max(cx, 0), w), min(max(cy, 0), h)  # keep the tile in view
        self.redraw()

    def drag_start(self, e):
        self.canvas.focus_set()
        self._drag = (e.x, e.y, self.cx, self.cy)

    def drag(self, e):
        if self.img is not None and self._drag:
            x, y, cx, cy = self._drag
            self.set_view(self.z, cx - (e.x - x) / self.z, cy - (e.y - y) / self.z)

    def toggle_flip(self):
        """HR <-> LR layout. Not saved, so blinking doesn't rewrite the settings file."""
        if self.layout in ("hr", "lr"):
            self.layout = "lr" if self.layout == "hr" else "hr"
            self.layout_var.set(self.layout)
            self.redraw()

    # ---- settings
    def toggle_settings(self):
        self.settings_on = not self.settings_on
        if self.settings_on:
            self.settings.pack(side="bottom", fill="x", padx=6, pady=(3, 0), after=self.bot)
        else:
            self.settings.pack_forget()

    def set_layout(self):
        self.layout = self.layout_var.get()
        self.save_settings()
        self.canvas.focus_set()
        self.redraw()

    def set_bands(self, _e=None):
        self.bands = [int(cb.get()) for cb in self.band_boxes]
        self.save_settings()
        self.canvas.focus_set()
        if self.pos:
            self.show()
            self.prefetch()

    def save_settings(self):
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump({"layout": self.layout, "bands": self.bands}, f)
        except OSError:
            log_error("save settings")

    def fill_info(self, prov):
        try:
            txt = json.dumps(json.loads(prov), indent=2)
        except Exception:
            txt = prov or "(no provenance in this file)"
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", txt)
        self.info.config(state="disabled")

    def toggle_stretch(self):
        self.stretch = not self.stretch
        self.str_btn.config(text=f"Stretch: {'on' if self.stretch else 'off'} (S)",
                            relief="sunken" if self.stretch else "raised")
        if self.pos:
            self.show()
            self.prefetch()

    def toggle_info(self):
        self.info_on = not self.info_on
        if self.info_on:  # below the button bar; the canvas keeps the spare room
            self.info.pack(side="bottom", fill="x", padx=6, pady=(0, 6), before=self.bot)
            if self.pos:
                self.fill_info(self.get(self.tile_at(self.pos))[2])
        else:
            self.info.pack_forget()


def setup_crash_log():
    """pythonw has no console: send errors and hard crashes to judge_app.log."""
    try:
        f = open(LOG, "a", encoding="utf-8", buffering=1)
        import faulthandler
        faulthandler.enable(file=f)
        if sys.stderr is None:
            sys.stderr = f
        import datetime
        f.write(f"=== start {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  python {sys.version.split()[0]}"
                f"  tk {tk.TkVersion}  {sys.platform}\n")
        return f
    except Exception:
        return None


def main():
    _log = setup_crash_log()  # noqa: F841 (keep file open)
    if sys.platform == "win32":
        try:  # crisp rendering on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    try:  # optional: better drag & drop if tkinterdnd2 is installed
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    root.report_callback_exception = lambda *a: (__import__("traceback").print_exception(*a, file=sys.stderr))
    app = Viewer(root)
    if len(sys.argv) > 1:
        root.after(0, app.open_path, sys.argv[1])
    root.mainloop()


if __name__ == "__main__":
    main()
