# -*- coding: utf-8 -*-
"""
gui_component.py
----------------
A scrollable ttk.Frame used by the GUI.

Split out from ndsb_3tm_gui_magnon_compact.py.
"""
from __future__ import annotations

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None


class ScrollableFrame(ttk.Frame):  # type: ignore[misc]
    """A vertically scrollable frame (ttk) for long parameter forms."""
    def __init__(self, master, width=420, height=520, *args, **kwargs):
        if tk is None or ttk is None:
            raise RuntimeError("tkinter is required for ScrollableFrame.")
        super().__init__(master, *args, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width, height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.interior = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        def _on_interior_config(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            try:
                self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
            except Exception:
                pass

        def _on_canvas_config(event=None):
            try:
                self.canvas.itemconfigure(self._win, width=event.width)
            except Exception:
                pass

        self.interior.bind("<Configure>", _on_interior_config)
        self.canvas.bind("<Configure>", _on_canvas_config)

        # mousewheel support (Windows/macOS/Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._active = False
        self.canvas.bind("<Enter>", lambda e: setattr(self, "_active", True))
        self.canvas.bind("<Leave>", lambda e: setattr(self, "_active", False))

    def _on_mousewheel(self, event):
        if not getattr(self, "_active", False):
            return
        try:
            if hasattr(event, "delta") and event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(3, "units")
        except Exception:
            pass
