# Done by Frannecklp
# Done by Werseter
# https://github.com/Werseter

import numpy as np
import win32gui, win32ui, win32con, win32api

class GrabScreen:
    def __init__(self, region=None, window_title='Game', window_handle=None):
        self.region = region
        self.window_title = window_title
        self.window_handle = window_handle
        self.screen = WindowsScreenGrab(window_title=self.window_title,
                                        region=self.region,
                                        window_handle=self.window_handle)

    def get_frame(self):
        bits = self.screen.get_screen_bits()
        rgb = self.screen.get_rgb_from_bits(bits)
        if self.window_handle is None:
                self.window_handle = self.screen.getHandle()
        return rgb

    def clear(self):
        self.screen.clean_up()

class WindowsScreenGrab:
    _hwnd = 0

    def enumHandler(self, hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if self._search_str in title:
                self.hwnd = hwnd
    
    def __init__(self, window_title: str, region=None, window_handle=None):
        self._search_str = window_title
        if window_handle:
            self._hwnd = window_handle
        else:
            #import pdb; pdb.set_trace()
            win32gui.EnumWindows(self.enumHandler, None)
            if self._hwnd == 0:
                message = 'window_title not found: {}'.format(window_title)
                raise ValueError(message)

        hwnd = self._hwnd
        rect = win32gui.GetWindowRect(hwnd) if not region else region
        t, l, w, h = rect[0], rect[1], rect[2], rect[3]

        dataBitMap = win32ui.CreateBitMap()
        w_handle_DC = win32gui.GetWindowDC(hwnd)
        windowDC = win32ui.CreateDCFromHandle(w_handle_DC)
        memDC = windowDC.CreateCompatibleDC()
        dataBitMap.CreateCompatibleBitmap(windowDC, w, h)
        memDC.SelectObject(dataBitMap)
        
        self._w_handle_DC = w_handle_DC
        self._dataBitMap =  dataBitMap
        self._memDC = memDC
        self._windowDC = windowDC
        self._height = h
        self._width = h
        self._top = t
        self._left = l
        self._rgb = np.zeros((3, h, w), dtype=np.int)
        
    def get_screen_bits(self):
        self._memDC.StretchBlt((0,0), (self._width, self._height),
                               self._windowDC, (self._top, self._left), 
                               (self._width, self._height), win32con.SRCCOPY)
        bits = np.fromstring(self._dataBitMap.GetBitmapBits(True), np.unit8)
        bits = np.delete(bits, slice(3, None, 4))
        return bits
    
    def get_rgb_from_bits(self, bits):
        bits.shape = (self._height, self._width, 3)
        self._rgb = bits
        return self._rgb

    def clean_up(self):
        self._windowDC.DeleteDC()
        self._memDC.DeleteDC()
        win32gui.ReleaseDC(self._hwnd, self._w_handle_DC)
        win32gui.DeleteObject(self._dataBitMap.GetHandle())

    def getHandle(self):
        return self._hwnd
