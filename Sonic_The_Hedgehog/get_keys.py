import win32api as wapi
import time

keyList = [i for i in "\b" + "ABCDEFGHIJKLMNOPQRSTUVQXYZ 123456789,.'APS$/\\"]

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
