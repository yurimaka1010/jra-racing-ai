"""Explicit source encodings; no magnitude-based unit inference."""
import re

def jv_time_seconds(value):
    """JV SE item 44: one minute digit, two second digits, one tenth digit."""
    if not isinstance(value,str) or not re.fullmatch(r'[0-9]{4}',value):
        return None
    if value=='0000' or int(value[1:3])>=60:
        return None
    return int(value[0])*60+int(value[1:3])+int(value[3])/10
