#!/usr/bin/env python3
import soundfile as sf
from Signal import Signal

def readwavefile(filename):
    with sf.SoundFile(filename) as f:
        data=[i[0] for i in f.read()]
        sr=f.samplerate

    data=data

    return Signal(data,sr)
