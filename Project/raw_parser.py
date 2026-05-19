from __future__ import annotations
import numpy as np

def _read_header(f):
    info = {'width': 1280, 'height': 720, 'evt': 3}
    while True:
        pos = f.tell()
        raw_line = f.readline()
        if not raw_line: break
        try:
            line = raw_line.decode('latin-1').rstrip()
        except Exception:
            f.seek(pos); break
        if not line.startswith('%'):
            f.seek(pos); break
        low = line.lower()
        if 'geometry' in low:
            for token in line.split():
                if 'x' in token:
                    try:
                        w, h = token.split('x')
                        info['width'] = int(w); info['height'] = int(h)
                    except: pass
        if 'width' in low:
            parts = line.split()
            for i, p in enumerate(parts):
                if 'width' in p.lower() and i+1 < len(parts):
                    try: info['width'] = int(parts[i+1])
                    except: pass
        if 'height' in low:
            parts = line.split()
            for i, p in enumerate(parts):
                if 'height' in p.lower() and i+1 < len(parts):
                    try: info['height'] = int(parts[i+1])
                    except: pass
    return info

def _parse_evt3(data, max_events=None):
    n_words = len(data) // 2
    if n_words == 0:
        return np.zeros((0, 4), dtype=np.float64)

    words   = np.frombuffer(data[:n_words*2], dtype='<u2')
    types   = (words >> 12) & 0xF
    payload = words & 0xFFF

    out_t, out_x, out_y, out_p = [], [], [], []

    cur_y      = 0
    cur_t      = 0.0
    time_low   = 0      # 12 bits
    time_high  = 0      # 12 bits  (bits 23:12 of timestamp)
    base_x     = 0

    for i in range(n_words):
        if max_events and len(out_t) >= max_events:
            break

        tp  = int(types[i])
        pay = int(payload[i])

        if tp == 0x0:           # ADDR_Y
            cur_y = pay & 0x7FF

        elif tp == 0x1:         # CD_X pol=0
            out_t.append(cur_t); out_x.append(pay & 0x7FF)
            out_y.append(cur_y); out_p.append(0.0)

        elif tp == 0x2:         # CD_X pol=1
            out_t.append(cur_t); out_x.append(pay & 0x7FF)
            out_y.append(cur_y); out_p.append(1.0)

        elif tp == 0x3:         # TIME_LOW — bits [11:0]
            time_low = pay & 0xFFF
            cur_t    = float((time_high << 12) | time_low)

        elif tp == 0x4:         # CONTINUED_4 — bits [15:12]
            # extends time_low with 4 more bits
            time_low = ((pay & 0xF) << 12) | time_low
            cur_t    = float((time_high << 16) | time_low)

        elif tp == 0x5:         # TIME_HIGH — bits [23:12]
            time_high = pay & 0xFFF
            time_low  = 0       # reset low bits on new high
            cur_t     = float(time_high << 12)

        elif tp == 0x6:         # VECT_BASE_X
            base_x = pay & 0xFFF

        elif tp == 0x7:         # VECT_12
            for bit in range(12):
                if max_events and len(out_t) >= max_events: break
                out_t.append(cur_t); out_x.append(float(base_x + bit))
                out_y.append(float(cur_y)); out_p.append(float((pay >> bit) & 1))
            base_x += 12

        elif tp == 0x8:         # VECT_8
            for bit in range(8):
                if max_events and len(out_t) >= max_events: break
                out_t.append(cur_t); out_x.append(float(base_x + bit))
                out_y.append(float(cur_y)); out_p.append(float((pay >> bit) & 1))
            base_x += 8

    if not out_t:
        return np.zeros((0, 4), dtype=np.float64)

    return np.stack([
        np.array(out_t, dtype=np.float64),
        np.array(out_x, dtype=np.float32),
        np.array(out_y, dtype=np.float32),
        np.array(out_p, dtype=np.float32),
    ], axis=1)

def parse_raw_file(filepath, max_events=None):
    with open(filepath, 'rb') as f:
        info = _read_header(f)
        data = f.read()
    events = _parse_evt3(data, max_events)
    if len(events) > 1:
        events = events[events[:,0].argsort()]
    return events, info

if __name__ == '__main__':
    import sys, os
    path = sys.argv[1]
    print('Parsing: %s  (%.1f MB)' % (path, os.path.getsize(path)/1e6))
    events, info = parse_raw_file(path, max_events=500_000)
    print('Sensor : %dx%d  EVT%s' % (info['width'], info['height'], info.get('evt','?')))
    print('Events : %d' % len(events))
    if len(events) > 0:
        dur = (events[-1,0] - events[0,0]) / 1e6
        print('Time   : %.4fs -> %.4fs  (%.2fs duration)' % (events[0,0]/1e6, events[-1,0]/1e6, dur))
        print('X range: %d - %d' % (int(events[:,1].min()), int(events[:,1].max())))
        print('Y range: %d - %d' % (int(events[:,2].min()), int(events[:,2].max())))
        print('Pol    : neg=%d  pos=%d' % (int((events[:,3]==0).sum()), int((events[:,3]==1).sum())))
        print(events[:5])
