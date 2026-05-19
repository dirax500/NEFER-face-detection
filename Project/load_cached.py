# Patch dataset.py to use .npy cache instead of .raw files
path = '/raid/didar_rakhimbay/Project/dataset.py'
with open(path) as f:
    code = f.read()

old = """    def _get_events(self, raw_path):
        if raw_path not in self.cache:
            try:
                events, _ = parse_raw_file(raw_path)
                self.cache[raw_path] = events
            except Exception:
                self.cache[raw_path] = np.zeros((0, 4), dtype=np.float64)
        return self.cache[raw_path]"""

new = """    def _get_events(self, raw_path):
        if raw_path not in self.cache:
            try:
                npy = raw_path.replace('/data/raw/', '/data/cache/').replace('.raw', '.npy')
                if os.path.isfile(npy):
                    events = np.load(npy)
                else:
                    events, _ = parse_raw_file(raw_path)
                self.cache[raw_path] = events
            except Exception:
                self.cache[raw_path] = np.zeros((0, 4), dtype=np.float64)
        return self.cache[raw_path]"""

if old in code:
    code = code.replace(old, new)
    with open(path, 'w') as f:
        f.write(code)
    print('dataset.py patched to use cache!')
else:
    print('Pattern not found - already patched or different version')
