import os
import pickle


def do_or_load(f, path, **args):
    if os.path.exists(f) and os.path.isfile(f):
        print('[utils] loading checkpoint from {}'.format(path))
        return pickle.load(open(path, 'wb'))
    else:
        print('[utils] checkpoint not found, doing')
        obj = f(**args)
        print('[utils] saving checkpoing from {}'.format(path))
        pickle.dump(obj, open(path, 'rb'))
        return obj