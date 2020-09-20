import importlib

def instantiate(module, cls):
    try:
        print("... importing " + module)
        module = importlib.import_module(module)
        cls_instance = getattr(module, cls)
        print(cls_instance)
    except Exception as err:
        print("!!! Error creating: {0}".format(err))
        exit(-1)
    return cls_instance