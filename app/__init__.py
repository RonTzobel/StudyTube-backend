# This file makes `app` a regular Python package.
# Without it, Python 3 treats `app` as a namespace package, which can cause
# RQ's dotted-path job resolution (importlib.import_module("app.worker.jobs"))
# to fall back to attribute-walking and raise AttributeError.
