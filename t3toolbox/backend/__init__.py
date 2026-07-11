"""The pure-functional backend: stateless functions on raw core-tuple / supercore data.

Deliberately empty: backend users import submodules explicitly (e.g.
``from t3toolbox.backend import probing``) -- each module namespaces one
(representation family) x (operation kind) cell, and each carries its own ``__all__``.
See ``docs/naming_conventions.md`` for the family prefix grammar and the module map.
"""
