"""
Plant-specific topology builders.

A module in this subpackage exposes a ``build_<plant>_plant(schema)``
function returning a fully wired :class:`pyadm1.BiogasPlant` for one
physical layout. Topologies are kept as Python because the PyADM1ODE
plant API is most naturally expressed as code (component graph +
initialization); a declarative YAML topology may be introduced later if
several plants justify the abstraction.

Topologies of real plants are not part of this package - they live
outside the repository and are loaded from there.
"""

__all__: list[str] = []
