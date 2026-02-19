=========
Changelog
=========

Snap
===========

v0.0.4
===========
- Network model expanded: concurrent transmissions with shared bandwidth and congestion-aware latency
- Router model introduced for network path and bandwidth decisions
- Memory model introduced with bounded allocation semantics
- CPU model completed with safer defaults and observer updates on release
- Scene simulation expanded: YAML/dict factory, time and cycle durations, allocation handling, serializable reports
- External-event wait segment added
- InfluxDB logger improved: buffered async writes, float/int encoding fixes
- Packaging consolidated to ``pyproject.toml``; linting migrated to ruff

v0.0.3
===========
- added listeners for influx db, to enable persitant storage and visualisation (e.g. on a grafana dashboard) 
- description field included in Compute model
- reader for a nexuss sim (from yaml to a dict)
- Scene model (translates the scene description into a nexus simulation)

v0.0.2
===========
- Initial concepts of a program
- Several segment behaviours included
- Basic behaviour for CPU consumption
- Several examples

v0.0.1
===========

- Initial project scaffold
- Initial concept for segments
- Initial concept for resources in the compute
- Docs can be generated
- Examples for usage initiated.
- Basic segment for constant load usage
- Basic segment for variable load usage
