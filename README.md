# nexussim

NexusSim is a cutting-edge simulator designed to assess the performance of applications
deployed across the cloud-fog-edge nexus. It empowers researchers and developers to
model, analyze, and optimize interactions in distributed systems, uncovering insights
into resource usage and application behavior across all layers of the computing
continuum.


## Making Changes & Contributing

External contributors should follow the guidelines in
[CONTRIBUTING.rst](CONTRIBUTING.rst)

## Referencing

This repository has a [DOI](https://doi.org/10.82135/x7s8-ce73), which can be used for referrencing in articles or quick link.

## Requirements
- Python tested with version 3.10, 3.11, 3.12, 3.13 and 3.14
- [pyDynaa](https://github.com/TNO/cDynAA)

## Installation

After cloning the project, make sure you put the following in place:

1. Create a python environment within the root of the cloned
   repository, run:

```bash
python -m venv .venv
```

2. Install [pyDynaa](https://pypi.org/project/pydynaa/) or install [from source](https://pypi.org/project/pydynaa/).


3. Make your local repository an editable installation of the `nexussim` package:

```bash
python -m pip install --verbose --editable .
```


## Usage examples

The `examples` folder, contains several examples on how to create and run nexussim models.  They are executable python scripts.

Some examples needs extra requirements.  For installing all necessary extras, run:
```bash
pip install -r ./examples/examples_requirements.txt
```

## Development notes
### Influx
The CPU logger has tha ability to write to an influx database.
The configuration for the database can be found in `influx.ini`. To get started first copy 'influx.ini.default' to `influx.ini`. Alternatively the configuration can be set via environment variables
`INFLUXDB_V2_URL`, `INFLUXDB_V2_ORG`, `INFLUXDB_V2_TOKEN` etc. see [influxdb-client docs](https://influxdb-client.readthedocs.io/en/stable/api.html#influxdb_client.InfluxDBClient.from_env_properties)
In that case the bucket is defined via the environment variable `INFLUXDB_V2_BUCKET`.

A simple influx database can be started in docker for example as follows:
`docker run --name influx2 -p 8086:8086 -e DOCKER_INFLUXDB_INIT_MODE=setup -e DOCKER_INFLUXDB_INIT_USERNAME=admin -e DOCKER_INFLUXDB_INIT_PASSWORD=admin123 -e DOCKER_INFLUXDB_INIT_ORG=TNO -e DOCKER_INFLUXDB_INIT_BUCKET=nexussim -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-token -d influxdb:2.7`

The example above workes with the default influx.ini when run on the local machine, Ortherwise url, port and token need to be changed in both influx.ini and docker run command

### Ci pipline
There is a ci pipline. It is worth noting that this runs in main development repository in TNO's gitlab and not in the public github mirror. It uses a pydynaa wheel stored in the gitlab package registry.
