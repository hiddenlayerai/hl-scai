import logging
import sys

import click

from ..config.settings import get_config
from ..core import Agent

# Configure basic logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr
)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("-d", "--directory", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
def scan(directory: str) -> None:
    config = get_config()
    agent = Agent(config)

    results = agent.analyze_directory(directory)
    click.echo(results.model_dump_json())


if __name__ == "__main__":
    cli()
