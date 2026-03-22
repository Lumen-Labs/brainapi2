import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="plugins", help="BrainAPI plugin management")

PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", "plugins"))
REGISTRY_URL = os.getenv("PLUGIN_REGISTRY_URL", "https://registry.brain-api.dev")
PUBLISHER_ID = os.getenv("PLUGIN_PUBLISHER_ID", "")
PUBLISHER_API_KEY = os.getenv("PLUGIN_PUBLISHER_API_KEY", "")


def _get_manager():
    from src.core.plugins.manager import PluginManager

    return PluginManager(
        plugins_dir=PLUGINS_DIR,
        registry_url=REGISTRY_URL or None,
        publisher_id=PUBLISHER_ID or None,
        publisher_api_key=PUBLISHER_API_KEY or None,
    )


@app.command()
def install(
    name: str = typer.Argument(..., help="Plugin name to install"),
    version: str = typer.Option("latest", "--version", "-v", help="Plugin version"),
):
    manager = _get_manager()
    try:
        manifest = manager.install(name, version=version)
        typer.echo(f"Installed {manifest.name} v{manifest.version}")
    except Exception as exc:
        typer.echo(f"Failed to install '{name}': {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def uninstall(
    name: str = typer.Argument(..., help="Plugin name to uninstall"),
):
    manager = _get_manager()
    if manager.uninstall(name):
        typer.echo(f"Uninstalled '{name}'")
    else:
        typer.echo(f"Plugin '{name}' not found", err=True)
        raise typer.Exit(code=1)


@app.command(name="list")
def list_plugins(
    remote: bool = typer.Option(False, "--remote", "-r", help="List available plugins from registry"),
):
    manager = _get_manager()
    if remote:
        try:
            plugins = manager.list_available()
        except Exception as exc:
            typer.echo(f"Failed to fetch remote plugins: {exc}", err=True)
            raise typer.Exit(code=1)
        if not plugins:
            typer.echo("No plugins available in the registry")
            return
        typer.echo("Available plugins:")
        for p in plugins:
            typer.echo(f"  {p.name} v{p.version} - {p.description}")
    else:
        plugins = manager.list_installed()
        if not plugins:
            typer.echo("No plugins installed")
            return
        typer.echo("Installed plugins:")
        for p in plugins:
            typer.echo(f"  {p.name} v{p.version} - {p.description}")


@app.command()
def info(
    name: str = typer.Argument(..., help="Plugin name to get info for"),
):
    manager = _get_manager()
    try:
        meta = manager.get_info(name)
        typer.echo(f"Name: {meta.name}")
        typer.echo(f"Version: {meta.version}")
        typer.echo(f"Author: {meta.author}")
        typer.echo(f"Description: {meta.description}")
        if meta.versions:
            typer.echo(f"Available versions: {', '.join(meta.versions)}")
    except Exception as exc:
        typer.echo(f"Failed to get info for '{name}': {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def update(
    name: str = typer.Argument(..., help="Plugin name to update"),
):
    manager = _get_manager()
    try:
        manifest = manager.update(name)
        typer.echo(f"Updated {manifest.name} to v{manifest.version}")
    except Exception as exc:
        typer.echo(f"Failed to update '{name}': {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def publish(
    archive: Path = typer.Argument(..., help="Path to plugin archive (.tar.gz)"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Plugin name override"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing version"),
):
    manager = _get_manager()
    try:
        result = manager.publish(archive, name=name, force=force)
        typer.echo(f"Published: {result}")
    except Exception as exc:
        typer.echo(f"Failed to publish: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def depublish(
    name: str = typer.Argument(..., help="Plugin name to remove from registry"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Specific version to remove (omit to remove all)"),
):
    manager = _get_manager()
    try:
        result = manager.delete_remote(name, version=version)
        label = f"{name} v{version}" if version else name
        typer.echo(f"Removed '{label}' from registry: {result}")
    except Exception as exc:
        typer.echo(f"Failed to depublish '{name}': {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def register(
    publisher_id: Optional[str] = typer.Option(None, "--id", help="Desired publisher ID (auto-generated if omitted)"),
):
    manager = _get_manager()
    try:
        result = manager.register_publisher(publisher_id=publisher_id)
        typer.echo("")
        typer.echo(f"  Publisher registered successfully!")
        typer.echo(f"  Publisher ID:  {result['publisher_id']}")
        typer.echo(f"  API Key:       {result['api_key']}")
        typer.echo("")
        typer.echo("  Save these to your environment:")
        typer.echo(f"    export PLUGIN_PUBLISHER_ID={result['publisher_id']}")
        typer.echo(f"    export PLUGIN_PUBLISHER_API_KEY={result['api_key']}")
        typer.echo("")
    except Exception as exc:
        typer.echo(f"Failed to register: {exc}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
