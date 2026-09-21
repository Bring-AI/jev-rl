"""Build a self-contained static replay site; no API key or backend is included."""

import argparse
import shutil
from pathlib import Path

from jev_reward.server import STATIC

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=Path,
    default=(STATIC / "jev.json" if (STATIC / "jev.json").exists() else STATIC / "demo.json"),
)
parser.add_argument("--output", type=Path, default=Path("dist/site"))
args = parser.parse_args()
assets = args.output / "static"
assets.mkdir(parents=True, exist_ok=True)
shutil.copy2(STATIC / "classic" / "index.html", args.output / "index.html")
key_quest = args.output / "key-quest"
key_quest.mkdir(exist_ok=True)
shutil.copy2(STATIC / "index.html", key_quest / "index.html")
shutil.copytree(STATIC / "classic", assets / "classic", dirs_exist_ok=True)
for name in ("app.js", "style.css"):
    shutil.copy2(STATIC / name, assets / name)
shutil.copy2(args.run, assets / "demo.json")
# Use relative paths so the export also works below a GitHub Pages project prefix.
home = args.output / "index.html"
home.write_text(
    home.read_text()
    .replace('"/static/', '"static/')
    .replace('href="/key-quest"', 'href="key-quest/"')
    .replace('href="/"', 'href="./"')
)
classic_js = assets / "classic" / "app.js"
classic_js.write_text(
    classic_js.read_text().replace('"/static/', '"static/').replace("`/static/", "`static/")
)
legacy = key_quest / "index.html"
legacy.write_text(
    legacy.read_text().replace('"/static/', '"../static/').replace('"static/', '"../static/')
)
(key_quest / "static").mkdir(exist_ok=True)
for name in ("demo.json",):
    shutil.copy2(assets / name, key_quest / "static" / name)
(args.output / ".nojekyll").touch()
print(f"Static replay exported to {args.output.resolve()}")
