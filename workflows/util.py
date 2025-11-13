from pathlib import Path

import shutil
import stat
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config


def render_source(parameters):
    # --- 1) stage a working copy into tmp/ ---
    src = config.paths.model_config.resolve()
    dst = (src / "rendered").resolve()
    
    # copy everything except an existing tmp/
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("rendered"))
    
    # --- 2) set up Jinja to load from tmp/ and render files in-place ---
    env = Environment(
        loader=FileSystemLoader(str(dst)),
        undefined=StrictUndefined,         # error on missing variables
        autoescape=False,                  # plain text files
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    
    rendered = []

    
    for relname, context in parameters.items():
        relpath = Path(relname)
        target = dst / relpath
        if not target.exists():
            raise FileNotFoundError(f"Template not found in tmp/: {target}")
    
        # load by path relative to dst
        template = env.get_template(str(relpath.as_posix()))
        rendered_text = template.render(**context)
    
        # preserve original permissions when writing back
        try:
            orig_mode = target.stat().st_mode
        except FileNotFoundError:
            orig_mode = None
    
        target.write_text(rendered_text)
    
        if orig_mode is not None:
            target.chmod(stat.S_IMODE(orig_mode))
    
        rendered.append(str(target))
    
    print("Rendered files:")
    for f in rendered:
        print("  -", f)