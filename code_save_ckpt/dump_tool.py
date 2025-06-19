import ast
import os
import inspect
import torch
from pathlib import Path
from typing import Set, Dict, List, Optional
import json
import pathlib
from typing import Union, Dict

class ProjectDumper:
    def __init__(self, model: Union[str, torch.nn.Module], entry_file: Optional[str] = None):
        self.entry_file = self.resolve_entry_file(entry_file).resolve()
        self.project_root = self.find_project_root()
        self.seen_files: Set[Path] = set()
        self.result: List[Dict[str, str]] = []
        self.tool_file = Path(__file__).resolve()
        self.model = self._load_model(model)

    def _load_model(self, model):
      if isinstance(model, str):
        self.model = torch.load(model, weights_only=False)
      
    def resolve_entry_file(self, entry_file: Optional[str]) -> Path:
        if entry_file:
            return Path(entry_file)

        frame = inspect.stack()[2]
        caller_file = frame.filename
        return Path(caller_file)

    def find_project_root(self) -> Path:
        path = self.entry_file.parent.resolve()
        while path != path.parent:
            if any((path / marker).exists() for marker in [".git", "pyproject.toml", "setup.py"]):
                return path
            path = path.parent
        return self.entry_file.parent.resolve()

    def parse_imports(self, source_code: str) -> Set[str]:
        tree = ast.parse(source_code)
        modules = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module)
        return modules

    def module_to_path(self, module: str) -> Optional[Path]:
        parts = module.split(".")
        py_path = self.project_root.joinpath(*parts).with_suffix(".py")
        if py_path.exists():
            return py_path.resolve()

        init_path = self.project_root.joinpath(*parts, "__init__.py")
        if init_path.exists():
            return init_path.resolve()

        return None

    def visit(self, file_path: Path):
        file_path = file_path.resolve()

        if file_path == self.tool_file:
            return

        if file_path in self.seen_files or not file_path.exists():
            return

        self.seen_files.add(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return

        rel_path = os.path.relpath(file_path, self.project_root)
        self.result.append({"path": rel_path, "source": source})

        imports = self.parse_imports(source)
        for mod in imports:
            mod_path = self.module_to_path(mod)
            if mod_path:
                self.visit(mod_path)

    def dump(self) -> List[Dict[str, str]]:
        self.visit(self.entry_file)
        new_ckpt = {
          'model':self.model,
          'deps':self.result
        } 
        print(json.dumps(new_ckpt['deps'], indent=2))
        return new_ckpt

class ProjectUpdater:
    def __init__(self, project_path: Union[str, pathlib.Path], checkpoint: Union[str, pathlib.Path, Dict]):
        self.project_path = pathlib.Path(project_path).resolve()
        self.checkpoint = self._load_checkpoint(checkpoint)
        self.deps = self.checkpoint.get('deps', [])

    def _load_checkpoint(self, checkpoint: Union[str, pathlib.Path, Dict]) -> Dict:
        if isinstance(checkpoint, dict):
            return checkpoint

        import torch
        try:
            loaded = torch.load(checkpoint, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint}: {e}")
        return loaded

    def update_files(self):
        for dep in self.deps:
            rel_path = dep['path']
            full_path = self.project_path / rel_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)

            source = dep['source']
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(source)
                print("Updated file from checkpoint to {}".format(full_path))
            except Exception as e:
                raise RuntimeError(f"Failed to update file {full_path}: {e}")


