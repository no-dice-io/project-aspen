import os
import platform
from pathlib import Path
from pydantic import (
    BaseModel
    , PrivateAttr
    , model_validator
)
from typing import (
    Optional
    , Dict
)

PREFERRED_SETUP_DIRS = {
    "bin": {
        "{app_name}": {}
    }
    , "share": {
        "{app_name}": {
            "assistants": {}
            , "config": {}
            , "data": {
                "logs": {}
            }
        }
    }
}


class AspenDirSetup(BaseModel):
    _app_name: Optional[str] = PrivateAttr("aspen")
    _preferred_path: Optional[str] = PrivateAttr(
        os.path.expanduser('~/.local')
    )
    _system: str = PrivateAttr(platform.system().lower())
    _preferred_setup: Dict = PrivateAttr(PREFERRED_SETUP_DIRS)

    def traverse(self, data: dict, parent_keys: list=None):
        """Traverse a dictionary and return the paths of empty dictionaries"""
        if parent_keys is None:
            parent_keys = []
        results = []
        
        for key, value in data.items():
            current_path = parent_keys + [key]
            if isinstance(value, dict):
                if not value:  # Empty dictionary found
                    results.append(current_path)
                else:
                    # Recursively traverse non-empty dictionaries
                    results.extend(self.traverse(value, current_path))

        return results

    def check_perms(self, path: str) -> dict:
        """Validate permissions for a given path"""
        
        return {
            "read": os.access(path, os.R_OK)
            , "write": os.access(path, os.W_OK)
            , "execute": os.access(path, os.X_OK)
        }

    @property
    def app_name(self) -> str:
        """Return the application name"""

        return self._app_name.lower().replace(" ", "")

    @property
    def preferred_path(self) -> str:
        """Return the preferred path for the application"""

        path = (
            self._preferred_path 
            if self._system != "windows" 
            else os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        )

        if not Path(path).exists():
            err = f"Selected path, {path}, does not exist"
            raise FileNotFoundError(err)
        
        access = self.check_perms(path)
        if not all(access.values()):
            err = f"Insufficient permissions for {path}"
            raise PermissionError(err)

        return path
    
    @property
    def paths(self) -> list:
        """Return the paths to create for the application"""

        raw_paths = self.traverse(self._preferred_setup)
        raw_paths = [os.path.join(*path) for path in raw_paths]

        return [
            os.path.join(
                self.preferred_path
                , path
            ).format(app_name=self.app_name)
            for path in raw_paths
        ]
    
    def create_ignore_dir(self, path: str) -> None:
        """Create a directory if it does not exist"""
        
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)

        return path

    @model_validator(mode='after')
    def setup(self):
        """Setup the directory structure for the application"""

        # fyi
        # validation basically happens mostly within 
        # paths_to_create and preferred_path

        for path in self.paths:
            self.create_ignore_dir(path)

        return self

'''
test = AspenDirSetup()
expected_path = os.path.expanduser('~/.local')
assert test.preferred_path == expected_path, f"Expected {expected_path}, got {test.preferred_path}"
expected_paths = [
    os.path.expanduser('~/.local/bin/aspen')
    , os.path.expanduser('~/.local/share/aspen/assistants')
    , os.path.expanduser('~/.local/share/aspen/config')
    , os.path.expanduser('~/.local/share/aspen/data/logs')
]
assert test.paths_to_create == expected_paths, f"Expected {expected_paths}, got {test.paths_to_create}"
for path in expected_paths:
    assert Path(path).exists(), path  # Check if the paths exist

print("AspenDirSetup tests passed")
'''