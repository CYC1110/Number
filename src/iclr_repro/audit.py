"""Static release scan; findings contain locations/categories, never secret values."""
import ast
import hashlib
import re
from . import paths


def audit():
    findings=[];files=[]
    patterns={"personal_absolute_path":r"(?:/(?:home|Users|media|mnt)/[A-Za-z0-9_.-]+/|[A-Z]:\\Users\\)",
              "email":r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
              "credential_like":r"(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
              "literal_secret_assignment":r"(?i)(?:api_key|access_token|password|secret_key)\s*=\s*['\"][^'\"]{8,}['\"]"}
    for file in sorted(paths.ROOT.rglob("*")):
        if file.is_symlink():
            findings.append({"file":str(file.relative_to(paths.ROOT)),"category":"symlink"});continue
        if not file.is_file():continue
        relative=str(file.relative_to(paths.ROOT))
        if file.suffix in [".pyc",".pth",".pt",".npz",".npy",".zip",".ipynb"]:
            findings.append({"file":relative,"category":"unexpected_release_binary_or_notebook"});continue
        if any(part in [".venv","__pycache__",".git"] for part in file.parts):continue
        if file.suffix not in [".py",".json",".md",".txt",".toml",""] and file.name!=".gitignore":continue
        text=file.read_text()
        for category,pattern in patterns.items():
            for match in re.finditer(pattern,text):
                findings.append({"file":relative,"line":text.count("\n",0,match.start())+1,"category":category})
        if file.suffix==".py":
            try:ast.parse(text)
            except SyntaxError as error:findings.append({"file":relative,"line":error.lineno,"category":"syntax_error"})
        files.append({"file":relative,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
    return {"status":"PASS" if not findings else "REVIEW_REQUIRED","scanned_files":len(files),"findings":findings,
            "scope":"source/config/docs; binary assets intentionally excluded; heuristic scan plus manual review",
            "networks_by_state":paths.state_counts(),"numerical_reproduction_performed":False,
            "author_identity":"withheld","license":"undecided"}
