#!/usr/bin/env python3
"""把插件打包成 AstrBot 可安装的 ZIP。

打包规则（与 AstrBot 的安装逻辑对齐）：

* 压缩包内**只有一个顶层目录**，名字就是插件名 —— AstrBot 解压后会把
  这个目录里的内容摊平到 ``data/plugins/<插件名>/``；
* 必须包含 ``main.py`` 与 ``metadata.yaml``（缺任何一个都不算合法插件包）；
* 排除版本控制、CI、测试、缓存与本地构建产物；
* 同时输出 ``.sha256`` 校验文件。

用法::

    python scripts/build_release.py                 # 版本号取自 metadata.yaml
    python scripts/build_release.py --tag v1.0.0    # 校验标签与版本一致
    python scripts/build_release.py --outdir dist

只用标准库，Windows / Linux / macOS 都能跑。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

#: 插件目录名（= metadata.yaml 的 name）。保持与仓库目录名无关，
#: 这样仓库叫什么（neko-chem）都不影响安装后的插件目录名。
PLUGIN_NAME = "astrbot_plugin_chem_structure"

#: 不进入发布包的路径（相对仓库根目录，前缀匹配）。
#:
#: 只排除版本控制元数据、构建产物与缓存；``tests/`` 与 ``scripts/`` 是**保留**的
#: —— 这样 README 里"python tests/test_chem.py"这条自检指令对装了发布包的
#: 用户同样有效。
EXCLUDED_PREFIXES = (
    ".git/",
    ".github/",
    ".gitignore",
    ".gitattributes",
    "dist/",
    "data/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
    ".venv/",
    "venv/",
)

#: 不进入发布包的文件名后缀。
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".zip", ".sha256", ".log")

#: 发布包里必须存在的文件。
REQUIRED_FILES = ("main.py", "metadata.yaml")


def repo_root() -> Path:
    """仓库根目录 = 本脚本的上一级。"""
    return Path(__file__).resolve().parent.parent


def read_metadata_version(root: Path) -> str:
    """从 metadata.yaml 里读出 version 字段。

    只做最朴素的逐行解析，避免为打包引入 PyYAML 依赖。
    """
    metadata = root / "metadata.yaml"
    if not metadata.is_file():
        raise SystemExit(f"找不到 {metadata}")

    lines = metadata.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        key, separator, value = line.partition(":")
        if separator and key.strip() == "version":
            return value.strip().strip("\"'")

    raise SystemExit("metadata.yaml 里没有 version 字段")


def iter_payload(root: Path):
    """按排除规则遍历要打进包的文件，返回相对路径。"""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        if any(part == "__pycache__" for part in path.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path, relative


def build(root: Path, outdir: Path, tag: str | None) -> tuple[Path, Path]:
    """构建发布包，返回 ``(zip 路径, sha256 路径)``。"""
    version = read_metadata_version(root)
    if tag:
        expected = tag if tag.startswith("v") else f"v{tag}"
        if expected != f"v{version}":
            raise SystemExit(
                f"标签 {expected} 与 metadata.yaml 里的 version {version} 不一致",
            )

    release_tag = f"v{version}"
    stage = outdir / PLUGIN_NAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    written = 0
    for path, relative in iter_payload(root):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        written += 1

    for required in REQUIRED_FILES:
        if not (stage / required).is_file():
            raise SystemExit(f"发布包缺少必需文件：{required}")

    archive = outdir / f"{PLUGIN_NAME}-{release_tag}.zip"
    if archive.exists():
        archive.unlink()

    # 顶层目录名必须是插件名，且压缩包内路径统一用正斜杠
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            member = Path(PLUGIN_NAME) / path.relative_to(stage)
            zf.write(path, member.as_posix())

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

    print(f"插件名     : {PLUGIN_NAME}")
    print(f"版本       : {release_tag}")
    print(f"文件数     : {written}")
    print(f"压缩包     : {archive}")
    print(f"大小       : {archive.stat().st_size / 1024:.1f} KiB")
    print(f"SHA256     : {digest}")
    return archive, checksum


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 AstrBot 插件发布包")
    parser.add_argument("--outdir", default="dist", help="输出目录（默认 dist）")
    parser.add_argument("--tag", default=None, help="发布标签，如 v1.0.0，用于校验版本")
    args = parser.parse_args()

    root = repo_root()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    build(root, outdir, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
