"""把交互式 viewer 打包成自包含单文件 HTML。

内联 vendor/three.min.js、vendor/OrbitControls.js 与 data/ 下全部场景 JSON，
去掉对本地 http 服务器和相对路径的依赖。生成的 HTML 可在内置预览面板、
或下载到任意机器双击直接打开（无需联网、无需服务器）。
"""
import json
from pathlib import Path

DEMO_A = Path("/workspace/demo_a")
SRC = DEMO_A / "index.html"
VENDOR = DEMO_A / "vendor"
DATA = DEMO_A / "data"
OUT = DEMO_A / "demo_a_viewer_standalone.html"


def safe_inline(js_text: str) -> str:
    # 避免内联文本中出现 </script> 提前闭合脚本
    return js_text.replace("</script", "<\\/script")


def main():
    html = SRC.read_text(encoding="utf-8")
    three = safe_inline((VENDOR / "three.min.js").read_text(encoding="utf-8"))
    orbit = safe_inline((VENDOR / "OrbitControls.js").read_text(encoding="utf-8"))

    # 收集数据
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    scenes = {}
    for s in manifest["scenes"]:
        name = s["scene"]
        scenes[name] = json.loads((DATA / s["file"]).read_text(encoding="utf-8"))
    data_blob = json.dumps({"manifest": manifest, "scenes": scenes}, ensure_ascii=False, separators=(",", ":"))

    # 内联 three / orbit
    html = html.replace('<script src="vendor/three.min.js"></script>',
                        "<script>\n" + three + "\n</script>")
    html = html.replace('<script src="vendor/OrbitControls.js"></script>',
                        "<script>\n" + orbit + "\n</script>")

    # 注入数据 + 改写加载器
    html = html.replace(
        "const DATA_BASE = 'data/';",
        "window.__VIEWER_DATA__ = " + data_blob + ";\nconst __DATA__ = window.__VIEWER_DATA__;\nconst DATA_BASE = null;",
    )
    html = html.replace(
        "  const data = await fetch(DATA_BASE+name+'.json').then(r=>r.json());",
        "  const data = __DATA__.scenes[name];",
    )
    html = html.replace(
        "  state.manifest = await fetch(DATA_BASE+'manifest.json').then(r=>r.json());",
        "  state.manifest = __DATA__.manifest;",
    )

    # 防御性校验：不应再残留 fetch('data/...')
    assert "fetch(DATA_BASE" not in html, "仍有未改写的 fetch 调用"

    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size // 1024
    print(f"写出自包含 viewer: {OUT}  ({size_kb} KB)")
    print(f"内联场景数: {len(scenes)}  manifest 场景: {len(manifest['scenes'])}")


if __name__ == "__main__":
    main()
