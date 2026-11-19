"""Self-test used by installers: load the model and process the reference image."""
import sys


def selftest(out_path=None) -> int:
    """Load the model, process the bundled reference image, report the device."""
    import time
    from .engine import Engine, Params, load_rgb, resource_dir, system_info
    lines = []
    log = lines.append
    try:
        engine = Engine(log=log)
        rgb = load_rgb(resource_dir() / "models" / "reference.jpg")
        t0 = time.perf_counter()
        tr = engine.process(rgb, Params())
        log(f"Reference image: {len(tr)} trichomes in {time.perf_counter() - t0:.2f} s on {engine.device_label}")
        log(f"System: {system_info()}")
        log("SELFTEST OK")
        code = 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        log(f"SELFTEST FAILED: {exc}\n{traceback.format_exc()}")
        code = 1
    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return code
