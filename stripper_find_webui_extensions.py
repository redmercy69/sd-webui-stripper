import importlib

def find_controlnet():
    """Find ControlNet external_code

    Returns:
        module: ControlNet external_code module
    """
    try:
        cnet = importlib.import_module("extensions.sd-webui-controlnet.scripts.external_code")
    except Exception:
        try:
            cnet = importlib.import_module("extensions-builtin.sd-webui-controlnet.scripts.external_code")
        except Exception:
            cnet = None

    return cnet