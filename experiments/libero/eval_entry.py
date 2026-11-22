"""Launch evaluation without importing policy modules in spawned simulator children."""

if __name__ == "__main__":
    import os

    # Preserve Hydra's original script-relative configuration discovery.
    if not any(os.environ.get(name) for name in (
        "HYDRA_MAIN_MODULE", "FB_PAR_MAIN_MODULE", "FB_XAR_MAIN_MODULE"
    )):
        os.environ["HYDRA_MAIN_MODULE"] = "__main__"
    from eval_libero_single import eval_single_process

    eval_single_process()
