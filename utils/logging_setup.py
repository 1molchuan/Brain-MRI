import logging, builtins

# Initialize basic logging for the process (only if not configured elsewhere)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Redirect built-in print to logging.info for better log capture
def _print_to_log(*args, **kwargs):
    try:
        logging.info(' '.join(str(a) for a in args))
    except Exception:
        logging.info(args)

builtins.print = _print_to_log
