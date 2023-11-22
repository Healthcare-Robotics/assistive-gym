import logging

APP_NAME = "assistive_gym"
LOG_LEVEL = logging.INFO
def get_logger():
    logger = logging.getLogger(APP_NAME)
    # # Create a handler
    # c_handler = logging.StreamHandler()
    # # link handler to logger
    # logger.addHandler(c_handler)
    logger.setLevel(LOG_LEVEL)
    return logger
