from config.settings import Logger


def get_logger(name):
    """Получение логгера с указанным именем"""
    return Logger.get_logger(name)
