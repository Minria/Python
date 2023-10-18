import logging

class Logger:
    def __init__(self, file_path='logger/logger.txt'):
        self.logging = logging
        self.file_path = file_path
        self.log_level = logging.DEBUG
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.logging.basicConfig(filename=self.file_path, level=self.log_level,
                                 format=self.log_format, datefmt=self.date_format)

    def write(self, mode='info', msg='没有输入信息'):
        if mode == 'debug':
            self.logging.debug(msg)
        if mode == 'info':
            self.logging.info(msg)
        if mode == 'warning':
            self.logging.warning(msg)
        if mode == 'error':
            self.logging.error(msg)
        if mode == 'critical':
            self.logging.critical(msg)


