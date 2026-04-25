import time
import random
from threading import Semaphore
from utils.logger import logger
from abc import ABC, abstractmethod


class LLMBase(ABC):
    SAFE_TO_RETRY = [429, 500, 503, 502]

    def __init__(
        self,
        max_retry=5,
        delay_sec=5,
        max_parallel_call=5,
        timeout=100_000
    ):
        self.max_retry = max_retry
        self.delay_sec = delay_sec
        self.max_parallel_call = max_parallel_call
        self.timeout = timeout

        self.semaphore = Semaphore(self.MAX_PARALLEL_CALL)

    def handle_request(self, prompt):
        for i in range(self.max_retry):
            self.semaphore.acquire()

            try:
                response = self.generate(prompt)
                self.semaphore.release()
                # TODO: implement token logging later
                # logger.info(f'Success - Remaining tokens:{self.token_bucket.tokens}')
                return response

            except Exception as e:
                status_code, error_message = self.extract_message(e)
                logger.error(f'{error_message}')

                if status_code not in self.SAFE_TO_RETRY:
                    return

                jitter = random.random() * 10
                delay_sec = self.delay_sec * (2 ** i) + jitter
                logger.info(f'Retrying after {delay_sec} secs - attempt {i}')
                time.sleep(delay_sec)

            logger.info(
                f'Request failed after {self.max_retry} out of retries')

    @abstractmethod
    def generate(self, prompt):
        ...

    @abstractmethod
    def extract_message(self, e):
        ...
