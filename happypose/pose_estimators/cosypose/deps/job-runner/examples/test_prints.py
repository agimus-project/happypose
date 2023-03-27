import sys
import logging
import time
from tqdm import tqdm
from colorama import Fore, Style

logging.basicConfig(format='%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    logger.info('A log.')
    print("A print.")
    print(f"{Fore.GREEN}A colored print{Style.RESET_ALL}.")

    secs = 10
    interval = 0.1
    iterator = tqdm(range(int(secs / interval)), file=sys.stdout)
    for n in iterator:
        iterator.set_postfix(step=n)
        time.sleep(interval)

    raise ValueError('An error.')
