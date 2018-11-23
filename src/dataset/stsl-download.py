#!/usr/bin/python3

import argparse
import asyncio
import os
import time
from multiprocessing import (cpu_count)
from urllib.request import urlopen

import cv2
import numpy as np


async def save_img(link: str, path: str):
    """Get image via HTTP and save it asynchronously."""
    cv2.imwrite(path, cv2.imdecode(np.asarray(bytearray(urlopen(link).read()), dtype='uint8'), cv2.IMREAD_COLOR))


async def save_imgs(link_path_pairs: [(str, str)]):
    """Get images via HTTP and save them asynchronously."""
    await asyncio.wait([save_img(p[0], p[1]) for p in link_path_pairs])


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


desc_str = r"""Download images from Trinity Lavra of St. Sergius - one of the most important Russian monastery.

Unfortunately, there are very few tagged data for binarizing document images, so sooner or later
you will want to create your own dataset. One of the best sources for data is the archive of
the Trinity-Sergius Lavra, because there are many old books in its archive, that answer the main
problems of the old documents. This script makes it easy to download books from there.

"""


def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(prog='stsl-download',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=desc_str)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v0.1')
    parser.add_argument('-l', '--link', type=str, default='',
                        help=r'part of link to book (for example: "http://old.stsl.ru/files/manuscripts/oldprint/1600-IV-1026/1026-").')
    parser.add_argument('-b', '--begin', type=int, default=0,
                        help=r'beginning number (default: %(default)s)')
    parser.add_argument('-e', '--end', type=int, default=0,
                        help=r'ending number (default: %(default)s)')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help=r'directory for output train and ground-truth images suitable for U-net (default: "%(default)s")')
    return parser.parse_args()


def main():
    start_time = time.time()

    args = parse_args()

    if args.link == '':
        print('no link specified, stopping program')
    else:
        mkdir_s(args.output)
        links = [((args.link + '{0:04d}.jpg').format(i),
                  os.path.join(args.output, str(i) + '_in.png'))
                 for i in range(args.begin, args.end + 1)]
        event_loop = asyncio.get_event_loop()
        try:
            event_loop.run_until_complete(save_imgs(links))
        finally:
            event_loop.close()

    print('finished in {0:.2f} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
