import pandas as pd
from typing import List
from pathlib import Path
from collections import defaultdict
import logging
import pycocotools.mask as pct
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import argparse
import json

from analysis import mask_analysis


logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("Phenotyping")


# Reads file of type COCO and returns pandas dataframe
def phenotype_file(file: Path, scale: float, idx: int):
    name = file.stem
    logging.info(f'{idx}')

    with open(file, 'r') as f:
        data_in = json.load(f)

    data_out = defaultdict(list)

    for i, (rle, bbox, score) in enumerate(zip(data_in['masks'], data_in['bboxes'], data_in['scores'])):
        mask = pct.decode(rle)
        mask = np.ascontiguousarray(mask)

        if np.count_nonzero(mask) == 0:
            continue

        data_out['name'].append(name)
        data_out['organ_id'].append(i)
        data_out['score'].append(score)
        data_out['bbox'].append([round(p) for p in bbox])

        measurements_mm, measurements_px, mm, ml = mask_analysis(mask, scale, idx, multiple_masks=True, multiple_lengths=True, pixel_vals=True)
        length_mm, perimeter_mm, width_mm, width_m_mm, area_mm2, volume_mm3 = measurements_mm
        data_out['length_mm'].append(length_mm)
        data_out['width_mm'].append(width_mm)
        data_out['width_m_mm'].append(width_m_mm)
        data_out['perimeter_mm'].append(perimeter_mm)
        data_out['area_mm2'].append(area_mm2)
        data_out['volume_mm3'].append(volume_mm3)

        length_px, perimeter_px, width_px, width_m_px, area_px2, volume_px3 = measurements_px
        data_out['length_px'].append(length_px)
        data_out['width_px'].append(width_px)
        data_out['width_m_px'].append(width_m_px)
        data_out['perimeter_px'].append(perimeter_px)
        data_out['area_px2'].append(area_px2)
        data_out['volume_px3'].append(volume_px3)

        data_out['multiple_masks'].append(mm)
        data_out['multiple_lengths'].append(ml)

    # x = ''
    # for k, v in data_out.items():
    #     x = f'{x} {len(v)}'
    # logging.info(x)

    return data_out


def main(paths: List[str], scale: float, out: str):
    paths = [Path(p) for p in paths]
    out = Path(out)
    for p in paths:
        logger.info(str(p))

    files = []
    for p in paths:
        fs = list(p.glob('*.json'))
        files.extend(fs)

    logger.info(f"Found {len(files)} .json files")

    with ThreadPoolExecutor() as e:
        dataframes = list(e.map(phenotype_file, files, [scale] * len(files), range(len(files))))

    data = pd.concat([pd.DataFrame(d) for d in dataframes], ignore_index=True)
    data.to_csv(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Phenotype the mask output of a network.')
    parser.add_argument('directories', type=str, nargs='*')
    parser.add_argument('--scale', type=float, nargs=1, default=1.0)
    parser.add_argument('--out', type=str, nargs=1, default='./phenotype.csv')
    args = parser.parse_args()

    main(args.directories, args.scale[0], args.out[0])
