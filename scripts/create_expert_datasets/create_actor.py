# Actor creation script from within the player directory

from os import path
import os
from zipfile import ZipFile

import argparse

file1_path = '__init__.py'
file2_path = 'actors.py'
file3_path = 'learn.py'
file4_path = 'pystk_actor.py'

parser = argparse.ArgumentParser(description='Zip files')

parser.add_argument('--output_zip_path', type=str, help='Output zip file path')

args = parser.parse_args()

output_zip_path = args.output_zip_path

if path.exists(output_zip_path):
    os.remove(output_zip_path)

with ZipFile(output_zip_path, 'w') as zipObj:
    zipObj.write(file1_path, path.basename(path.normpath(file1_path)))
    zipObj.write(file2_path, path.basename(path.normpath(file2_path)))
    zipObj.write(file3_path, path.basename(path.normpath(file3_path)))
    zipObj.write(file4_path, path.basename(path.normpath(file4_path)))