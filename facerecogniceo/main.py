import cv2
import os
from PIL import Image
import numpy as np
from createfolder import createFolder
from train import trainprodict
from colectdata import colectdata
colectdata()
trainprodict()