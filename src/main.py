import random
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import scrython
from einops import rearrange
from fpdf import FPDF
from pyfzf.pyfzf import FzfPrompt
from tqdm import tqdm


def load_deck(path):
    if path.suffix == 'csv':
        deck = pd.read_csv(path, names=['name', 'count'])
    else:
        with path.open('r') as file:
            lines = file.read().splitlines()
        cards = [re.split(" ", line, maxsplit=1) for line in lines if line != ""]
        deck = pd.DataFrame(cards, columns=['count', 'name'])
        deck['count'] = deck['count'].astype(int)

    return deck.sort_values(by=['count', 'name'])


def get_infos_images(infos, image_list):
    card_data = scrython.cards.Named(fuzzy=infos['name'], format='json')

    if card_data.type_line().startswith('Basic Land'):
        candidates = scrython.cards.Search(q=f"++{infos['name']}").data()
        image_list.extend(map(fetch_image, random.sample(candidates, infos['count'])))
    else:
        image = fetch_image(card_data)
        image_list.extend([image] * int(infos['count']))

    time.sleep(0.05)


def fetch_image(card_data):
    if isinstance(card_data, dict):
        image_url = card_data['image_uris']['normal']
    else:
        image_url = card_data.image_uris()['normal']

    img_data = requests.get(image_url, stream=True).raw
    image = np.asarray(bytearray(img_data.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def tile_in_pages(image_list):
    COLS_PER_PAGE = 3
    ROWS_PER_PAGE = 3
    CARDS_PER_PAGE = COLS_PER_PAGE * ROWS_PER_PAGE

    nb_images = len(image_list)
    nb_missing = CARDS_PER_PAGE - (nb_images % CARDS_PER_PAGE)

    img_shape = np.asarray(image_list[0]).shape
    card_height, card_width, channels = img_shape

    WHITE = 255
    canvas = np.vstack([image_list, np.full([nb_missing, *img_shape], WHITE)])
    canvas = rearrange(canvas, '(p rows cols) h w c -> p (rows h) (cols w) c', rows=3, cols=3)

    W_INCHES, H_INCHES = 2.49, 3.48
    MISSING_WIDTH = (8.5 - W_INCHES * COLS_PER_PAGE) / W_INCHES * card_width
    MISSING_HEIGHT = (11 - H_INCHES * ROWS_PER_PAGE) / H_INCHES * card_height

    padding = np.zeros([len(canvas.shape), 1], dtype=int)
    padding[1, 0] = round(MISSING_HEIGHT / 2)
    padding[2, 0] = round(MISSING_WIDTH / 2)

    canvas = np.pad(canvas, padding, constant_values=WHITE)
    return canvas


def generate_pdf(path, canvas):
    with tempfile.TemporaryDirectory() as dirname:
        pdf = FPDF()

        out_dir = Path(dirname)
        for i, page in enumerate(canvas):
            out_path = (out_dir / f"page_{i}.jpg").as_posix()
            cv2.imwrite(out_path, page)

            pdf.add_page()
            pdf.image(out_path, 0, 0, 210, 297)

        pdf.output(path.with_suffix(".pdf"), 'F')


if __name__ == "__main__":
    fzf = FzfPrompt()
    choices = [path for path in Path().rglob("*.[tc][xs][tv]") if not str.startswith(path.as_posix(), '.venv/')]
    path = Path(fzf.prompt(choices)[0])

    deck = load_deck(path)

    image_list = []
    for idx, infos in tqdm(deck.iterrows(), total=len(deck)):
        get_infos_images(infos, image_list)

    canvas = tile_in_pages(image_list)
    generate_pdf(path, canvas)
