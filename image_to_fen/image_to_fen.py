from train import ChessResNeXt
from train import ChessDataModule
import pytorch_lightning as pl
import torch
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import numpy as np
MODEL_PATH = "image_to_fen/models/model.ckpt"
IMAGE_PATH = "image_to_fen/images/G000_IMG001.jpg"


class ImageToFen():
    def __init__(self):
        self.model = ChessResNeXt()
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.transform = transforms.Compose([
            transforms.Resize(1024, antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])
        self.categories = pd.DataFrame([
            {"id": 0, "name": "white-pawn", "fen": "P"},
            {"id": 1, "name": "white-rook", "fen": "R"},
            {"id": 2, "name": "white-knight", "fen": "N"},
            {"id": 3, "name": "white-bishop", "fen": "B"},
            {"id": 4, "name": "white-queen", "fen": "Q"},
            {"id": 5, "name": "white-king", "fen": "K"},
            {"id": 6, "name": "black-pawn", "fen": "p"},
            {"id": 7, "name": "black-rook", "fen": "r"},
            {"id": 8, "name": "black-knight", "fen": "k"},
            {"id": 9, "name": "black-bishop", "fen":"b"},
            {"id": 10, "name": "black-queen", "fen":"q"},
            {"id": 11, "name": "black-king", "fen": "q"},
            {"id": 12, "name": "empty", "fen": 1}
        ])
        self.categories_dict = self.categories.to_dict('records')
        self.model.eval()
    
    def image_to_tensor(self, image_path: str) -> torch.Tensor:
        img = read_image(str(image_path)).float()
        img = self.transform(img)

        return img
    
    def image_to_output(self, image_path: str):
        input_tensor = self.image_to_tensor(image_path).unsqueeze(0)
        return self.model(input_tensor)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def reverse_inference(self, output_tensor: torch.Tensor):
        output_tensor = output_tensor.view(64, 13)
        decoded_output = torch.argmax(output_tensor, dim=1)
        output_list = decoded_output.tolist()
        return list(self.chunks(output_list, 8))
    
    def convert_to_fen(self, board):
        def convert_row(row):
            fen_row = ''
            empty_count = 0
            for elem in row:
                square = self.categories_dict[elem]["fen"]
                if square == 1:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += square
            if empty_count > 0:
                fen_row += str(empty_count)
            return fen_row

        fen_rows = [convert_row(row) for row in board]
        fen_string = '/'.join(fen_rows)
        return fen_string
    
    def convert(self):
        output = self.image_to_output(IMAGE_PATH)
        result = self.reverse_inference(output)
        fen = self.convert_to_fen(result)
        return fen
