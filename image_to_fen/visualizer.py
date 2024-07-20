import chess

from fentoimage.board import BoardImage
from image_to_fen.image_to_fen import ImageToFen

itf = ImageToFen()
fen = itf.convert()

renderer = BoardImage(fen)
image = renderer.render(highlighted_squares=(chess.F8, chess.B4))
image.show()