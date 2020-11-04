from Base.config import img_size, channel
from Base.model import build_model
from Base.utils import get_best_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

model_name = get_best_model(path='models/')
model = build_model()

model.load_weights(model_name)

input_a = Input((img_size, img_size, channel), name='anchor')
image_encoder = model.layers[3]
normalize = model.layers[4]
x = image_encoder(input_a)
output_a = normalize(x)
new_model = Model(input_a, output_a)
new_model.save(filepath='models/1in1out.h5')
print('Done')
