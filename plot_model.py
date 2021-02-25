from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
model = load_model('best_model_101class.hdf5')
plot_model(model, to_file='best_model_101class.png', show_shapes=True,show_layer_names=True)
