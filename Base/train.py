import argparse
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from Base.config import patience, epochs, num_train_samples, num_lfw_valid_samples, batch_size
from Base.data_generator import DataGenSequence
from Base.model import build_model
from Base.utils import get_available_cpus, ensure_folder, triplet_loss, get_smallest_loss, get_best_model

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())
    checkpoint_models_path = '../models/'
    pretrained_path = get_best_model()
    ensure_folder('models/')

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=int(patience / 2), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            smallest_loss = get_smallest_loss()
            if float(logs['val_loss']) < smallest_loss:
                self.model_to_save.save(fmt % (epoch, logs['val_loss']))


    # Load our model
    new_model = build_model()
    if pretrained_path is not None:
        new_model.load_weights(pretrained_path)

    sgd = keras.optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-6)
    # adam = keras.optimizers.Adam(lr=0.001)
    new_model.compile(optimizer=sgd, loss=triplet_loss)

    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    new_model.fit_generator(DataGenSequence('train'),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=DataGenSequence('valid'),
                            validation_steps=num_lfw_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=get_available_cpus() // 2
                            )
