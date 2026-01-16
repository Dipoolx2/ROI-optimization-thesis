import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split

import gc

def preprocess_batch(X_batch):
    # Handle grayscale
    if X_batch.ndim == 3:
        X_batch = np.expand_dims(X_batch, -1)
    if X_batch.shape[-1] == 1:
        X_batch = np.repeat(X_batch, 3, axis=-1)
    
    # Normalize to [0,1]
    if X_batch.dtype != np.float32:
        X_batch = X_batch.astype(np.float32) / 255.0
    else:
        if X_batch.max() > 1.0:
            X_batch = X_batch / 255.0
    
    return X_batch
    
def make_generator(X_memmap, y_array, index_array, seed, batch_size, shuffle=True):
    rng = np.random.RandomState(seed)
    n = len(index_array)
    
    while True:
        if shuffle:
            order = rng.permutation(n)
        else:
            order = np.arange(n)
        
        for i in range(0, n, batch_size):
            batch_inds_local = order[i:i+batch_size]
            batch_ids = index_array[batch_inds_local]
            
            X_batch = X_memmap[batch_ids]
            y_batch = y_array[batch_ids]
            
            X_batch = preprocess_batch(X_batch)
            
            yield X_batch, y_batch

def predict_in_batches(model, X_memmap, id_array, batch_size=64):
    preds = []
    for start in range(0, len(id_array), batch_size):
        ids = id_array[start:start+batch_size]
        Xb = X_memmap[ids]
        
        Xb = preprocess_batch(Xb)
        
        p = model.predict(Xb, verbose=0)
        preds.append(p)
    
    return np.vstack(preds)
    
def build_model(input_shape, seed):    
    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.15),  # Increased from 0.055
        keras.layers.RandomTranslation(0.2, 0.2),  # Reduced from 0.2
        keras.layers.RandomZoom(0.2),  # Reduced from 0.2
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomContrast(0.15),  # Added for fundus
    ])
    
    base_model = MobileNetV2(
        include_top=False, 
        weights='imagenet',
        input_shape=input_shape,
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)

    # Augmentation only during training
    x = data_augmentation(inputs)
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Improved architecture for padded images
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    return base_model, Model(inputs=inputs, outputs=predictions)

# Custom callback
class RestoreBestWeights(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        value = logs.get(self.monitor)
        if value is None:
            return

        improved = (
            value < self.best if self.mode == "min"
            else value > self.best
        )
            
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            print(
                f"[RestoreBestWeights] Restoring weights from epoch {self.best_epoch} "
                f"({self.monitor}={self.best:.6f})"
            )
            self.model.set_weights(self.best_weights)


def run_fold(
    save_folder,
    fold_idx,
    X_loc,
    SEED,
    BATCH_SIZE,
    final
):
    tf.keras.backend.clear_session()

    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    
    # Load folds
    fold_data = np.load("./training_data/data_split_indices.npz")
    
    # filter_indices = fold_data['filter_indices']

    # Load data
    data_dict = np.load(X_loc, mmap_mode='r') # X stays on disk
    X_memmap = data_dict['X']

    y_full = np.load("./training_data/ground_truth.npz")['y']


    if final:
        train_val_idx = fold_data['train_val_indices']
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            random_state=SEED,
            shuffle=True,
            stratify=y_full[train_val_idx]
        )
    else:
        train_idx = fold_data['fold_train_indices'][fold_idx]
        val_idx = fold_data['fold_val_indices'][fold_idx]
    test_idx = fold_data['test_indices']

    print(f"  Class distribution (train): {np.bincount(y_full[train_idx])}")
    print(f"  Class distribution (val): {np.bincount(y_full[val_idx])}")
    
    train_gen = make_generator(
        X_memmap, y_full, train_idx,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_gen = make_generator(
        X_memmap, y_full, val_idx,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED
    )

    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(val_idx) / BATCH_SIZE))

    # --- model ---
    input_shape = (X_memmap.shape[1], X_memmap.shape[2], 3)
    base_model, model = build_model(input_shape, seed=SEED)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    # --- head training ---
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=50,
        callbacks=[
            RestoreBestWeights(monitor="val_loss"),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            ),
        ],
        verbose=1,
    )

    # --- fine-tuning ---
    base_model.trainable = True
    unfreeze_from = int(len(base_model.layers) * 0.7)

    for i, layer in enumerate(base_model.layers):
        layer.trainable = (
            i >= unfreeze_from
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=30,
        callbacks=[
            RestoreBestWeights(monitor="val_loss"),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-7,
            ),
        ],
        verbose=1,
    )
    
    # --- validation predictions ---
    y_hat_val = predict_in_batches(
        model, X_memmap, val_idx, batch_size=64
    )[:, 1]
    y_hat_test = predict_in_batches(
        model, X_memmap, test_idx, batch_size=64
    )[:, 1]
    
    # Saving
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    np.savez(
        f"./{save_folder}/val_preds_fold_{fold_idx}.npz",
        y_hat_test=y_hat_test,
        y_test=y_full[test_idx],
        y_hat_val=y_hat_val,
        y_val=y_full[val_idx]
    )

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--x_ident", type=str, required=True)
    parser.add_argument("--final", type=bool, required=True)
    args = parser.parse_args()

    X_loc = f"./images_data/fundus_images_{args.x_ident}.npz"
    save_folder = f"results/training_run_{args.timestamp}/{args.x_ident}"

    run_fold(save_folder, args.fold, X_loc, args.seed, args.batch_size, args.final)










    