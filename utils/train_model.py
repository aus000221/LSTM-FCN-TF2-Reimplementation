from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam
from datetime import datetime
import os
import numpy as np
from tensorflow.keras import backend as K

##### Self-Written packages #####
from utils.load_data import load_ucr_dataset
from utils.build_model import build_lstm_fcn,build_alstm_fcn
from utils.save_best import save_best_result
#################################

# written by yc4324 & kl3606
# Build and Train the Model
def train_lstm_fcn(model_name,dataset_name, 
                   ucr_dir = "./data",
                   num_cells_range= [8], 
                   epochs=2000, 
                   batch_size=128, 
                   verbose=1):
    """
    Train an LSTM-FCN model on a specific UCR dataset.
    """
    # Load dataset
    X_train, y_train_ohp, X_test, y_test_ohp, y_train, y_test, num_classes = load_ucr_dataset(dataset_name, ucr_dir)
    
    # Initial parameters
    initial_lr = 1e-3
    final_lr = 1e-4
    
    # create folder to save model
    models_dir = "./models/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Store results
    results = {}
    best_accuracy = 0
    best_num_cells = None
    best_model_path = None
    default_weight = 1.0
    
    # search best cell number
    for num_cells in num_cells_range:
        print(f"Training {dataset_name} with num_cells = {num_cells}")
        K.clear_session()
        
        # Build model
        input_shape =  (X_train.shape[1], X_train.shape[2])
        # Build the model
        if model_name == "LSTM":
            model = build_lstm_fcn(input_shape ,num_classes, num_cells= num_cells)
        elif model_name == "ALSTM":
            model = build_alstm_fcn(input_shape ,num_classes, num_cells= num_cells)
        else:
            print("Wrong model name, please choose LSTM or ALSTM")
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=initial_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # deal with inbalanced class weight
        # Compute class weights
        unique_labels = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=y_train
        )
        #num_classes = len(set(y_train))
        class_weights_dict = dict(zip(unique_labels, class_weights))
        # Fix the class_weight keys to be integers
        class_weights_dict = {int(k): v for k, v in class_weights_dict.items()}
        # Fill missing classes weight with default weight
        #print(f"the class weight of {dataset_name} is missing! fill with default")
        class_weights_dict = {
            i: class_weights_dict.get(i, default_weight) for i in range(num_classes)
        }
        
        # Callbacks
        #learning rate decay
        reduce_lr  = ReduceLROnPlateau(
        monitor='loss',
        factor=1 / np.cbrt(2),
        patience=100,  # Reduce learning rate after 100 epochs of no improvement
        min_lr=final_lr,
        verbose=verbose
        )
        # save best model
        file_path = os.path.join(models_dir, 
                                 f"{model.name}__{dataset_name}__{num_cells}.keras")
        model_checkpoint = ModelCheckpoint(file_path, verbose=verbose,
                                        monitor='loss', save_best_only=True, save_weights_only=False)
        
        
        if epochs > 2000:
            # Train model with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
            history = model.fit(
            X_train, y_train_ohp,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test,y_test_ohp),
            callbacks=[reduce_lr,early_stopping, model_checkpoint],
            class_weight=class_weights_dict,
            verbose=verbose
        )
        else :
            # Train model without early stopping
            history = model.fit(
                X_train, y_train_ohp,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test,y_test_ohp),
                callbacks=[reduce_lr, model_checkpoint],
                class_weight=class_weights_dict,
                verbose=verbose
            )
        
        # output model validation accuracy
        test_loss, test_accuracy = model.evaluate(X_test, y_test_ohp, verbose=0)
        print(f"dname = {dataset_name}, num_cells = {num_cells}, Test Accuracy = {test_accuracy:.4f}, Test Loss = {test_loss:.4f}")
        results[num_cells] = {'accuracy': test_accuracy, 'loss': test_loss}

        # Update best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_loss = test_loss
            best_num_cells = num_cells
            best_model_path = file_path

    print(f"BEST: dname = {dataset_name}, Best num_cells: {best_num_cells}, Test Accuracy: {best_accuracy:.4f}, Test Loss: {best_loss:.4f}")

    return best_num_cells, best_accuracy, best_loss, results, best_model_path
