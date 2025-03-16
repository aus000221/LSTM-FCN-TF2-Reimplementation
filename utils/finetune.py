from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam
import os
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

##### Self-written packages #####
from utils.load_data import load_ucr_dataset
from utils.ALSTM import AttentionLSTM, AttentionLSTMCell
#################################

# written by yc4324 & zz3128
def fine_tune_model(model_name,
                    dataset_name, 
                    trained_model_path,
                    ucr_dir, 
                    initial_lr=1e-3, 
                    initial_batch_size=32,
                    k=5, 
                    epochs_per_iteration=50, 
                    verbose = 1):

    # Load dataset
    X_train, y_train_ohp, X_test, y_test_ohp, y_train, y_test, num_classes = load_ucr_dataset(dataset_name, ucr_dir)
    
    # Initialize parameters for fine-tuning
    lr = initial_lr
    batch_size = initial_batch_size
    
    # load trained model and prepare model save location
    models_dir = './models/finetune'
    os.makedirs(models_dir, exist_ok=True)
    custom_objects = {'AttentionLSTM': AttentionLSTM,"AttentionLSTMCell":AttentionLSTMCell} #for cutomized AttentionLSTM
    if os.path.exists(trained_model_path):
        model = load_model(trained_model_path,custom_objects = custom_objects) #load model
        print(f"Loaded model from {trained_model_path}")
    else:
        print(f"Warning: Initial model {trained_model_path} does not exist.")
    
    # deal with inbalanced class weight
    # Compute class weights
    unique_labels = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=y_train)
    #num_classes = len(set(y_train))
    class_weights_dict = dict(zip(unique_labels, class_weights))
    # Fix the class_weight keys to be integers
    class_weights_dict = {int(k): v for k, v in class_weights_dict.items()}
    # Fill missing classes weight with default weight
    default_weight = 1
    class_weights_dict = {
        i: class_weights_dict.get(i, default_weight) for i in range(num_classes)}
    
    # Store results
    best_accuracy = 0
    best_iteration = 0
    best_loss = np.inf
    best_model_path = None

    # start iteration
    for i in range(k):
        print(f"Fine-tuning iteration {i+1}/{k} for dataset {dataset_name}")
        K.clear_session()
        
        # Compile the model with the current learning rate
        model.compile(optimizer=Adam(learning_rate=lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Define a ModelCheckpoint callback to save the model for this iteration
        checkpoint_path = os.path.join(models_dir,f'finetuned_{model_name}_{dataset_name}_iteration{i+1}.keras')
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_weights_only=False,
                                                verbose=verbose)

        try:
            # Train the model
            history = model.fit(X_train, y_train_ohp,
                                validation_data=(X_test, y_test_ohp),
                                epochs=epochs_per_iteration,
                                batch_size=batch_size,
                                callbacks=[checkpoint_callback],
                                class_weight=class_weights_dict,
                                verbose=verbose)
        except Exception as e:
            print(f"Error during fine-tuning for {dataset_name} iteration {i+1}: {e}")
            continue

        # Get the best validation loss and accuracy from this iteration
        val_loss, val_accuracy = model.evaluate(X_test, y_test_ohp, verbose=0)
        print(f"Iteration {i+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Update best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_iteration = i+1
            best_model_path = checkpoint_path
            

        # Update learning rate and batch size
        lr /= 2  # Halve the learning rate
        # Halve batch size every alternate iteration and Ensure batch size does not go below 1
        if (i + 1) % 2 == 0: #every alternate iteration
            batch_size = max(1, batch_size // 2) # batch_size should not smaller than 1
    
    print(f"BEST: dname = {dataset_name}, Best model appears at iteration: {best_iteration}, Val Accuracy: {best_accuracy:.4f}, Val Loss: {best_loss:.4f}")

    return best_iteration, best_accuracy, best_loss, best_model_path
