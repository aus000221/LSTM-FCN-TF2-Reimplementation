# LSTM FCN for Time Series Classification

This is a LSTM-FCN Tensorflow 2.0 reimplementation. 
It's created originally as a course project for Columbia University ECBM-4040 2024 Fall.

LSTM FCN models, from the paper [LSTM Fully Convolutional Networks for Time Series Classification](https://ieeexplore.ieee.org/document/8141873/), augment the fast classification performance of Temporal Convolutional layers with the precise classification of Long Short Term Memory Recurrent Neural Networks. We aimed to perform a detailed study to replicate many of the result and compare the classification results of the authors**

Repository: [LSTM-FCN-Ablation](https://github.com/titu1994/LSTM-FCN-Ablation)


### load_data.py
 The data can be obtained as a zip file from here - http://www.cs.ucr.edu/~eamonn/time_series_data/
Run the first three cells in main Jupyter notebook and it will create 128 datasets in csv format.
 
### build_model.py
We build the LSTM-FCN and ALSTM-FCN models through the 'build_lstm_fcn' and 'build_alstm_fcn'.

### train_model.py 
`train_lstm_fcn` function was sets up to do hyperparameter search and training LSTM-FCN and ALSTM-FCN models, which returns best_num_cells, best_accuracy, best_loss,results, best_model_paths, best_model_paths was saved for transfer learning.

Note model_name="LSTM" or "ALSTM" to do model training of LSTM-FCN and ALSTM-FCN model,respectively.
    ```
### fine_tune.py 
 `fine_tune_model` function receive the best model path and returns best_iteration, best_accuracy, best_loss, best_model_path
 
### main.ipynb
import ALSTM
from load_data import load_ucr_dataset
from train_model import train_lstm_fcn
from save_best import save_best_result
from finetune import fine_tune_model
from move import move_files
First, we loaded and preproessed data using 'load_ucr_dataset' fucntion to normalize the features of the traning data and get One-Hot Encoded labels.  

Secondly, we run 200 epochs to select best number of cells [8,64,128] using 'train_lstm_fcn' to record of best number of cells for hyper parameter search. 

After training 200 epochs, we have several .keras files which includes all trained model with three different values of cells and one csv file include the hyperparameters: best_num_cells. 

Moving these .keras files to hyper_search for clearer organization.

Thirdly, train the whole datasets for 2000 epochs using the best number of cells, which were chosen before, instead of parallel processing hyperparameters and model weights due to resources constraints.
After training 2000 epochs, now occurs several files: 
- Model files will automatically be saved in the correct directories (models) and can be used for fine tuning.
- One csv file include dataset_name, best_num_cells, accuracy,loss,best_model_paths of training.

Finally, we tried transfer learning to fine tune the model where the 'fine_tune_model' function in finetune.py receive the best model path, refer to the saved folder and model paths after training , and it will return best_iteration, best_accuracy, best_loss, best_model_path

After fine tuning we now have:
- All fine tuned models saved under (models/finetuned) of all k iterations(number of fine tuning rounds)
- One csv file includes dataset_name, best_iteration, accuracy, loss, best_model_paths of fine tuning.

The results of LSTM and ALSTM, with and without fine tuning , are saved in two csv file for reference.

Run all the operations in the main Jupyter notebook.

# Results
See the csv files of all LSTM and ALSTM  model accuracy results in our repository. All best trained model 
and fine tuned model are saved in our working directory and locally but cannot be uploaded on github repositories due to the limitations on file sizes.
Thus, A sample LSTM trained model "Beef" along with ALSTM "Beef" model, with and without fine tuning, was uploaded for reference. 
The results are compared to the authors in the tables.

# Citation
```
@article{karim2018lstm,
  title={LSTM fully convolutional networks for time series classification},
  author={Karim, Fazle and Majumdar, Somshubra and Darabi, Houshang and Chen, Shun},
  journal={IEEE Access},
  volume={6},
  pages={1662--1669},
  year={2018},
  publisher={IEEE}
}
```

# Organization of this directory
To be populated by students, as shown in previous assignments.

TODO: Create a directory/file tree
```
E4040-2024FALL-PROJECT-CJHZ-YC4324.ZZ3128.KL3606
|   E4040.2024Fall.CJHZ.report.yc4324.zz3128.kl3606.pdf
|   main.ipynb
|   README.md
|   
+---ALSTM_results
|       ALSTM_train_2000_epoches.csv
|       ALSTM_train_best_cell.csv
|       ALSTM__finetune.csv
|       
+---figures
|       e4040-2024fall-project-CJHZ-yc4324.zz3128.kl3606_gcp_work_example_screenshot_1.png
|       e4040-2024fall-project-CJHZ-yc4324.zz3128.kl3606_gcp_work_example_screenshot_2.png
|       e4040-2024fall-project-CJHZ-yc4324.zz3128.kl3606_gcp_work_example_screenshot_3.png
|       
+---LSTM_results
|       LSTM_train_2000_epoches.csv
|       LSTM_train_best_cell.csv
|       LSTM__finetune.csv
|       
+---Paper
|       LSTM_Fully_Convolutional_Networks_for_Time_Series_Classification.pdf
|       
+---saved_models
|       ALSTM__Beef__64.keras
|       finetuned_ALSTM_Beef_iteration5.keras
|       finetuned_LSTM_Beef_iteration5.keras
|       LSTM__Beef__128.keras
|       
\---utils
        ALSTM.py
        build_model.py
        data_extraction.py
        finetune.py
        load_data.py
        move.py
        read_data.py
        save_best.py
        train_model.py
```
