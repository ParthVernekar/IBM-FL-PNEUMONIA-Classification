# IBM-FL-PNEUMONIA-Classification
PSL internship project .
-----------------------------------------------------------------------------------------------------------------------------------------------
#datasets used :
1. data64.npz - data images of dimensions 64 X 64
2. data128.npz - data images of dimensions 128 X 128
3. dataAugmented.npz - The augmented dataset of data images having dimensions 64 X 64 (The number of images in the dataset was increased here)
4. dataAugmentedWithReshape.npz - the shape of the array in the augmented dataset is (no_of_images,64,64,1)
-----------------------------------------------------------------------------------------------------------------------------------------------
#changes made in files
1. keras_fl_model.py : 
   changes in the batch size is made here
   
2. mnist_keras_data_handler.py :
   - changed number of num_classes to 1  ( due to binary classification )
   - img_rows,img_cols a.k.a image dimensions were changed here
   - an 'if' section of the code was removed here (change has been highlighted using comments in the file)
   
3. generate_configs.py :
   - changed num_classes to 1
   - img_rows,img_cols a.k.a image dimensions were changed here
   - used 5 layer CNN model
   - added 2 dense functions, used 'sigmoid' as activation function
   - added metrics precision , recall
  
4. convert_to_npz.py :
   - used pillow module to read images and convert it to greyscale
   - converted the list of images into a numpy array
   - converted the numpy array to .npz format
   
5. data_augumentation.py :
   - Due to an imbalance in the number of images in the dataset we used data_augumentation.py to generate more images 
   - used ImageDataGenerator function to increase the size of dataset by applying transformation to existing images
   - no. of images in the dataset were balanced
-----------------------------------------------------------------------------------------------------------------------------------------------
#training the model
   - to train the model on any of the five datasets (.npz files) mentioned above :-
     choose a dataset ( eg: data64.npz ) and rename it to mnist.npz 
       ( we tried to make changes in some files so that we can use the name 'data64.npz' instead of 'mnist.npz' but we found some errors which
         could not be resolved by us so we chose to renaming the 'data64.npz' file to 'mnist.npz' )
   - To proceed with further training using federated learning, we followed the IBM FL github page (quickstart.md)
----------------------------------------------------------------------------------------------------------------------------------------------
    
  






