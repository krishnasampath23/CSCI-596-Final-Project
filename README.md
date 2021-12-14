# CSCI-596-Final-Project

 **Name:** Krishna Sampath Mangalapalli

 **USC ID:** 9852708473
 
 ### Project Idea
 
 The main goal is to solve a machine learning classification problem using Tensorflow and Cloud TPU (using free trial) and analyze the performance of our solution on TPU vs conventional GPU. 
 
 ### Comparision of TPU vs GPU (Local and Cloud) using the Flowers Dataset
 
 Here, I am using a dataset called flowers dataset which has images of five classes (which represent five different flower types). There are more than 4300 images in this dataset which allows us to observe interesting results for each hardware. I initially wanted to run with a much larger dataset, but loading a large dataset has used up all my credits in the Google Cloud free trail and stopped me from running larger tasks on TPU. 
 
 (Note: Google is offering free trail for all their cloud services (including TPU) by giving a 30 day free trial and 300$ free account credit. If we use up the account credit by using many cloud resources like I did, the free trial will expire even if the 30 days are not yet done)
 
 This dataset can be downloaded from here: https://www.kaggle.com/alxmamaev/flowers-recognition. To run my files, you have to first download the dataset from this link and extract the flowers program to the directory where you are running the code.
 
 The program for running on local GPU is GPU_Flowers_Recognition_Local_Machine.py. This basically trains on this dataset for 12 epochs using the local machine's GPU (using Tensorflow GPU). The GPU I used is NVIDIA RTX 3060 Laptop GPU. 
 
  The program for running on cloud GPU, which in this case is Google Colab is GPU_Flowers_Recognition_Colab.ipynb. This also trains for 12 epochs using the exact same code with the difference that it is run completely on google cloud. The GPU used is from Google Colab and all the dataset images are loaded to Google Drive so that the program can access it. Using dataset from google drive (cloud) is very essential as we do the same for TPU and the GPU colab program is the closest comparision for it. You can find the executions of each cell on the ipynb file itself.
  
  The program for running on cloud TPU is TPU_Flowers_Recognition.ipynb. Since initialization of TPU is different from GPU, we need to take additional steps for it. The flowers dataset is pre loaded in Google Cloud buckets (which is much faster than accessing google drive) which is a feature available. This code for using TPUs in Colab Notebooks can be found Google's official documentation.
  
  The images below show the total time taken for 12 epochs, first and second epoch of training.
  
  ![#c5f015](https://github.com/krishnasampath23/CSCI-596-Final-Project/blob/main/bar_plot_12_epochs.png?raw=true)
  
  ![alt text](https://github.com/krishnasampath23/CSCI-596-Final-Project/blob/main/bar_plot_first_epoch.png?raw=true)
  
  ![alt text](https://github.com/krishnasampath23/CSCI-596-Final-Project/blob/main/bar_plot_second_epoch.png?raw=true)
  
   ### Observations
   
   From these three, the best performance of training time is found on local GPU machine closely followed by cloud TPU. In fact, the difference between these two is almost negligible, which is amazing considering that the dataset is stored locally for the local GPU program but the cloud TPU is accessing data from the google cloud buckets. If we notice the time taken for the first epochs, we can see that TPU took a little longer as initially loadinbg data from cloud would make it slower. From the second step, we can see that TPU's performance is as good as the local machine if not better.
   
   A fair comparision of TPU would be comparing it with cloud GPU. We can observe a stark difference between their performances. Cloud GPU severely bottlenecks from loading data from google drive as opposed the fast cloud storage buckets that the TPU program uses. This can be seen clearly in the first epoch. Even for the subsequent epochs, the TPU outperforms cloud GPU, mainly due to the TPU's architecture and how it fully utilizes it's parallelism.
   
   
  ### Video Activity Dataset and Expired Free Trial
  
  Next I wanted to check the TPU's performance on the UCF 101 dataset (https://www.crcv.ucf.edu/data/UCF101.php), which has over 7 GB worth of videos of 101 different activity classes. 
  The program files for this are code_gpu.py and code_tpu.py. We also need the generator.py file to be in the same directory (this file has a class that is needed for converting videos to sequences of frames). I loaded them to a google cloud bucket but I could not execute the TPU program on cloud as I have utilized all the free credits available in the google cloud free trial. 
  
   ![alt text](https://github.com/krishnasampath23/CSCI-596-Final-Project/blob/main/Cloud_Bucket.png?raw=true)
   
   If anyone has access to Google Cloud Services, they can try to run these files to train on the UCF101 dataset. I previously executed the GPU version of the code on a local GPU machine using tensorflow GPU. It took about 12 hours to train for 150 epochs on an NVIDIA RTX 2070 Super GPU. I also tried the same on Google Colab, which did not work well as loading such large data from google drive makes it impossible to train efficiently and also google colab logs off after 12 hours of use, which is not ideal for training large datasets. It will be interesting to see the performance of TPU for this.

  
  
