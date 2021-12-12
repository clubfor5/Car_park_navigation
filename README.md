# Car_park_navigation

This is a very ugly code for carpark navigation. in Python. For the old version of SDK in Java, please contact me.  



The main entry for car localization is car_park_engine.py. However, you may look at the "未命名.ipynb" file to see 
how I use this code to generate some graphs. 

The algorithm lies in the main entry: car_park_engine.py. 

1. First step raw data files we need:   shape_description.shape (telling us drivable areas) beaconTable.cfg.
   From the files, we use "grid_generator.py" to generate grid points required. BeaconLoc.py processes the raw beacon table data.
2. At the car_park_engine.py, please see the function "modified_main()" to know the workflow. 
3. Please use SinInt to collect the raw data.

In a word, please look at the .inpy file and  modified_main()" in car_park_engine.py first and everything will be clear.  
The code is like 
"orientation,hmmErrorList,iBeaconErrorList,error_scatter,period_hmm= cpe.modified_main(folder,INS_samples=20,point_dis=grid,n_hop=6)"


There are two critical parameters: Number of hops H and Grid size d. For tuning paramters, look at the "未命名.ipynb" file.

