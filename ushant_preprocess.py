import os
import numpy as np
import pandas as pd

# a final pre-processed dataframe
final_df = pd.DataFrame(columns=('trip_id', 'traj_long_lat','traj_lat_long',
                                 'avg_x_velocty','avg_y_velocity','total_duration_seconds'))

path = "data" # path to data file

# loop through each .txt and grab the relevant data

for i, filename in enumerate(os.listdir(path)):

    traject = np.loadtxt(f"data/{filename}", delimiter = ";", skiprows = 1)

    traj_df = pd.DataFrame(traject, columns = ["long", "lat", "x_velocity",
                                               "y_velocity","seconds_since_start"])

    final_df.loc[i] = [i,
                       list(zip(traj_df["long"], traj_df["lat"])),
                       list(zip(traj_df["lat"], traj_df["long"])),
                       traj_df["x_velocity"].mean(),
                       traj_df["y_velocity"].mean(),
                       traj_df["seconds_since_start"].max()]

# dump to csv and pkl
final_df.to_csv("ushant_trajectories.csv",index=False)
final_df.to_pickle("ushant_trajectories.pkl")

print("Done preprocessing Ushant AIS data")