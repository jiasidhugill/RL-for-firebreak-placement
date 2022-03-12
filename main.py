# import libraries
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import subprocess
import shapefile as sf
from tensorflow import keras
import tifffile as tiff
import time

# global constants
barren = 99
fuel_layer = 3


# set up the environment class
class FlamMap:
   def __init__(self, state_path):
       self.state_path = state_path
       self.benchmark = self.calc_benchmark()

   def end_of_path(self, input_path):
       name = input_path[input_path.rindex("\\") + 1:]
       return name

   def random_ignition(self):
       ign_path = r"C:\Users\PHSTech\Desktop\Current_Run\curr_ignitions.shp"
       writer = sf.Writer(ign_path)
       writer.field("x", "y")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point1")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point2")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point3")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point4")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point5")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point6")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point7")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point8")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point9")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point10")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point11")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point12")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point13")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point14")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point15")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point16")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point17")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point18")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point19")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point20")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point21")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point22")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point23")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point24")
       writer.point(random.randint(-9000, 9000), random.randint(-9000, 9000))
       writer.record("point25")
       writer.close()

   def time_now(self):
       return datetime.now()

   def simulate(self, input_path):
       # clear the Current_Run folder
       os.system(r"del /q C:\Users\PHSTech\Desktop\Current_Run\Landscape\*")
       os.system(r"del /q C:\Users\PHSTech\Desktop\Current_Run\Outputs\*")
       os.system(r"copy C:\Users\PHSTech\Desktop\Datasets\{} C:\Users\PHSTech\Desktop\Current_Run\Landscape".format(
           input_path))
       # renames the file (to change the path to what FlamMap expects)
       os.system(r"rename C:\Users\PHSTech\Desktop\Current_Run\Landscape\{} curr_lcp.tif".format(input_path))
       # open FlamMap 6 with save_output macro
       timeout_seconds = 50
       try:
           save_cmd = r"C:\Users\PHSTech\Desktop\MacroRecorder.lnk " \
             r"-play=C:\Users\PHSTech\Desktop\MacroClips\save_output_fast.mrf"
           # print(f"{self.time_now()} Calling macro")
           subprocess.check_output(save_cmd, shell=True, timeout=timeout_seconds)
           # print(f"{self.time_now()} Called macro")
       except subprocess.TimeoutExpired:
           print(f"{self.time_now()} Waited {timeout_seconds} seconds for macro to complete")
       except Exception as e:
           print(f"{self.time_now()} Unhandled Exception: {repr(e)}")
       print(f"{self.time_now()} Waiting another {timeout_seconds} seconds")
       time.sleep(timeout_seconds)
       print(f"{self.time_now()} macro finished")
       # copy into correct directory, delete once copied
       os.system(
           r"copy C:\Users\PHSTech\Desktop\Current_Run\output.tif "
           r"C:\Users\PHSTech\Desktop\Current_Run\Outputs\output.tif")
       os.system(r"del /f C:\Users\PHSTech\Desktop\Current_Run\output.tif")
       return tiff.imread(r"C:\Users\PHSTech\Desktop\Current_Run\Outputs\output.tif")

   def calc_benchmark(self):
       # calculates the same level of burn with no firebreaks (same ignitions)
       self.simulate("curr_state.tif")
       output_path = r"C:\Users\PHSTech\Desktop\Current_Run\Outputs\output.tif"
       output = tiff.imread(output_path)
       count = 0
       for i in range(output.shape[0]):
           for j in range(output.shape[1]):
               if output[i, j] != -9999:
                   count += 1
       self.benchmark = count
       return self.benchmark

   def reward(self):
       output_path = r"C:\Users\PHSTech\Desktop\Current_Run\Outputs\output.tif"
       output = tiff.imread(output_path)
       count = 0
       for i in range(output.shape[0]):
           for j in range(output.shape[1]):
               if output[i, j] != -9999:
                   count += 1
       return (self.benchmark - count) / self.benchmark


def rotate(degrees, points):  # (UNFINISHED)
   # create a rotation matrix
   # multiply points by rotation matrix
   return degrees, points


def random_action():
   center = (random.randint(-7000, 7000), random.randint(-7000, 7000))
   if random.random() >= 0.5:
       degrees = -1
   else:
       degrees = 1
   # degrees = random.randint(0, 180)  # (UNFINISHED) eventually, be able to rotate with a rotation matrix
   return center, degrees


def take_action(center, degrees, act_num):
   path = r"C:\Users\PHSTech\Desktop\Current_Run\Firebreaks\curr_firebreaks_{}.shp".format(act_num)
   writer = sf.Writer(path)
   writer.field("name", "C")
   points = []
   if degrees == -1:
       point1 = [center[0] + 1000, center[1] + 50]
       point2 = [center[0] - 1000, center[1] + 50]
       point3 = [center[0] - 1000, center[1] - 50]
       point4 = [center[0] + 1000, center[1] - 50]
   else:
       point1 = [center[0] + 50, center[1] + 1000]
       point2 = [center[0] - 50, center[1] + 1000]
       point3 = [center[0] - 50, center[1] - 1000]
       point4 = [center[0] + 50, center[1] - 1000]
   points.append([point1, point2, point3, point4])
   # rotate polygon (UNFINISHED)
   # rotate(degrees, points)
   writer.poly(points)
   writer.record("polygon")
   writer.close()
   return points


def clear_state():
   path = r"C:\Users\PHSTech\Desktop\Current_Run\firebreaks\*"
   os.system(r"del /q {}".format(path))


# # DQN starts here (as of rn: only one landscape used. Will cycle through landscapes once this part is coded)
path = r"C:\Users\PHSTech\Desktop\Datasets\curr_state.tif"
env = FlamMap(path)
actions_per_state = 10  # each action is one firebreak placed down
exp_replay_frames = 100  # 500

# Step 1: Experience Replay
for state in range(exp_replay_frames):
   os.system(r"copy C:\Users\PHSTech\Desktop\Datasets\TestData.tif C:\Users\PHSTech\Desktop\Datasets\curr_state.tif")
   q_table = pd.DataFrame()
   clear_state()
   env.random_ignition()  # for now each state is a random ignition; later, it'll be different landscapes, too
   # env.calc_benchmark()
   for action in range(actions_per_state):
       print(f"{datetime.now()} Doing State: {state}, action: {action}")
       center, degrees = random_action()
       take_action(center, degrees, action)
       q_table.loc[action, "x"] = center[0]
       q_table.loc[action, "y"] = center[1]
       q_table.loc[action, "degrees"] = degrees
   env.simulate(r"curr_state.tif")
   q_table["reward"] = env.reward()
   q_table.to_csv(r"C:\Users\PHSTech\Desktop\Q-tables\Experience_Replay\q_table_state_{}.csv".format(state))
   print(f"Completed State {state}")

# Step 2: Build Neural Network

x_exp = []
Y_exp = []

for i in range(exp_replay_frames):
   temp_df = pd.read_csv(r"C:\Users\PHSTech\Desktop\Q-tables\Experience_Replay\q_table_state_{}.csv".format(i)) \
       .drop(columns="Unnamed: 0")
   temp_df = np.asarray(temp_df)
   x_exp.append(temp_df[0, 3])
   Y_exp.append(temp_df[:, :3].reshape(1, 30))

# print(max(x_exp))

x_exp = np.asarray(x_exp)
Y_exp = np.asarray(Y_exp)

inputs = keras.Input(shape=(1,))
layer1 = keras.layers.Dense(5, activation="relu")(inputs)
layer2 = keras.layers.Dense(10, activation="relu")(layer1)
layer3 = keras.layers.Dense(20, activation="relu")(layer2)
outputs = keras.layers.Dense(30, activation="relu")(layer3)

q_model = keras.Model(inputs=inputs, outputs=outputs)

q_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse",
               metrics=["accuracy"])

q_model.fit(x_exp, Y_exp)

# Step 3: Calculate Target Reward Value (UNFINISHED)
target_reward = 0.5
# sticking to definition of success according to our proposal; however, you need to check max of
# random to accurately decide this

# Step 4: Explore, Exploit, and Update the Q-Network (until solved) (UNFINISHED)

max_frames = 1000
frame_count = 0
max_success_frames = 10
success_count = 0
epsilon = 0.9
reward_df = pd.DataFrame(columns=["frame", "reward"])
accuracy_df = pd.DataFrame(columns=["frame", "mse"])
x = []
Y = []
target_now = 0.2
train_start = 503  # DON'T FORGET to also change the names of the csvs

for i in range(train_start):
   temp_df = pd.read_csv(r"C:\Users\PHSTech\Desktop\Q-tables\Training_Loop\q_table_state_{}".format(i)) \
       .drop(columns="Unnamed: 0")
   temp_df = np.asarray(temp_df)
   x.append(temp_df[0, 3])
   Y.append(temp_df[:, :3].reshape(1, 30))

while True:
   clear_state()
   env.random_ignition()
   # env.calc_benchmark()
   if random.random() > epsilon:
       q_table = pd.DataFrame()
       for action in range(10):
           center, degrees = random_action()
           take_action(center, degrees, action)
           q_table.loc[action, "x"] = center[0]
           q_table.loc[action, "y"] = center[1]
           q_table.loc[action, "degrees"] = degrees
   else:
       q_table = q_model.predict([target_now])
       q_table = np.array(q_table).reshape(10, 3)
       for i in range(10):
           if q_table[i, 2] >= 0:
               q_table[i, 2] = 1
           else:
               q_table[i, 2] = -1
       for i in range(10):
           take_action([q_table[i, 0], q_table[i, 1]], q_table[i, 2], i)
   env.simulate(r"curr_state.tif")
   reward = env.reward()
   if reward >= target_now and target_reward > target_now:
       target_now += 0.1
       print(f"Frame {frame_count}: target reward raised to {target_now}")
   q_table = pd.DataFrame(q_table)
   q_table.loc[:, 3] = reward
   pd.DataFrame(q_table).to_csv(
       r"C:\Users\PHSTech\Desktop\Q-tables\Training_Loop\q_table_state_{}".format(frame_count + train_start))
   x.append(np.array(q_table.iloc[0, 3]))
   Y.append((np.array(q_table)[:, :3]).reshape(1, 30))
   mse = keras.metrics.mean_squared_error([np.array(q_table.iloc[:, :3]).reshape(1, 30)], [q_model.predict([reward])])
   if frame_count % 4 == 0:
       q_model.fit(np.array(x), np.array(Y))
       q_model.save("q_model")
   if reward > target_reward:
       success_count += 1
   frame_count += 1
   reward_df.loc[frame_count, "frame"] = frame_count
   reward_df.loc[frame_count, "reward"] = reward
   accuracy_df.loc[frame_count, "frame"] = frame_count
   accuracy_df.loc[frame_count, "mse"] = np.array(mse)[0][0]
   accuracy_df.to_csv("accuracy_4.csv")
   reward_df.to_csv("reward_4.csv")
   if (max_frames <= frame_count) or (max_success_frames <= success_count):
       break

if success_count >= max_success_frames:
   print("success")
else:
   print("failure")
