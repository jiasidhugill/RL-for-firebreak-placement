# RL-for-firebreak-placement
The success of a firebreak is heavily dependent on its location. This project uses reinforcement learning to determine the best placement for a firebreak in a given landscape. (Submitted to the Synopsys Science Fair 2022.)

# Abstract
Fires help to clear dead vegetation for new growth and are a natural part of many
ecosystems. Currently, fires rage out of control, destroying miles and miles of land
due to higher temperatures, drier vegetation, and a lack of precipitation. The success
of firebreaks, areas of land without vegetation to prevent fires from spreading and
minimize fire risk, is heavily dependent on their location. Our project goal was to use a
reinforcement learning model to optimize the placement of firebreaks to minimize fire
spread. In order to do this, we used the FlamMap 6 simulator, a fire simulator
maintained by the U.S. Forest Service and commonly used to study fire spread and
behavior. We used the Python library keras to create a deep learning model that
learned from previous attempts to place firebreaks to minimize the total area burned.
Our model was able to increase reward and accuracy as it continued to run. We were
able to decrease the area burned by about 39%. We found that firebreak placement
can decrease firebreak spread, but not by more than a certain benchmark,
highlighting the lack of efficacy of firebreaks in intense fires. Further research would
include training on a larger variety of landscapes—this project focused on a small
region of California—and improving the efficiency of the model to decrease the time
taken for it to run. Currently, the model must run for several hours to train. Our
findings have the ability to inform fire management and improve fire management
strategies.

# Acknowledgements
I did this project with my partner, Danica Kubota. If you're reading this—hi! If you're not Danica...hello anyway!
