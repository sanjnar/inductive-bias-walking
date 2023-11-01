# Creation of new (or tweaked) environments in mujoco on MIT supercloud

## Create a new xml model

The first step is to create an update xml file by copying the one you want to reproduce. Then, depending on what you need, do any of the following steps 

- remove some actuators (see the list of actuators at the end of the file)
- add some sites to specific body parts (requires to add them within the mujoco hierarchical structure)

Mujoco documentation can be found [here] (https://mujoco.readthedocs.io/en/latest/overview.html)

## Create the python file to call the new environment

Similar to the first step, copy the python file in the folder gymnasium/envs/mujoco that you want to branch from. 
Search for the place where it calls the xml file and change it such that it fetches your newly created file. 
❗ Make sure to also update the class name to reflect your new environment

## Register your new environment 

In the file *gymnasium/envs/register.py*, find the entry corresponding to the environment you are branching from, copy it and edit it to reflect your newly created environment. You should be able to specify the path towards the newly created environment (i.e. the file you updates in the previous section). 
