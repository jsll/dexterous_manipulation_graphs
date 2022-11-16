# Dexterous Manipulation Graphs

This is a Python implementation of [Dexterous Manipulation Graphs](https://arxiv.org/pdf/1803.00346.pdf)

# License

The source code is released under [MIT License](LICENSE).

# Installation

We recommend Conda for installation. First, create a Conda environment with the following:

```
conda create --name dmg --file dmg-env.txt
```

After installing the Conda environment, test the installation as follows:
```
conda activate dmg 
python test.py
```

All the tests should pass. If not, then something went wrong with your installation.

# Test

To try the code with a box, run the code below and then follow the steps:

```
python main.py -o box
```

or if you want to try all objects, which can take a prohibitively long time for highly tessellated objects, run

```
python main.py
```

For more information about the command line parameters, please check [main.py](./main.py) or write in the terminal python main.py --help.

Once the code runs, you need to do the following:

1. Wait until the first visualization shows up (remember this can take a long time for highly tessellated objects). 
2. Choose a start and goal node by clicking on the respective node in the figure. Note that the start and goal nodes cannot be the same. 
3. Choose the gripper angle for the start and goal node by clicking on the respective arrow in the figure. 

Once all the data is given, the path is calculated and visualized on the screen. 

An example of the above test procedure is shown below: 

![](gif/DMG.gif)

# Good to know 

There are infinitely many reference systems for each face or angular component. This is because the normal to each face or super voxel only constrains one degree of freedom of the reference coordinate system. I solved this issue by randomly choosing one axis to be perpendicular to the normal axis and then calculating the last coordinate axis as the cross-product between the normal and randomly chosen axis.

# Known Issues

1. I have not implemented the super voxel clustering as Python-PCL currently cannot run with Python 3.10.
2. Collision checking is performed by checking if the points sampled on the mesh are inside the object. Although reasonable, this solution is sensitive to the number of points sampled on the gripper: too many points make it slow, while too few points make collision checking unreliable. Another potentially more accurate option is to use [trimesh collision manager](https://trimsh.org/trimesh.collision.html).
3. The visualization of angles when picking the start and goal angle should only visualize collision-free angles.

# Citation

If you find this work useful for your research, please consider citing my implementation:

```
@article{dextrous-manipulation-graphs,
      Author = {Jens Lundell},
      Title = {Dexterous Manipulation Graphs},
      Journal = {<https://github.com/jsll/dextrous-manipulation-graphs},>
      Year = {2022}
}
```
