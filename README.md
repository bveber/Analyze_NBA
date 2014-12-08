Analyze_NBA
===========

This program has been tested with Python 3.4 on Windows 8 and Ubuntu 14

From the command line the program can be run by issuing the command:
python NBAMakeTrees.py

or open an instance of python3.4, then enter the commands:
import NBAMakeTrees
NBAMakeTrees.main()

It will then make and save tree graphics for each teams top scoring player.
These graphics will show the most relevant partitions in the data.  The convention for reading the 
tree graphic is as follow:
-The top line shows the condition, if true -> follow the box to the left, else -> right
-The gini value shows the gini impurity for each partition
-The samples value shows how many shots were taken in that particular situation
-The value array shows the number of occurences of each class.  For example if this line
 reads: value=[21 8], then that means 29 (21+8) shots were taken in that particular
 situation, with 21 made shots, and 8 missed shots
 
This program could be extended to account for all current players, and could include
data from 2013-2014 season if available.  This concept could also be applied to passing,
rebounding, defense, etc.

