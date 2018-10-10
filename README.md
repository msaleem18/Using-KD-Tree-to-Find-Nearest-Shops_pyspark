# Work_Sample_Project

##Description
Project is to find the Nearest Neighbour to a bunch of location data. Code can be run in Spark (implemented using pyspark).

##Assumptions
- The distance between the points can be approximated using Haversine distance formula

- The best method to divide the POI is using K-D Tree, assumption is that a K-D Tree structure will be maintained before hand

- Assuming that POI1 is equal to POI2 as they have same coordinates. However, this may refer to a Multi-Unit building. Assuming there are no 'height' coordinates

- There is enough memory on the stack that K-D Tree recursion steps won't break the system

- The POI density distribution can be explained using Z-Score
