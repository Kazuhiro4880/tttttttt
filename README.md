Mining Social and Geographic Datasets
-----------------------------------

GEOG0051 Computer Lab 4: Gravity Models
-------------------------------

Note: Notebook might contain scripts and instructions adapted from GEOG0115, GEOG0051. 
Contributors: Stephen Law, Mateo Neira, Nikki Tanu, Thomas Keel, Gong Jie, Jason Tang and Demin Hu.

Overview of Content in this Jupyter Notebook
===============
> ### Lab Notebook 4: Commuting Flow Analysis and Models

> ### Lab Exercise 4: Gravity Model of commuters around Manchester & Birmingham

Lab Notebook 4.1: Commuting Flow Models
-------------------------------

In this notebook, we will focus on visualising and modelling commuting flow datasets of the UK using `NetworkX` and `GeoPandas`. To recap, NetworkX was introduced last week and is a package used for complex network analysis. GeoPandas is the counterpart of the Pandas package, used to handle data frames in Python, with the difference being that GeoPandas supports the usage of geographical data. 

We will then estimate a gravity model using PySal's ```spint``` package gravity model function and alternatively ```Statsmodel```, a standard python package for statistical inference. If you need to install ```Spint``` from PySal or ```Statsmodel```, you can use the `conda` command or the `pip` command. Please note that the "!"  is used to perform an installation command inside the notebook. 

```python3
!pip install spint
!pip install statsmodels
```

Linked is a useful reference on gravity models that you can optionally read for more information:
* [Oshan, T. (2016) A primer for working with the Spatial Interaction modeling (SpInt) module in the python spatial analysis library (PySAL)](https://openjournals.wu.ac.at/region/paper_175/175.html)

In this first chunk of code, we import the Python packages that we need to handle data frames, make calculations and visualise the results later on. 

What we first recommend that you do, is to explore the two datasets first, after having loaded them into the Jupyter notebook, to understand what kind of data you are dealing with. For a start, you could use the method (built-in function) `head()`, which shows you a preview of the first `X` rows (observations) in the data frame, where X is the argument (numeric) you put between the parantheses.


```python
# let's first import the standard python modules
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
```


```python
# let's read the UK flows where you have the origin ('residence'), the destination ('workplace') and its attributes.
df=pd.read_csv('datasets/UK_Flows.csv',index_col=0)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Residence</th>
      <th>Workplace</th>
      <th>Distance</th>
      <th>Commuters</th>
      <th>O_Pop</th>
      <th>O_Workplace</th>
      <th>D_Pop</th>
      <th>D_Workplace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aberdeenshire</td>
      <td>Aberdeen City</td>
      <td>29282</td>
      <td>41224</td>
      <td>257740</td>
      <td>101816</td>
      <td>227130</td>
      <td>98610</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adur</td>
      <td>Aberdeen City</td>
      <td>712858</td>
      <td>2</td>
      <td>62505</td>
      <td>23437</td>
      <td>227130</td>
      <td>98610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Allerdale</td>
      <td>Aberdeen City</td>
      <td>280203</td>
      <td>11</td>
      <td>96208</td>
      <td>37322</td>
      <td>227130</td>
      <td>98610</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Amber Valley</td>
      <td>Aberdeen City</td>
      <td>461424</td>
      <td>0</td>
      <td>123498</td>
      <td>49535</td>
      <td>227130</td>
      <td>98610</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angus</td>
      <td>Aberdeen City</td>
      <td>66028</td>
      <td>1992</td>
      <td>116240</td>
      <td>42639</td>
      <td>227130</td>
      <td>98610</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let's read in the place location coordinates where we will just take a subset of Local Authorities in Greater Birmingham
df2=pd.read_csv('datasets/UK_Birmingham_pts.csv')
df2=df2[['Name','X','Y']]
df2.head()
#len(df2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tewkesbury</td>
      <td>394086.0613</td>
      <td>227461.1678</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Erewash</td>
      <td>444007.0273</td>
      <td>337765.2693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nuneaton and Bedworth</td>
      <td>435676.4351</td>
      <td>289376.2917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tamworth</td>
      <td>421903.9782</td>
      <td>303234.2767</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Leicester</td>
      <td>459074.5879</td>
      <td>304818.6533</td>
    </tr>
  </tbody>
</table>
</div>




```python
gpd.__version__
```




    '0.14.2'



Why do you gather each of the two data frames that we loaded above are important to the subsequent analysis on commuting patterns we will delve into?

.
.
.

* `UK_Flows` about the extent of commuter flows between a pair of locations (one residence and one workplace location), and 
* `UK_Birmingham_pts` tells us the geographical locations (in coordinates) of the said locations.

### Some preliminary visualisation

As we learnt in the last Lab notebook, we will plot the place locations in `GeoPandas`. 


```python
gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.X, df2.Y))

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>X</th>
      <th>Y</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tewkesbury</td>
      <td>394086.0613</td>
      <td>227461.1678</td>
      <td>POINT (394086.061 227461.168)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Erewash</td>
      <td>444007.0273</td>
      <td>337765.2693</td>
      <td>POINT (444007.027 337765.269)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nuneaton and Bedworth</td>
      <td>435676.4351</td>
      <td>289376.2917</td>
      <td>POINT (435676.435 289376.292)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tamworth</td>
      <td>421903.9782</td>
      <td>303234.2767</td>
      <td>POINT (421903.978 303234.277)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Leicester</td>
      <td>459074.5879</td>
      <td>304818.6533</td>
      <td>POINT (459074.588 304818.653)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>North Warwickshire</td>
      <td>425319.3151</td>
      <td>294468.3746</td>
      <td>POINT (425319.315 294468.375)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bromsgrove</td>
      <td>399006.3967</td>
      <td>273744.0193</td>
      <td>POINT (399006.397 273744.019)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rugby</td>
      <td>445814.9430</td>
      <td>276471.0934</td>
      <td>POINT (445814.943 276471.093)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stratford-on-Avon</td>
      <td>425886.6406</td>
      <td>253459.3536</td>
      <td>POINT (425886.641 253459.354)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hinckley and Bosworth</td>
      <td>441325.8916</td>
      <td>302252.9667</td>
      <td>POINT (441325.892 302252.967)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Telford and Wrekin</td>
      <td>367088.9036</td>
      <td>314840.4595</td>
      <td>POINT (367088.904 314840.459)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>South Staffordshire</td>
      <td>388128.5891</td>
      <td>303085.4165</td>
      <td>POINT (388128.589 303085.416)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Warwick</td>
      <td>428413.2966</td>
      <td>268068.8134</td>
      <td>POINT (428413.297 268068.813)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stafford</td>
      <td>388210.5431</td>
      <td>327864.7562</td>
      <td>POINT (388210.543 327864.756)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Walsall</td>
      <td>402375.5701</td>
      <td>300163.9784</td>
      <td>POINT (402375.570 300163.978)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Shropshire</td>
      <td>350418.6374</td>
      <td>304381.0877</td>
      <td>POINT (350418.637 304381.088)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Harborough</td>
      <td>467369.5169</td>
      <td>293956.8063</td>
      <td>POINT (467369.517 293956.806)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Wolverhampton</td>
      <td>391899.8422</td>
      <td>299050.6823</td>
      <td>POINT (391899.842 299050.682)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Daventry</td>
      <td>464975.5414</td>
      <td>269079.5732</td>
      <td>POINT (464975.541 269079.573)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Newcastle-under-Lyme</td>
      <td>378693.4296</td>
      <td>344024.7757</td>
      <td>POINT (378693.430 344024.776)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Coventry</td>
      <td>432759.0667</td>
      <td>279983.0005</td>
      <td>POINT (432759.067 279983.001)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Charnwood</td>
      <td>457748.9052</td>
      <td>315827.3534</td>
      <td>POINT (457748.905 315827.353)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Sandwell</td>
      <td>399486.4660</td>
      <td>290839.0240</td>
      <td>POINT (399486.466 290839.024)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Blaby</td>
      <td>452827.6533</td>
      <td>297360.7728</td>
      <td>POINT (452827.653 297360.773)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Worcester</td>
      <td>385652.6308</td>
      <td>255405.3014</td>
      <td>POINT (385652.631 255405.301)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>East Staffordshire</td>
      <td>412744.8054</td>
      <td>329548.3923</td>
      <td>POINT (412744.805 329548.392)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Stoke-on-Trent</td>
      <td>389059.9595</td>
      <td>346759.3219</td>
      <td>POINT (389059.959 346759.322)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>North West Leicestershire</td>
      <td>440247.9102</td>
      <td>317443.9832</td>
      <td>POINT (440247.910 317443.983)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Lichfield</td>
      <td>413283.8347</td>
      <td>310071.1585</td>
      <td>POINT (413283.835 310071.159)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Derby</td>
      <td>435849.8062</td>
      <td>335145.3644</td>
      <td>POINT (435849.806 335145.364)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Redditch</td>
      <td>403381.0320</td>
      <td>264905.8074</td>
      <td>POINT (403381.032 264905.807)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Dudley</td>
      <td>392713.4046</td>
      <td>287218.2806</td>
      <td>POINT (392713.405 287218.281)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Wychavon</td>
      <td>396620.9873</td>
      <td>251769.8821</td>
      <td>POINT (396620.987 251769.882)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Solihull</td>
      <td>418659.7508</td>
      <td>279492.3303</td>
      <td>POINT (418659.751 279492.330)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Wyre Forest</td>
      <td>381400.3207</td>
      <td>275762.2966</td>
      <td>POINT (381400.321 275762.297)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>South Derbyshire</td>
      <td>429772.3196</td>
      <td>326031.8441</td>
      <td>POINT (429772.320 326031.844)</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oadby and Wigston</td>
      <td>461516.9614</td>
      <td>299198.5539</td>
      <td>POINT (461516.961 299198.554)</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Malvern Hills</td>
      <td>376531.2530</td>
      <td>253890.3459</td>
      <td>POINT (376531.253 253890.346)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Cannock Chase</td>
      <td>401432.6396</td>
      <td>312746.9342</td>
      <td>POINT (401432.640 312746.934)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Birmingham</td>
      <td>408598.1153</td>
      <td>287856.7593</td>
      <td>POINT (408598.115 287856.759)</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creates a GeoDataFrame from the data frame df2, taking the rows 'X' and 'Y' as the geographical variables
gdf2=gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.X, df2.Y))
#sets the CRS, which is the coordinate reference system of the map
gdf2=gdf2.set_crs(epsg=27700)
gdf2 = gdf2.to_crs(epsg=3857) # setting crs to 3857
#creates an empty grid to plot the map onto
fig, ax = plt.subplots(figsize=(12,20))
#and fits the points from the gdf2 geodataframe onto this grid
gdf2.plot(ax=ax,figsize=(10,10),color='black')


```




    <Axes: >




    
![png](GEOG0051_Lab4%20%28Questions%29_files/GEOG0051_Lab4%20%28Questions%29_11_1.png)
    



```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>X</th>
      <th>Y</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tewkesbury</td>
      <td>394086.0613</td>
      <td>227461.1678</td>
      <td>POINT (394086.061 227461.168)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Erewash</td>
      <td>444007.0273</td>
      <td>337765.2693</td>
      <td>POINT (444007.027 337765.269)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nuneaton and Bedworth</td>
      <td>435676.4351</td>
      <td>289376.2917</td>
      <td>POINT (435676.435 289376.292)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tamworth</td>
      <td>421903.9782</td>
      <td>303234.2767</td>
      <td>POINT (421903.978 303234.277)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Leicester</td>
      <td>459074.5879</td>
      <td>304818.6533</td>
      <td>POINT (459074.588 304818.653)</td>
    </tr>
  </tbody>
</table>
</div>



What do you think? The map of coordinates and points corresponding to coordinate pairs on its own isn't too useful, is it?

It seems better that we layer the points with a basemap underneath using `contextily`, to give context to the viewer about the locations in the map. 


```python
#creates an empty grid to plot the map onto
fig, ax = plt.subplots(figsize=(12,20))
#and fits the points from the gdf2 geodataframe onto this grid
gdf2.plot(ax=ax,figsize=(10,10),color='black')

import contextily as ctx
#adds a basemap to give the viewer visual context
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.axis('off') #removes the display of axes, so it looks more like a map and less like a chart
plt.show()
```


    
![png](GEOG0051_Lab4%20%28Questions%29_files/GEOG0051_Lab4%20%28Questions%29_14_0.png)
    


#### 🤨 TASK
can you visualise as an interactive map using the "Stamen Toner" basemap? (hint: `GeoDataFrame.explore()`)

*Replace the `???` with your answer...*


```python
gpd.GeoDataFrame.explore(df2)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_6164\165915763.py in ?()
    ----> 1 gpd.GeoDataFrame.explore(df2)
    

    ~\anaconda3\envs\envGEOG0051\Lib\site-packages\geopandas\geodataframe.py in ?(self, *args, **kwargs)
       2111     @doc(_explore)
       2112     def explore(self, *args, **kwargs):
    -> 2113         return _explore(self, *args, **kwargs)
    

    ~\anaconda3\envs\envGEOG0051\Lib\site-packages\geopandas\explore.py in ?(df, column, cmap, color, m, tiles, attr, tooltip, popup, highlight, categorical, legend, scheme, k, vmin, vmax, width, height, categories, classification_kwds, control_scale, marker_type, marker_kwds, style_kwds, highlight_kwds, missing_kwds, tooltip_kwds, popup_kwds, legend_kwds, map_kwds, **kwargs)
        310 
        311     gdf = df.copy()
        312 
        313     # convert LinearRing to LineString
    --> 314     rings_mask = df.geom_type == "LinearRing"
        315     if rings_mask.any():
        316         gdf.geometry[rings_mask] = gdf.geometry[rings_mask].apply(
        317             lambda g: LineString(g)
    

    ~\anaconda3\envs\envGEOG0051\Lib\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'geom_type'


### Visualise the commuting data as a network 

Now, going back to the initial two data frames that we had `df`, showing commuter patterns between each origin-destination pair, and `df2`, encompassing information about the x-, y-coordinates of each location), we `pd.merge()` the location attributes back to the edgelists for visualisation as a network graph. 


```python
#merges the the source and target location attributes
#firstly, for the places of residence
df2 = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.X, df2.Y))
df2.columns = ['Name','source_X','source_Y','source_geom']
df = pd.merge(left=df,right=df2,left_on='Residence',right_on='Name')

#and for the workplaces
df2.columns = ['Name','target_X','target_Y','target_geom']
df = pd.merge(left=df,right=df2,left_on='Workplace',right_on='Name')
df2.columns=['Name','X','Y','geometry']

#deletes unnecessary rows
del df['Name_x']
del df['Name_y']
```


```python
df.columns
```




    Index(['Residence', 'Workplace', 'Distance', 'Commuters', 'O_Pop',
           'O_Workplace', 'D_Pop', 'D_Workplace', 'source_X', 'source_Y',
           'source_geom', 'target_X', 'target_Y', 'target_geom'],
          dtype='object')




```python
df.head(n=5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Residence</th>
      <th>Workplace</th>
      <th>Distance</th>
      <th>Commuters</th>
      <th>O_Pop</th>
      <th>O_Workplace</th>
      <th>D_Pop</th>
      <th>D_Workplace</th>
      <th>source_X</th>
      <th>source_Y</th>
      <th>source_geom</th>
      <th>target_X</th>
      <th>target_Y</th>
      <th>target_geom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Birmingham</td>
      <td>Blaby</td>
      <td>45239</td>
      <td>213</td>
      <td>1092330</td>
      <td>357433</td>
      <td>95092</td>
      <td>39948</td>
      <td>408598.1153</td>
      <td>287856.7593</td>
      <td>POINT (408598.115 287856.759)</td>
      <td>452827.6533</td>
      <td>297360.7728</td>
      <td>POINT (452827.653 297360.773)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bromsgrove</td>
      <td>Blaby</td>
      <td>58774</td>
      <td>19</td>
      <td>94744</td>
      <td>37289</td>
      <td>95092</td>
      <td>39948</td>
      <td>399006.3967</td>
      <td>273744.0193</td>
      <td>POINT (399006.397 273744.019)</td>
      <td>452827.6533</td>
      <td>297360.7728</td>
      <td>POINT (452827.653 297360.773)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cannock Chase</td>
      <td>Blaby</td>
      <td>53649</td>
      <td>13</td>
      <td>98119</td>
      <td>39371</td>
      <td>95092</td>
      <td>39948</td>
      <td>401432.6396</td>
      <td>312746.9342</td>
      <td>POINT (401432.640 312746.934)</td>
      <td>452827.6533</td>
      <td>297360.7728</td>
      <td>POINT (452827.653 297360.773)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charnwood</td>
      <td>Blaby</td>
      <td>19111</td>
      <td>3452</td>
      <td>170645</td>
      <td>66682</td>
      <td>95092</td>
      <td>39948</td>
      <td>457748.9052</td>
      <td>315827.3534</td>
      <td>POINT (457748.905 315827.353)</td>
      <td>452827.6533</td>
      <td>297360.7728</td>
      <td>POINT (452827.653 297360.773)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Coventry</td>
      <td>Blaby</td>
      <td>26546</td>
      <td>265</td>
      <td>329810</td>
      <td>118367</td>
      <td>95092</td>
      <td>39948</td>
      <td>432759.0667</td>
      <td>279983.0005</td>
      <td>POINT (432759.067 279983.001)</td>
      <td>452827.6533</td>
      <td>297360.7728</td>
      <td>POINT (452827.653 297360.773)</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.head(n=5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>X</th>
      <th>Y</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tewkesbury</td>
      <td>394086.0613</td>
      <td>227461.1678</td>
      <td>POINT (394086.061 227461.168)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Erewash</td>
      <td>444007.0273</td>
      <td>337765.2693</td>
      <td>POINT (444007.027 337765.269)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nuneaton and Bedworth</td>
      <td>435676.4351</td>
      <td>289376.2917</td>
      <td>POINT (435676.435 289376.292)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tamworth</td>
      <td>421903.9782</td>
      <td>303234.2767</td>
      <td>POINT (421903.978 303234.277)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Leicester</td>
      <td>459074.5879</td>
      <td>304818.6533</td>
      <td>POINT (459074.588 304818.653)</td>
    </tr>
  </tbody>
</table>
</div>



The output data frame from the above operations, ```df```, will have, in every row, the commuter flows between one pair of residence and workplace locations, **together with** their respective locations in coordinate pairs. Following this, we will use `NetworkX`, which we learnt last week in conjunction with `OSMnx`, - only this time, we will be visualising the commuting flow network. 


```python
import networkx as nx

#first, compiles a network graph from the merged data frame
G = nx.from_pandas_edgelist(df, 'Residence', 'Workplace', ['Distance', 'Commuters', 'O_Pop',
       'O_Workplace', 'D_Pop', 'D_Workplace', 'source_X', 'source_Y',
       'target_X', 'target_Y'])

#calculates the weighted degree for the commuting graph based on the intensity of commuter flows between each pair of points
deg = nx.degree(G, weight='Commuters')

#inscribes these node attributes back to the network graph
nx.set_node_attributes(G,dict(deg),'deg')

#and colours the node attributes differentially
n_color=[(i[1]['deg']) for i in G.nodes(data=True)]
```

Once again, this is a reminder to take a pause and look at individual lines of code, and individual arguments within functions that are in the code chunks - what does each of the arguments change? What inputs do the functions take and what outputs do they make? As a generally trustworthy way to break down your code to understand it in steps, try printing/calling the intermediate outputs that we are creating whenever we use the `=` (attribute) function. For instance, try calling/printing `n_color` and see the object type it takes.


```python
#finally, visualises the commuting graph by its node weighted degree
fig,ax = plt.subplots(figsize=(20,14))

# creates a dictionary object, `pos`, which displays information of Place names linked to their x- and y- coordinates
pos = {i[1]['Name']:(i[1]['X'],i[1]['Y']) for i in df2.iterrows()}


# plots the network graph with customisation options
nx.draw_networkx(G,
                 pos,
                 node_size=200,
                 #node_color='black',
                 node_color=n_color,
                 edge_color='grey',
                 with_labels=True,
                 label=True,
                 font_size=12,
                 ax=ax,
                 width=0.1)

#optional: to add colorbar
net=nx.draw_networkx_nodes(G,
                 pos,
                 node_size=0,
                 node_color=n_color,
                 ax=ax)
plt.colorbar(net)


plt.show()


```


    
![png](GEOG0051_Lab4%20%28Questions%29_files/GEOG0051_Lab4%20%28Questions%29_25_0.png)
    


What do you see in the plot above? Which city has the highest commuting flow degree? 


#### 🤨 TASK
can you visualise the same commuting network map including a contextilly background as practice? 

*Replace the `???` with your answer...*

```python
# lets put a basemap as well.
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.axis('off') #removes the display of axes, so it looks more like a map and less like a chart
plt.show()
```


```python
#finally, visualises the commuting graph by its node weighted degree
fig,ax = plt.subplots(figsize=(20,14))

# creates a dictionary object, `pos`, which displays information of Place names linked to their x- and y- coordinates
pos = {i[1]['Name']:(i[1]['X'],i[1]['Y']) for i in df2.iterrows()}


# plots the network graph with customisation options
nx.draw_networkx(G,
                 pos,
                 node_size=200,
                 #node_color='black',
                 node_color=n_color,
                 edge_color='grey',
                 with_labels=True,
                 label=True,
                 font_size=12,
                 ax=ax,
                 width=0.1)

#optional: to add colorbar
net=nx.draw_networkx_nodes(G,
                 pos,
                 node_size=0,
                 node_color=n_color,
                 ax=ax)
plt.colorbar(net)

import contextily as ctx
# lets put a basemap as well.
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.axis('off') #removes the display of axes, so it looks more like a map and less like a chart
plt.show()
```


    
![png](GEOG0051_Lab4%20%28Questions%29_files/GEOG0051_Lab4%20%28Questions%29_27_0.png)
    


### A Very Brief Introduction to PySal

![Pysal Logo](https://pysal.org/pysal1.png)

**PySAL** stands for **Python spatial data analysis library** which is an open-source project designed to support geographic data science in Python. The initiative was founded as a collaboration between Serge Rey and Luc Anselin in 2005. The first formal release of PySAL was July 2010 and the project has continued on a six-month release cycle since then. In 2018, PySAL was restructured as a meta-package that brings together a family of packages for spatial analysis. 

It contains the following libraries. 

* `libpysal` for core spatial data structures
* `esda`, `giddy`, `inequality`, `momepy`, `pointspats`, `segregation`, `sphaghetti` for exploratory spatial analysis
* `access`, `mgwr`, `spglm`, `spint`, `spopt`, `spreg`, `spvcm`, `tobler` for spatial modelling
* `legendgram`, `mapclassify` and `splot` for visualisation

We will only experiment with ```spint``` in this week's notebook, but you are suggested to experiment with the other libraries in Pysal. These libraries correspond to similar functions you learnt in principles of spatial analysis GEOG0114 for Python. For example [`mgwr`](https://mgwr.readthedocs.io/en/latest/) implements geographic weighted regression in python. Try replicating some of these analysis in python as a learning outcome. 

**If you are interested to read more about the package, here are some links:**

* Rey, S., Arribas-Bel, D., & Wolf, L. J. (2023). Geographic data science with python. CRC Press. [Online version of the textbook](https://geographicdata.science/book/intro.html)

* [official website](https://pysal.org/)

### Gravity model building with SpInt in PySal

Spatial interaction models or Gravity models are often estimated using a **Generalised linear model** (GLM) either as a Linear regression or as a Poisson regression. The latter is often used for modelling count data [(Oshan 2016)](https://openjournals.wu.ac.at/region/paper_175/175.html).

In this week's exercise, we will be using a Poisson regression to estimate the commuter flows between two regions in the UK as a function of its origin population, destination population and the distance that seperates these pairs of origin-destination. The Base Gravity model estimated using a poisson regression has the form;

$ln T_{ij}=k+\alpha ln O_i + \mu ln D_j +\beta d_{ij}+\epsilon$ 

$T_{ij}$ is the commuting flows between $i$ and $j$, 

$O_i$ is the origin population, 

$D_j$ is the desitnation population, 

$d_{ij}$ is the distance between $i$ and $j$.

equation (1)

With that, we will estimate the gravity model using spint in PySal. 

```python
class spint.gravity.Gravity(flows, o_vars, d_vars, cost, cost_func, constant=True, framework='GLM', SF=None, CD=None, Lag=None, Quasi=False)
```

This part of the analysis had been adapted from this [PySal tutorial](https://pysal.org/notebooks/model/spint/Example_NYCBikes_AllFeatures.html).

[Oshan, T. (2016) A primer for working with the Spatial Interaction modeling (SpInt) module in the python spatial analysis library (PySAL)](https://openjournals.wu.ac.at/region/paper_175/175.html)

### Fitting and visualising the outputs of the linear regression model


```python
#let's imports the gravity library in SpInt
from spint.gravity import Gravity
```


```python
df=df.reset_index()
```


```python
df.head()
```

As the model requires as an input **individual arrays in column form**, we will need to reshape the Pandas data frames from earlier, following which we then would be able to run the standard gravity model easily. 


```python
#reshapes the input data to fit the requirements of the SpInt model
flows=df['Commuters'].values.reshape((-1,1))
o_vars=df['O_Pop'].values.reshape((-1,1))
d_vars=df['D_Pop'].values.reshape((-1,1))
cost=df['Distance'].values.reshape((-1,1))
o=df['Residence'].values.reshape((-1,1))
d=df['Workplace'].values.reshape((-1,1))
```

#### 🤨 TASK
can you plot the histogram for `flows`, `o_vars` and `d_vars`?

What do you notice?

What does the highly skewed distribution suggest? 

*Replace the `???` with your answer...*



```python
???
```

now let's run the standard `gravity` model in `spint`. The only argument to note here is `cost_func='exp'` which basically means we do not log transform the distance function as can be seen in equation1. The functional form for the distance function is a well studied topic in the literature. For more information, you can read more about it from the primer written by [Oshan 2016](https://openjournals.wu.ac.at/region/paper_175/175.html). 


```python
# now let's run the standard gravity model
pysal_model = Gravity(flows, o_vars, d_vars, cost, cost_func='exp') 
```

**Goodness of fit**

We can then print out the **AIC value** and **Pseudo r2**. Both are goodness of fit measure that gives an indication of how good of a fit the GLM we estimated was (the lower the AIC value, the better, the higher the pseudo-r2 the better). That being said, AIC values is a comparative measure but are less useful when being read as a single value. While pseudo r2 is a similar measure to r2 in linear regression that is bounded b. 



```python
# notice you do not need to log O or log D. this is a much quicker way to fit a gravity model.
print (f'number of params: {len(pysal_model.params)}')
print (f'model aic: {pysal_model.AIC}')
print (f'pseudo r2: {pysal_model.pseudoR2}')
```

#### 🤨 TASK
try testing a different functional form for distance here `cost_func=pow`? What do you notice? has AIC or pseudo r2 improve or gotten worst? Which functional form has a better fit?

*Replace the `???` with your answer...*



```python
???
```

let's first estimate the predicted flows between two regions, let's try `Birmingham` and `Warwick`.


```python
birmingham_warwick=df[(df['Residence']=='Birmingham') & (df['Workplace']=='Warwick')]
birmingham_warwick
```

#### 🤨 TASK
can you do this for another pair of origin and destinations? What do you notice when you choose two smaller cities vs two larger cities? 

*Replace the `???` with your answer...*


```python
???
```

We can also plot a scatter between the **observed** and the **predicted** flow as a visual indication of how well our model predicted the outcome variable - that is, commuting flow between two destinations. In an ideal case scenario, we would want our model to predict accurately, and that would be indicated by points that follow a diagonal line that goes from bottom-left (0,0) to top-right (25000,25000) - meaning that `observed_flow` **equals** `predicted_flow`.


```python
yhat=pd.DataFrame(pysal_model.yhat,columns=['yhat'])
y=pd.DataFrame(pysal_model.y,columns=['y'])
pd.concat([yhat,y],axis=1).plot.scatter('y','yhat',c='black')
```

#### Interpreting the coefficients



```python
# here are the coefficients,pvalue,standard errors of the model.
# pySal regression coefficients 
names=['const', 'O', 'D', 'Distance']
print (f'pysal coefficients: {list(zip(names,pysal_model.params))}')
print (f'stderror: {pysal_model.std_err}')
print (f'pvalue: {pysal_model.pvalues}')
```


```python
obs = birmingham_warwick['Commuters'].values[0]
pred = np.exp((np.log(1092330)*0.3953) + (np.log(138462)*0.69) - (28000*0.0001) - 3.358)
print (f'the predicted commuter flows: {int(pred)}')
print (f'the observed commuter flows: {obs}')
```

It looks like all three variables (origin, destination, impedance) are statistically significant and its sign are positive(+) for the origin and destination population features and negative(-) for the distance features. 

This just means commuting flow is positively correlated to the size of the origin and destination population and negatively correlated to the distance or separation between them. 

**Further interpretation**

To further interpret the coefficients(Oshan 2016), the distance decay coefficient $\beta=-0.0001$ can be interpreted by;

$pred=obs*exp(\beta)$

If we assume an observed flow of 2500, for a 1 unit increase in distance, we expect the number of predicted flows to decrease to 2499.75 (by approximately 0.25 or 0.01%). 



```python
np.round(2500*np.exp(-0.0001),3)
```

since the other two variables are logged, these are often interpreted as percentage change in the predicted response. If we increase the origin population by 1%, we can expect the commuter flow to be increased by  0.39% and similarly if we increase the destination population by 1%, we can expect the commuter flow to be increased by 0.69%, holding all factors constant. 

**statistical test for equidispersion**

It is important to test the models for violations of the equidispersion assumption of Poisson regression as commuting flows are often overdispersed. There is a function in `spint` that computes this. 



```python
from spint.dispersion import phi_disp
#test the hypotehsis of equidispersion (var[mu] = mu) 
Results = phi_disp(pysal_model)#[phi, tvalue, pvalue]
# print the pvalue for the hypothesis test
print (f'pvalue = {Results[2]} ')
```

It looks like the null hypothesis of equidispersion is rejected, meaning the data is over-dispersed. There are different GLM that accounts for over-dispersed data. One of them is a Quasi Poisson regression which is an arguement in the Gravity model function. Here are the coefficients with `Quasi=True`. What do you noticed?


```python
#Fit the same model using a QuasiPoisson framework
Quasi = Gravity(flows, o_vars, d_vars, cost, cost_func='exp', Quasi=True)
print (f'number of params: {len(Quasi.params)}')
names=['const', 'O', 'D', 'Distance']
# here are the coefficients,pvalue,standard errors of the model.
print (f'Quasi coef: {list(zip(names,Quasi.params))}') # same coefficients
print (f'Quasi stderror: {Quasi.std_err}') # larger std errors
print (f'Quasi pvalue: {Quasi.pvalues}') # larger pvalues
```

Here are the original coefficients without `Quasi=True`.


```python
# here are the coefficients,pvalue,standard errors of the model.
pysal_model = Gravity(flows, o_vars, d_vars, cost, cost_func='exp') 
names=['const', 'O', 'D', 'Distance']
print (f'number of params: {len(pysal_model.params)}')
print (f'pysal coefficients: {list(zip(names,pysal_model.params))}')
print (f'stderror: {pysal_model.std_err}')
print (f'pvalue: {pysal_model.pvalues}')
```

## What does this all mean?

Having fitted the parameters of the model? It means that if you are given the size of these origin and destination as well as the distance or impedance such as travel time between it, you would be able to estimate the distribution of commuting flows between places in the UK.

Such estimated flows can be used to help planners to allocate transport infrastructure or for retailer, how much shoppers are they expected to receive or which district to put their shops? It also acts as an important component for the four stage transportation model (Trip generation, Trip distribution, Mode choice, Route choice) namely the trip distribution stage. 




## Alternative package: Estimating gravity model with Statsmodel

We have also provided some information below on estimating a similar regression model using the Python package ```Statsmodel```, a statistical inference package in Python that has many built-in function that can run different types of generalise linear models for estimating a gravity model. [Click here for documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM)


```python
class statsmodels.genmod.generalized_linear_model.GLM(endog, exog, family=None, offset=None, exposure=None, freq_weights=None, var_weights=None, missing='none', **kwargs)
```



```python
#imports statsmodels package
import statsmodels.api as sm
```


```python
# with statsmodel you would need to log transform both the origin and destination population
df['log_O'] = np.log(df['O_Pop'])
df['log_D'] = np.log(df['D_Pop'])
```


```python
#creates object 'X' as a subset of the merged 'df' object, which contains
#the 'O' population, 'D' population and the distance of separation between 'O' and 'D'.
X = df[['log_O','log_D','Distance']]
X = sm.add_constant(X) #adds a constant value for the linear regression. Adding constant is required in statsmodel.
y = df['Commuters'] #and creates 'y' which contains only the 'Commuters' column of the merged 'df' data frame
```


```python
# estimating a poisson regression in statsmodel. 
model = sm.GLM(y, X, family=sm.families.Poisson())
# fits the model.
stm_model = model.fit()
```

Here we print the goodness of fit and regression coefficients which are largely the same. 


```python
# pySal goodness of fit (AIC)
print ('AIC with SpInt: '+ str(pysal_model.AIC))

# stastmodel goodness of fit (AIC)
print ('AIC with Statsmodel: '+ str(stm_model.aic))
```


```python
# pySal regression coefficients 
print (f'pysal coefficients: {list(zip(X.columns,pysal_model.params))}')

#statsmodel regression coefficients
print (f'statsmodel coefficients: {list(zip(stm_model.params.keys(),stm_model.params.values))}')
```

Lab Exercise 4
-------------------------------


### 4.1. Spatial Interaction Model Applied to Manchester Region

The first of two exercises is to apply a similar gravity model as we have for Birmingham, for the Manchester Region. 


**a)** Read in both the commuter flows dataset (`UK_Flows.csv`) and the locational data of places in the Manchester Region (`UK_Manchester_pts.csv`) dataset. <br/>
**b)** Merge the two data frames that you read into Jupyter in part (a) such that the merged data frame includes the x- and y-coordinates of the residence and workplace (origin and destination) pairs. <br/>

```python
[source_X, source_Y, target_X, target_Y]
```
**c)** Visualise the weighted (based on `commuters`) degree of the commuting flow dataset around Manchester. <br/>
**d)** Find the top 10 local authorities, in terms of degree (commuter flow), in the Manchester region (i.e. those with the highest degrees). <br/>
**e)** Find the bottom 10 local authorities, in terms of degree (commuter flow), in the Manchester region (i.e. those with the lowest degrees). <br/>
**f)** Run a gravity model using the `SpInt` (Gravity) package in `PySal`. Are the coefficients the same between Manchester and Birmingham using the exact same functional form? Are the result for the model that you chose as you expected?

#### a) Read in both the commuter flows dataset ('UK_Flows.csv') and the locational data of places in the Manchester Region ('UK_Manchester_pts.csv') dataset.

#### b) Merge the two data frames that you read into Jupyter in part (a) such that the merged data frame includes the x- and y-coordinates of the residence and workplace (origin and destination) pairs.

#### c) Visualise the weighted (based on 'commuters') degree of the commuting flow dataset around Manchester. 

#### d) Find the top 10 local authorities, in terms of degree (commuter flow), in the Manchester region (i.e. those with the highest degrees).

#### e) Find the bottom 10 local authorities, in terms of degree (commuter flow), in the Manchester region (i.e. those with the lowest degrees).

#### f) Run a gravity model using the `SpInt` (Gravity) package in `PySal`. Are the coefficients the same between Manchester and Birmingham using the exact same functional form?
