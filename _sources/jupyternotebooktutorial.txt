.. _jupyternotebooktutorial:

*******************
SEQ_TABLES TUTORIAL
*******************

.. code:: python

    """
    Seq tables will take in sequences and quality scores and put them into dataframe. In addition, each sequence
    of tables in which each row is a specific sequence and each column is a specific position within the sequence.
    this allows useful slicing and determination of hamming distance between sequences
    """

.. code:: python

    import sys
    import seq_tables
    from plotly.offline import iplot, init_notebook_mode
    import numpy as np
    import pandas as pd
    from plotly import graph_objs as go

.. code:: python

    %load_ext autoreload
    %autoreload 2

.. code:: python

    init_notebook_mode()

.. code:: python

    """
    There are multiple ways to initialize this class. Essentially it just needs a list of aligned sequences to one another.
    If you would like to you could also past in a list of quality scores
    """

.. code:: python

    seqs = ['AAA', 'ACT', 'ACA']
    quals = ['ZCA', 'DGJ', 'JJJ']
    sq = seq_tables.seqtable(seqs, quals)

.. code:: python

    # This class has three important attributes
    # 1) sq.seq_df => returns a dataframe of the original sequence and/or quality
    sq.seq_df




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>seqs</th>
          <th>quals</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>AAA</td>
          <td>ZCA</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ACT</td>
          <td>DGJ</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ACA</td>
          <td>JJJ</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # 2) sq.seq_table => this represents all letters in the sequences as a table of rows/columns of ascii values
    sq.seq_table




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>65</td>
          <td>65</td>
          <td>65</td>
        </tr>
        <tr>
          <th>1</th>
          <td>65</td>
          <td>67</td>
          <td>84</td>
        </tr>
        <tr>
          <th>2</th>
          <td>65</td>
          <td>67</td>
          <td>65</td>
        </tr>
      </tbody>
    </table>
    </div>

.. code:: python

    # if you would like, you could view the letters as actual letters instead of ascii
    sq.view_bases()

.. parsed-literal::

    array([['A', 'A', 'A'],
           ['A', 'C', 'T'],
           ['A', 'C', 'A']], 
          dtype='|S1')



.. code:: python

    # Now we can slice select positions
    sq.view_bases()[:, 1:3]




.. parsed-literal::

    array([['A', 'A'],
           ['C', 'T'],
           ['C', 'A']], 
          dtype='|S1')



.. code:: python

    # 3) sq.qual_table => this represents the qualitysequence as actual quality bases
    sq.qual_table




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>57</td>
          <td>34</td>
          <td>32</td>
        </tr>
        <tr>
          <th>1</th>
          <td>35</td>
          <td>38</td>
          <td>41</td>
        </tr>
        <tr>
          <th>2</th>
          <td>41</td>
          <td>41</td>
          <td>41</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    """
    Now lets actualy load real data from a fastq file
    """
    sq = seq_tables.read_fastq("test/Demo.fq")

.. code:: python

    sq.view_bases()




.. parsed-literal::

    array([['G', 'N', 'C', ..., 'T', 'G', 'A'],
           ['G', 'G', 'A', ..., 'T', 'C', 'C'],
           ['C', 'G', 'T', ..., 'A', 'A', 'C'],
           ..., 
           ['G', 'G', 'A', ..., 'A', 'G', 'C'],
           ['G', 'C', 'C', ..., 'A', 'C', 'A'],
           ['G', 'G', 'A', ..., 'G', 'C', 'C']], 
          dtype='|S1')



.. code:: python

    sq.seq_table.shape




.. parsed-literal::

    (3000, 250)



.. code:: python

    """
    We can get a distribution of the quality scores in the data
    """
    (dist, plotsmade) = sq.get_quality_dist()
    dist




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10-14</th>
          <th>...</th>
          <th>205-209</th>
          <th>210-214</th>
          <th>215-219</th>
          <th>220-224</th>
          <th>225-229</th>
          <th>230-234</th>
          <th>235-239</th>
          <th>240-244</th>
          <th>245-249</th>
          <th>250-254</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>15000.000000</td>
          <td>...</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>15000.000000</td>
          <td>3000.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>32.461333</td>
          <td>32.855000</td>
          <td>33.074667</td>
          <td>33.214000</td>
          <td>32.972000</td>
          <td>32.285000</td>
          <td>32.454667</td>
          <td>32.358333</td>
          <td>32.924000</td>
          <td>33.957000</td>
          <td>...</td>
          <td>32.416067</td>
          <td>32.170867</td>
          <td>31.733000</td>
          <td>31.514533</td>
          <td>30.088267</td>
          <td>29.954067</td>
          <td>30.340867</td>
          <td>30.250333</td>
          <td>30.004200</td>
          <td>22.639667</td>
        </tr>
        <tr>
          <th>std</th>
          <td>3.007134</td>
          <td>2.527338</td>
          <td>2.031042</td>
          <td>1.773486</td>
          <td>2.330313</td>
          <td>4.571879</td>
          <td>2.896678</td>
          <td>3.010695</td>
          <td>2.215821</td>
          <td>4.554324</td>
          <td>...</td>
          <td>8.113011</td>
          <td>8.209306</td>
          <td>8.263074</td>
          <td>8.380561</td>
          <td>9.213078</td>
          <td>9.250358</td>
          <td>9.030052</td>
          <td>9.235223</td>
          <td>9.219869</td>
          <td>9.778839</td>
        </tr>
        <tr>
          <th>min</th>
          <td>16.000000</td>
          <td>2.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>16.000000</td>
          <td>15.000000</td>
          <td>...</td>
          <td>12.000000</td>
          <td>2.000000</td>
          <td>2.000000</td>
          <td>2.000000</td>
          <td>12.000000</td>
          <td>2.000000</td>
          <td>2.000000</td>
          <td>12.000000</td>
          <td>12.000000</td>
          <td>12.000000</td>
        </tr>
        <tr>
          <th>10%</th>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>30.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>...</td>
          <td>14.000000</td>
          <td>14.000000</td>
          <td>13.000000</td>
          <td>13.000000</td>
          <td>13.000000</td>
          <td>13.000000</td>
          <td>14.000000</td>
          <td>13.000000</td>
          <td>14.000000</td>
          <td>12.000000</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>33.000000</td>
          <td>...</td>
          <td>32.000000</td>
          <td>32.000000</td>
          <td>31.000000</td>
          <td>30.000000</td>
          <td>26.000000</td>
          <td>24.000000</td>
          <td>25.000000</td>
          <td>25.000000</td>
          <td>24.000000</td>
          <td>14.000000</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>33.000000</td>
          <td>34.000000</td>
          <td>...</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>35.000000</td>
          <td>35.000000</td>
          <td>36.000000</td>
          <td>36.000000</td>
          <td>35.000000</td>
          <td>15.000000</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>38.000000</td>
          <td>...</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>33.000000</td>
        </tr>
        <tr>
          <th>90%</th>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>35.000000</td>
          <td>34.000000</td>
          <td>35.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>34.000000</td>
          <td>38.000000</td>
          <td>...</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
        </tr>
        <tr>
          <th>max</th>
          <td>36.000000</td>
          <td>36.000000</td>
          <td>36.000000</td>
          <td>36.000000</td>
          <td>36.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>37.000000</td>
          <td>38.000000</td>
          <td>...</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>39.000000</td>
          <td>38.000000</td>
        </tr>
      </tbody>
    </table>
    <p>10 rows × 57 columns</p>
    </div>



.. code:: python

    """
    The function get_quality_dist also returns plot objects that can be used by plotly
    """
    iplot(plotsmade)


.. code:: python

    """
    Filter out low quality sequences
    """
    sq.quality_filter(q=20, p=50)




.. parsed-literal::

                                                                                                     seqs  \
    M01012:51:000000000-A4H0H:1:1101:15861:1495 1:N...  GNCGTGACGTTGGACGAGTCCGGGGGCGGCCTCCAGACGCCCGGAG...   
    M01012:51:000000000-A4H0H:1:1101:16832:1547 1:N...  GGAGGAGACGATGACTTCGGTCCCGTGGCCCCATGCGTCGATACAG...   
    M01012:51:000000000-A4H0H:1:1101:14519:1548 1:N...  CGTGACGTTGGACGAGTCCGGGGGCGGCCTCCAGACGCCCGGAGGA...   
    
                                                                                                    quals  
    M01012:51:000000000-A4H0H:1:1101:15861:1495 1:N...  B#>AABCCCCCCGGGGGGGGGGGGGGEGGGGGHHHHHGGGGGGGGG...  
    M01012:51:000000000-A4H0H:1:1101:16832:1547 1:N...  A@ABB@A>AADAGGGGGGGGFGGHGH2EEBGEEHHHGGGGGFGGGH...  
    M01012:51:000000000-A4H0H:1:1101:14519:1548 1:N...  AAAAAA>>>>>1BE1AEBFFGG0EEGCGEFCC>FGGGG/EEEEGGG...  
    
    [2963 rows x 2 columns]



.. code:: python

    """
    Convert any low quality bases to N
    """
    sq.convert_low_bases_to_null(q=20)




.. parsed-literal::

                                                                                                     seqs  \
    M01012:51:000000000-A4H0H:1:1101:15861:1495 1:N...  GNCGTGACGTTGGACGAGTCCGGGGGCGGCCTCCAGACGCCCGGAG...   
    M01012:51:000000000-A4H0H:1:1101:16832:1547 1:N...  GGAGGAGACGATGACTTCGGTCCCGTNGCCCCATGCGTCGATACAG...   
    M01012:51:000000000-A4H0H:1:1101:14519:1548 1:N...  CGTGACGTTGGNCGNGTCCGGGNGCGGCCTCCAGACGCNCGGAGGA...   
    
                                                                                                    quals  
    M01012:51:000000000-A4H0H:1:1101:15861:1495 1:N...  B#>AABCCCCCCGGGGGGGGGGGGGGEGGGGGHHHHHGGGGGGGGG...  
    M01012:51:000000000-A4H0H:1:1101:16832:1547 1:N...  A@ABB@A>AADAGGGGGGGGFGGHGH2EEBGEEHHHGGGGGFGGGH...  
    M01012:51:000000000-A4H0H:1:1101:14519:1548 1:N...  AAAAAA>>>>>1BE1AEBFFGG0EEGCGEFCC>FGGGG/EEEEGGG...  
    
    [3000 rows x 2 columns]



.. code:: python

    """
    Get the hamming distances of all sequences to the first sequence
    """
    sq.hamming_distance(sq.seq_df.iloc[0]['seqs'])




.. parsed-literal::

    M01012:51:000000000-A4H0H:1:1101:15861:1495 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f1::552c261b9eb6363a487b62c9"}      0
    M01012:51:000000000-A4H0H:1:1101:16832:1547 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f2::552c261b9eb6363a487b62c9"}    189
    M01012:51:000000000-A4H0H:1:1101:14519:1548 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f3::552c261b9eb6363a487b62c9"}    190
    M01012:51:000000000-A4H0H:1:1101:14011:1558 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f4::552c261b9eb6363a487b62c9"}     43
    M01012:51:000000000-A4H0H:1:1101:16697:1561 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f5::552c261b9eb6363a487b62c9"}    197
    M01012:51:000000000-A4H0H:1:1101:14388:1594 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f6::552c261b9eb6363a487b62c9"}    189
    M01012:51:000000000-A4H0H:1:1101:16245:1601 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f7::552c261b9eb6363a487b62c9"}    196
    M01012:51:000000000-A4H0H:1:1101:14065:1607 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f8::552c261b9eb6363a487b62c9"}     50
    M01012:51:000000000-A4H0H:1:1101:15481:1612 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950f9::552c261b9eb6363a487b62c9"}     92
    M01012:51:000000000-A4H0H:1:1101:13999:1617 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950fa::552c261b9eb6363a487b62c9"}     91
    M01012:51:000000000-A4H0H:1:1101:14577:1618 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950fb::552c261b9eb6363a487b62c9"}    103
    M01012:51:000000000-A4H0H:1:1101:16257:1622 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950fc::552c261b9eb6363a487b62c9"}    197
    M01012:51:000000000-A4H0H:1:1101:16266:1647 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950fd::552c261b9eb6363a487b62c9"}    194
    M01012:51:000000000-A4H0H:1:1101:16222:1647 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950fe::552c261b9eb6363a487b62c9"}    194
    M01012:51:000000000-A4H0H:1:1101:15672:1648 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c950ff::552c261b9eb6363a487b62c9"}     43
    M01012:51:000000000-A4H0H:1:1101:16537:1664 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c95100::552c261b9eb6363a487b62c9"}     47
    M01012:51:000000000-A4H0H:1:1101:16542:1690 1:N:0:10| <{"SEQ_ID": "552c2da19eb63635e1c95101::552c261b9eb6363a487b62c9"}    194
    dtype: int64



.. code:: python

    """
    Or we determine the error rate at each position
    """
    error = sq.compare_to_reference(sq.seq_df.iloc[0]['seqs']).sum()/sq.seq_table.shape[0]
    iplot([go.Scatter(x = error.index, y = error, mode='markers')])



.. raw:: html

    <div id="dff6e7c3-4f5c-4f44-9a9f-acba040ed686" style="height: 525; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) {window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("dff6e7c3-4f5c-4f44-9a9f-acba040ed686", [{"y": [0.9393333333333334, 0.0013333333333333333, 0.5096666666666667, 0.934, 0.508, 0.5173333333333333, 0.5206666666666667, 0.5126666666666667, 0.5256666666666666, 0.511, 0.5113333333333333, 0.514, 0.9183333333333333, 0.9126666666666666, 0.9166666666666666, 0.515, 0.5093333333333333, 0.5116666666666667, 0.5053333333333333, 0.516, 0.507, 0.524, 0.515, 0.5196666666666667, 0.9206666666666666, 0.5153333333333333, 0.5146666666666667, 0.924, 0.5196666666666667, 0.9253333333333333, 0.9346666666666666, 0.5103333333333333, 0.523, 0.524, 0.528, 0.5033333333333333, 0.517, 0.5096666666666667, 0.5056666666666667, 0.5173333333333333, 0.51, 0.5316666666666666, 0.6553333333333333, 0.5636666666666666, 0.5896666666666667, 0.5543333333333333, 0.5673333333333334, 0.5416666666666666, 0.4553333333333333, 0.48133333333333334, 0.526, 0.5146666666666667, 0.6536666666666666, 0.658, 0.81, 0.5736666666666667, 0.614, 0.5783333333333334, 0.6016666666666667, 0.667, 0.523, 0.0003333333333333333, 0.6703333333333333, 0.5196666666666667, 0.5836666666666667, 0.641, 0.8446666666666667, 0.5486666666666666, 0.536, 0.564, 0.633, 0.6773333333333333, 0.5466666666666666, 0.6643333333333333, 0.6576666666666666, 0.0003333333333333333, 0.0003333333333333333, 0.5116666666666667, 0.5836666666666667, 0.6246666666666667, 0.639, 0.49733333333333335, 0.5276666666666666, 0.6383333333333333, 0.5296666666666666, 0.6343333333333333, 0.624, 0.5626666666666666, 0.5756666666666667, 0.48533333333333334, 0.4706666666666667, 0.5233333333333333, 0.49366666666666664, 0.411, 0.42533333333333334, 0.384, 0.06266666666666666, 0.23433333333333334, 0.5996666666666667, 0.5043333333333333, 0.6686666666666666, 0.49533333333333335, 0.21066666666666667, 0.22766666666666666, 0.13566666666666666, 0.5053333333333333, 0.652, 0.5253333333333333, 0.6473333333333333, 0.0003333333333333333, 0.537, 0.634, 0.622, 0.44066666666666665, 0.632, 0.5993333333333334, 0.606, 0.7626666666666667, 0.5226666666666666, 0.56, 0.6666666666666666, 0.5573333333333333, 0.5513333333333333, 0.6566666666666666, 0.515, 0.498, 0.5236666666666666, 0.6816666666666666, 0.721, 0.5906666666666667, 0.5133333333333333, 0.7036666666666667, 0.0003333333333333333, 0.5946666666666667, 0.11533333333333333, 0.5793333333333334, 0.0003333333333333333, 0.25233333333333335, 0.5793333333333334, 0.31866666666666665, 0.0003333333333333333, 0.593, 0.6046666666666667, 0.5566666666666666, 0.6246666666666667, 0.654, 0.4026666666666667, 0.11866666666666667, 0.239, 0.0006666666666666666, 0.49833333333333335, 0.0006666666666666666, 0.638, 0.095, 0.3406666666666667, 0.325, 0.37166666666666665, 0.27, 0.11033333333333334, 0.49766666666666665, 0.301, 0.6803333333333333, 0.7003333333333334, 0.6266666666666667, 0.647, 0.3943333333333333, 0.6026666666666667, 0.5843333333333334, 0.30033333333333334, 0.43833333333333335, 0.46266666666666667, 0.4786666666666667, 0.5553333333333333, 0.0006666666666666666, 0.23266666666666666, 0.3233333333333333, 0.21233333333333335, 0.4076666666666667, 0.0006666666666666666, 0.57, 0.513, 0.242, 0.49066666666666664, 0.38366666666666666, 0.5493333333333333, 0.5276666666666666, 0.5466666666666666, 0.5466666666666666, 0.539, 0.5423333333333333, 0.48633333333333334, 0.0003333333333333333, 0.39466666666666667, 0.0003333333333333333, 0.466, 0.546, 0.5946666666666667, 0.5353333333333333, 0.495, 0.5866666666666667, 0.5053333333333333, 0.532, 0.5923333333333334, 0.5943333333333334, 0.6166666666666667, 0.6386666666666667, 0.6433333333333333, 0.655, 0.4876666666666667, 0.0006666666666666666, 0.411, 0.6013333333333334, 0.43133333333333335, 0.6256666666666667, 0.454, 0.44633333333333336, 0.5, 0.5013333333333333, 0.0006666666666666666, 0.0003333333333333333, 0.5286666666666666, 0.0006666666666666666, 0.001, 0.47733333333333333, 0.44333333333333336, 0.43633333333333335, 0.513, 0.445, 0.553, 0.46166666666666667, 0.001, 0.515, 0.4723333333333333, 0.0003333333333333333, 0.0006666666666666666, 0.4736666666666667, 0.5116666666666667, 0.4226666666666667, 0.43133333333333335, 0.5506666666666666, 0.6733333333333333, 0.514, 0.5833333333333334, 0.6886666666666666, 0.44666666666666666, 0.5563333333333333, 0.6396666666666667, 0.508, 0.5516666666666666, 0.417], "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249], "type": "scatter", "mode": "markers"}], {}, {"linkText": "Export to plot.ly", "showLink": true})});</script>


.. code:: python

    """
    Lets say you want to not count specific bases or residues as mismatches
    """
    sq.compare_to_reference(sq.seq_df.iloc[0]['seqs'], ignore_characters=['N']).sum()/sq.seq_table.shape[0]




.. parsed-literal::

    0      0.939333
    1      1.000000
    2      0.509667
    3      0.934000
    4      0.508000
    5      0.517333
    6      0.520667
    7      0.512667
    8      0.525667
    9      0.511000
    10     0.511333
    11     0.514000
    12     0.918333
    13     0.912667
    dtype: float64



.. code:: python

    """
    Lets say you are only interested in looking at sections/regions of a sequence
    """
    sq.slice_sequences([3,10,20], name='sliced')




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sliced</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>M01012:51:000000000-A4H0H:1:1101:15861:1495 1:N:0:10| &lt;{"SEQ_ID": "552c2da19eb63635e1c950f1::552c261b9eb6363a487b62c9"}</th>
          <td>CTC</td>
        </tr>
        <tr>
          <th>M01012:51:000000000-A4H0H:1:1101:16832:1547 1:N:0:10| &lt;{"SEQ_ID": "552c2da19eb63635e1c950f2::552c261b9eb6363a487b62c9"}</th>
          <td>AGG</td>
        </tr>
        <tr>
          <th>M01012:51:000000000-A4H0H:1:1101:14519:1548 1:N:0:10| &lt;{"SEQ_ID": "552c2da19eb63635e1c950f3::552c261b9eb6363a487b62c9"}</th>
          <td>TGG</td>
        </tr>
      </tbody>
    </table>
    <p>3000 rows × 1 columns</p>
    </div>



.. code:: python

    """
    Determine the distribution of letters at each postion
    """
    dist = sq.get_seq_dist()
    dist




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10</th>
          <th>...</th>
          <th>241</th>
          <th>242</th>
          <th>243</th>
          <th>244</th>
          <th>245</th>
          <th>246</th>
          <th>247</th>
          <th>248</th>
          <th>249</th>
          <th>250</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>A</th>
          <td>56.0</td>
          <td>101</td>
          <td>1313.0</td>
          <td>57.0</td>
          <td>107.0</td>
          <td>1293.0</td>
          <td>1562.0</td>
          <td>1327.0</td>
          <td>57.0</td>
          <td>94.0</td>
          <td>...</td>
          <td>363.0</td>
          <td>559.0</td>
          <td>237.0</td>
          <td>238.0</td>
          <td>1340.0</td>
          <td>199.0</td>
          <td>310.0</td>
          <td>546.0</td>
          <td>183.0</td>
          <td>1251.0</td>
        </tr>
        <tr>
          <th>C</th>
          <td>71.0</td>
          <td>1520</td>
          <td>1529.0</td>
          <td>84.0</td>
          <td>44.0</td>
          <td>72.0</td>
          <td>55.0</td>
          <td>1538.0</td>
          <td>1289.0</td>
          <td>59.0</td>
          <td>...</td>
          <td>2020.0</td>
          <td>540.0</td>
          <td>700.0</td>
          <td>2066.0</td>
          <td>619.0</td>
          <td>789.0</td>
          <td>1919.0</td>
          <td>504.0</td>
          <td>823.0</td>
          <td>863.0</td>
        </tr>
        <tr>
          <th>G</th>
          <td>2818.0</td>
          <td>1317</td>
          <td>68.0</td>
          <td>2802.0</td>
          <td>1325.0</td>
          <td>1552.0</td>
          <td>1319.0</td>
          <td>71.0</td>
          <td>1577.0</td>
          <td>1314.0</td>
          <td>...</td>
          <td>410.0</td>
          <td>359.0</td>
          <td>1750.0</td>
          <td>460.0</td>
          <td>309.0</td>
          <td>1669.0</td>
          <td>539.0</td>
          <td>426.0</td>
          <td>1655.0</td>
          <td>655.0</td>
        </tr>
        <tr>
          <th>N</th>
          <td>0.0</td>
          <td>4</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>...</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>T</th>
          <td>55.0</td>
          <td>58</td>
          <td>90.0</td>
          <td>57.0</td>
          <td>1524.0</td>
          <td>83.0</td>
          <td>64.0</td>
          <td>64.0</td>
          <td>77.0</td>
          <td>1533.0</td>
          <td>...</td>
          <td>207.0</td>
          <td>1542.0</td>
          <td>313.0</td>
          <td>236.0</td>
          <td>732.0</td>
          <td>343.0</td>
          <td>232.0</td>
          <td>1524.0</td>
          <td>339.0</td>
          <td>231.0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 250 columns</p>
    </div>


.. code:: python

    """
    have not made functions for drawing up sequence logos yet...but maybe just plot distribution at each position
    """
    # lets put a dot at specific postions of interest
    mark_these_positions = [5, 10, 30]
    dist = dist/dist.sum()
    data = [
        go.Bar(
            x = dist.columns,
            y = dist.loc[let],
            name=let
        )
        for let in ['A', 'C', 'G', 'T']
    ]
    data.append(go.Scatter(x=mark_these_positions, y=[1.1]*len(mark_these_positions), mode='markers'))
    fig = go.Figure(data=data, layout = go.Layout(barmode='stack', bargap=0.6,width=5000, ))
    iplot(fig)
    dist_div = np.exp((-1*dist*np.log(dist)).sum())
    # also plot the "diversity"
    iplot([go.Scatter(x=dist.columns, y=dist_div,  mode='markers'), go.Scatter(x=mark_these_positions, y=[4.1]*len(mark_these_positions), mode='markers')])


.. code:: python

    """
    Return a random subsample of sequences
    """
    sq.subsample(10)




.. parsed-literal::

                                                                                                     seqs  \
    M01012:51:000000000-A4H0H:1:1101:4797:7093 1:N:...  GCCGTGACGTTGGACGAGTCCGGGGGCGGCCTCCAGACGCCCGGAG...   
    M01012:51:000000000-A4H0H:1:1101:16060:5388 1:N...  GCCGTGACGTTGGACGAGTCCGGGGGCGGCCTCCAGACGCCCGGAG...   
    M01012:51:000000000-A4H0H:1:1101:22232:5603 1:N...  GGAGGAGACGATGACTTCGGTCCCGCGGCCCCATGCGTCGATCCAA...   
    
                                                                                                    quals  
    M01012:51:000000000-A4H0H:1:1101:4797:7093 1:N:...  AABBBBBBBBBBGGGGGGGGGGGGGEEGGEGGHHHHHGGGGGGGGG...  
    M01012:51:000000000-A4H0H:1:1101:16060:5388 1:N...  ABBBADBBBBABGGGGGGGGGGCEGFGGGGGGHHCHHGGGGGFCEE...  
    M01012:51:000000000-A4H0H:1:1101:22232:5603 1:N...  BBBBBB>AAA2AFGGGGGGGFFGHGGGGGGGGGHHHFGGEEGHGFH...  
  
