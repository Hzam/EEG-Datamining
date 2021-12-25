<h2 align="center"><code> 🧠 Electroencephalogram (EEG) Data Mining </code></h2>
<h5 align="center">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; -- monitor patients’ real-time attention to track their recovery
</h5>
<br>



<p align="center">
    <img src="https://github.com/FelixLin99/EEG-Datamining/raw/main/pic/illustration/1.jpg" 
         width="80%">
</p>

<p align="center">"<i> Get to know more about brain signal! </i>"</p>

<br>
<div align="center">
  <sub>Created by
  <a href="https://github.com/FelixLin99/">@Shuhui</a>
</div>

***
# Introduction
- We are trying to monitor patients’ real-time attention to track their recovery
- We design our own experiment paradigm, build data-preprocessing pipeline and train ML model!
- Still in progress...
# Environment
- `Python` == 3.8.0
- `Keras` == 2.6.0
<br>

# Usage
To use model, place the contents of this folder in your PYTHONPATH environment variable.
<br>

Use [models_of_Felix.py](https://github.com/FelixLin99/EEG-Datamining/models_of_Felix.py) to try CNN-Attention-LSTM-ATTENTION model：
```python
from models_of_Felix import ACRNN_4D

ACRNN_object = ACRNN_4D(input_shape, class_num)
model = ACRNN_object.build_model_withAttention()
model.compile(...)
```

Use [test_EEGTCNET.py](https://github.com/FelixLin99/EEG-Datamining/test_EEGTCNET.py) to try EEGNET-TCNET-FUSION model
<br><br>

# How to preprocess EEG data?
<div align=center>
  <img src="https://github.com/FelixLin99/EEG-Datamining/raw/main/pic/illustration/2.jpg" height=180>
    </div>
    
You could find more details in [preprocessedPipeline.py](https://github.com/FelixLin99/EEG-Datamining/preprocessedPipeline.py)
<br><br>

# How to do feature extraction?
<div align=center>
  <img src="https://github.com/FelixLin99/EEG-Datamining/raw/main/pic/illustration/3.jpg" height=300>
    </div>

# About our experiment paradigm
<div align=center>
  <img src="https://github.com/FelixLin99/EEG-Datamining/raw/main/pic/illustration/4.jpg" height=300>
    </div>

  



    
    
    
