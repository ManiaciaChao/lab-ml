## Voice Gender

Gender Recognition by Voice and Speech Analysis

This database was created to identify a voice as male or female,  based upon acoustic properties of the voice and speech. The dataset  consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range  of 0hz-280hz ([human vocal range](https://en.wikipedia.org/wiki/Voice_frequency#Fundamental_frequency)).

## The Dataset

The following acoustic properties of each voice are measured and included within the CSV:

- **meanfreq**: mean frequency (in kHz) 频率均值
- **sd**: standard deviation of frequency 频率标准差
- **median**: median frequency (in kHz) 频率中位数
- **Q25**: first quantile (in kHz) 频率四分之一位点
- **Q75**: third quantile (in kHz) 频率四分之三位点
- **IQR**: interquantile range (in kHz) 频率四分位距 = Q75 - Q25
- **skew**: skewness (see note in specprop description)
- **kurt**: kurtosis (see note in specprop description)
- **sp.ent**: spectral entropy 谱熵
- **sfm**: spectral flatness 频谱平坦度
- **mode**: mode frequency 模频率
- **centroid**: frequency centroid (see specprop)
- **peakf**: peak frequency (frequency with highest energy) 尖峰频率
- **meanfun**: average of fundamental frequency measured across acoustic signal 基频均值
- **minfun**: minimum fundamental frequency measured across acoustic signal 基频最小值
- **maxfun**: maximum fundamental frequency measured across acoustic signal 基频最大值
- **meandom**: average of dominant frequency measured across acoustic signal 主频均值
- **mindom**: minimum of dominant frequency measured across acoustic signal 主频最小值
- **maxdom**: maximum of dominant frequency measured across acoustic signal 主频最大值
- **dfrange**: range of dominant frequency measured across acoustic signal 主频范围
- **modindx**: modulation index. Calculated as the  accumulated absolute difference between adjacent measurements of  fundamental frequencies divided by the frequency range
- **label**: male or female

## Accuracy

### Baseline (always predict male)

50% / 50%

### Logistic Regression

97% / 98%

### CART

96% / 97%

### Random Forest

100% / 98%

### SVM

100% / 99%

### XGBoost

100% / 99%

## Research Questions

An original analysis of the data-set can be found in the following article: 

[Identifying the Gender of a Voice using Machine Learning](http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/)

The best model achieves 99% accuracy on the test set. According to a  CART model, it appears that looking at the mean fundamental frequency  might be enough to accurately classify a voice. However, some male  voices use a higher frequency, even though their resonance differs from  female voices, and may be incorrectly classified as female. To the human ear, there is apparently more than simple frequency, that determines a  voice's gender.

### Questions

- What other features differ between male and female voices?
- Can we find a difference in resonance between male and female voices?
- Can we identify falsetto from regular voices? (separate data-set likely needed for this)
- Are there other interesting features in the data?

### CART Diagram

![CART model](http://i.imgur.com/Npr2U7O.png)

Mean fundamental frequency appears to be an indicator of voice  gender, with a threshold of 140hz separating male from female  classifications.

## References

[The Harvard-Haskins Database of Regularly-Timed Speech](http://www.nsi.edu/~ani/download.html)

[Telecommunications & Signal Processing Laboratory (TSP) Speech Database at McGill University](http://www-mmsp.ece.mcgill.ca/Documents../Downloads/TSPspeech/TSPspeech.pdf), [Home](http://www-mmsp.ece.mcgill.ca/Documents../Data/index.html)

[VoxForge Speech Corpus](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit/), [Home](http://www.voxforge.org)

[Festvox CMU_ARCTIC Speech Database at Carnegie Mellon University](http://festvox.org/cmu_arctic/) 