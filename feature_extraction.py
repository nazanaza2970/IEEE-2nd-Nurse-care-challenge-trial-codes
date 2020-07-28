import numpy as np
import pandas as pd
import scipy.stats as sp

df = pd.read_pickle('D:\Study\Training\Lab\Copy of labeled_lab.pkl',compression='gzip')

def feature_extraction(df):
    features=[]
    meanX = df['X'].apply(np.mean)
    features.append(meanX)
    meanY = df['Y'].apply(np.mean)
    features.append(meanY)
    meanZ = df['Z'].apply(np.mean)
    features.append(meanZ)
    stdX = df['X'].apply(np.std)
    features.append(stdX)
    stdY = df['Y'].apply(np.std)
    features.append(stdY)
    stdZ = df['Z'].apply(np.std)
    features.append(stdZ)
    madX = df['X'].apply(sp.median_absolute_deviation)
    features.append(madX)
    madY = df['Y'].apply(sp.median_absolute_deviation)
    features.append(madY)
    madZ = df['Z'].apply(sp.median_absolute_deviation)
    features.append(madZ)
    minX = df['X'].apply(min)
    features.append(minX)
    minY = df['Y'].apply(min)
    features.append(minY)
    minZ = df['Z'].apply(min)
    features.append(minZ)
    maxX = df['X'].apply(max)
    features.append(maxX)
    maxY = df['Y'].apply(max)
    features.append(maxY)
    maxZ = df['Z'].apply(max)
    features.append(maxZ)
    def energy(a):
        return sum(a**2)
    def entropy(a):
        a = (a-min(a))/(max(a)-min(a))
        return sp.entropy(a)
    energyX = df['X'].apply(energy)
    features.append(energyX)
    energyY = df['Y'].apply(energy)
    features.append(energyY)
    energyZ = df['Z'].apply(energy)
    features.append(energyZ)
    iqrX = df['X'].apply(sp.iqr)
    features.append(iqrX)
    iqrY = df['Y'].apply(sp.iqr)
    features.append(iqrY)
    iqrZ = df['Z'].apply(sp.iqr)
    features.append(iqrZ)  
    entropyX = df['X'].apply(entropy)
    features.append(entropyX)
    entropyY = df['Y'].apply(entropy)
    features.append(entropyY)
    entropyZ = df['Z'].apply(entropy)
    features.append(entropyZ)
    #corrXY,pValXY = df.apply(lambda x:sp.pearsonr(x.X,x.Y),axis = 1 )
    #corrYZ,pValYZ = df.apply(lambda x:sp.pearsonr(x.Y,x.Z),axis = 1 )
    #corrZX,pValZX = df.apply(lambda x:sp.pearsonr(x.Z,x.X),axis = 1 )
    #features.append([corrXY,corrYZ,corrZX,pValXY,pValYZ,pValZX])               
    
    
    print(len(features))
    featureDF = pd.DataFrame(features)
    featureDF = featureDF.transpose()
    column1 = ['meanX','meanY','meanZ','stdX','stdY','stdZ','madX','madY','madZ','minX','minY','minZ','maxX','maxY','maxZ']
    column2 = ['energyX','energyY','energyZ','iqrX','iqrY','iqrZ','entropyX','enttropyY','entropyZ']
    #column3 = ['corrXY','corrYZ','corrZX','pVAlXY','pValYZ','pValZX']
    featureDF.columns = column1+column2#+column3

    
    return featureDF

featureDf = feature_extraction(df)
    