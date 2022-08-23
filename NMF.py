import pandas as pd
import sys
from surprise import Dataset
from surprise import NMF
from surprise import Reader


# Main program
if __name__ == '__main__':

    # Read csv into a pandas dataframe
    df = pd.read_csv("train_rating.txt", sep="," )

    # Delete unused columns

    del df['date']


    # Set the rating scale and create the data for Surprise to use
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader)

    # Cross validation for tuning
    # Split in 5 folds
    # data.split(5)

    # This part is to use all the data to train and get the output
    train_set = data.build_full_trainset()

    # Use NMF with surprise
    algo = NMF()
    algo.fit(train_set)

    f = open('NMFOutput.csv','w')
    f.write("test_id,rating\n")
    dfTest = pd.read_csv("test_rating.txt", sep=",")
    for i in range(len(dfTest)) :
        prediction = algo.predict(df.at[i,'user_id'],df.at[i,'business_id'],r_ui=4,verbose=True)
        predRating = prediction.est
        f.write(str(i)+","+str(predRating)+'\n')

    f.close()
