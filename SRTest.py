import sys
from SRPredictor import Predictor 

def main():

    # name of modelfile as argument
    modelfile = sys.argv[1]

    predictor = Predictor(modelfile) 

    prompt = 'Y'
    while prompt == 'Y':
        predictor.record_and_predict()
        print('Would you like to go again?(y/n): ', end="")
        prompt = input().upper()

if __name__ == '__main__':
    main()
