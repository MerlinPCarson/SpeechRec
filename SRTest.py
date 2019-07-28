from SRPredictor import Predictor 

def main():

    predictor = Predictor() 

    prompt = 'Y'
    while prompt == 'Y':
        predictor.record_and_predict()
        print('Would you like to go again?(y/n): ', end="")
        prompt = input().upper()

if __name__ == '__main__':
    main()
